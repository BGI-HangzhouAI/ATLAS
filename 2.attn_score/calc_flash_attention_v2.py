
import argparse
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Any
from types import MethodType

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import math
from transformers.models.mixtral.modeling_mixtral import apply_rotary_pos_emb
import random

# device = torch.device("cuda:0")
captured: Dict[str, torch.Tensor] = {}


########################################################################
## 计算FA attention矩阵相关代码
########################################################################

@torch.no_grad()
def attn_column_sums_rowwise_two_pass(
    q: torch.Tensor,
    k: torch.Tensor,
    causal: bool = True,
    block_rows: int = 256,
):
    """
    - Two-pass + 按行(Query)分块
    - 显存复杂度 O(H * L * block_rows)
    
    q, k: [B, H, L, D]
    return: [B, L]  (所有 head 平均后的列和)
    """

    B, H, L, D = q.shape
    scale = 1.0 / math.sqrt(D)

    q = q.to(torch.float32)
    k = k.to(torch.float32)

    out = torch.zeros((B, L), dtype=torch.float32, device=q.device)

    for b in range(B):
        # K_b: [H, D, L]
        K_b = k[b].transpose(-1, -2)

        # -----------------------------
        # PASS 1: 统计全局 m 和 l
        # -----------------------------
        m = torch.full((H, L), -float("inf"), device=q.device)
        l = torch.zeros((H, L), device=q.device)

        for i0 in range(0, L, block_rows):
            i1 = min(i0 + block_rows, L)

            Q_chunk = q[b, :, i0:i1, :]     # [H, Br, D]
            S = torch.matmul(Q_chunk, K_b) * scale   # [H, Br, L]

            if causal:
                row_idx = torch.arange(i0, i1, device=q.device).view(1, -1, 1)
                col_idx = torch.arange(L, device=q.device).view(1, 1, -1)
                S = S.masked_fill(col_idx > row_idx, float("-inf"))

            block_max = S.max(dim=-1).values        # [H, Br]
            prev_m = m[:, i0:i1]                    # [H, Br]
            new_m = torch.maximum(prev_m, block_max)

            exp_term = torch.exp(S - new_m.unsqueeze(-1))

            l[:, i0:i1] = (
                l[:, i0:i1] * torch.exp(prev_m - new_m)
                + exp_term.sum(dim=-1)
            )

            m[:, i0:i1] = new_m

            del S, exp_term

        # -----------------------------
        # PASS 2: 计算真实概率并累加列和
        # -----------------------------
        col_sum_heads = torch.zeros((H, L), device=q.device)

        for i0 in range(0, L, block_rows):
            i1 = min(i0 + block_rows, L)

            Q_chunk = q[b, :, i0:i1, :]
            S = torch.matmul(Q_chunk, K_b) * scale

            if causal:
                row_idx = torch.arange(i0, i1, device=q.device).view(1, -1, 1)
                col_idx = torch.arange(L, device=q.device).view(1, 1, -1)
                S = S.masked_fill(col_idx > row_idx, float("-inf"))

            P = torch.exp(S - m[:, i0:i1].unsqueeze(-1)) / (
                l[:, i0:i1].unsqueeze(-1) + 1e-12
            )

            col_sum_heads += P.sum(dim=1)

            del S, P

        out[b] = col_sum_heads.mean(dim=0)

    return out

def _get_heads_from_model(model):
    H_q = getattr(getattr(model, "config", object()), "num_attention_heads", None)
    H_kv = getattr(getattr(model, "config", object()), "num_key_value_heads", None)
    return int(H_q), int(H_kv)

def _attach_qk_hooks(last_attn_module, model):
    """
    从 config 读 H_q/H_kv；Q 按 H_q reshape，K 按 H_kv reshape，再把 K 扩展到 H_q。
    捕获到的 q/k 形状最终都为 [B, H_q, L, D]。
    """
    q_linear = getattr(last_attn_module, "q_proj", None)
    k_linear = getattr(last_attn_module, "k_proj", None)
    if q_linear is None or k_linear is None:
        return []

    H_q, H_kv = _get_heads_from_model(model)
    group = H_q // H_kv
    captured["__H_q__"] = torch.tensor(H_q)
    captured["__H_kv__"] = torch.tensor(H_kv)

    hooks = []

    def _grab_q(module, inp, out):
        # out: [B, L, D_all_q] 其中 D_all_q = H_q * D
        B, L, Dall_q = out.shape
        assert Dall_q % H_q == 0, f"Q proj dim {Dall_q} not divisible by H_q={H_q}"
        D = Dall_q // H_q
        q = out.view(B, L, H_q, D).permute(0, 2, 1, 3).contiguous()  # [B,H_q,L,D]
        captured["q_linear"] = q.detach()

    def _grab_k(module, inp, out):
        # out: [B, L, D_all_k] 其中 D_all_k = H_kv * D
        B, L, Dall_k = out.shape
        assert Dall_k % H_kv == 0, f"K proj dim {Dall_k} not divisible by H_kv={H_kv}"
        D = Dall_k // H_kv
        k = out.view(B, L, H_kv, D).permute(0, 2, 1, 3).contiguous()  # [B,H_kv,L,D]
        if H_kv != H_q:
            k = k.repeat_interleave(group, dim=1)  # → [B,H_q,L,D]
        captured["k_linear"] = k.detach()

    hooks.append(q_linear.register_forward_hook(_grab_q))
    hooks.append(k_linear.register_forward_hook(_grab_k))
    return hooks

def _get_rotary(model):
    if hasattr(model, "rotary_emb"):
        return model.rotary_emb
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        return model.model.rotary_emb
    return None

def _apply_rope_if_possible(q, k, rotary=None, position_ids=None):
    """
    尝试用模块自带的 rotary_emb/rope 对 q/k 施加 RoPE。
    失败则直接返回原 q/k（会有偏差但不影响运行）。
    """
    if rotary is None:
        print("未找到 RoPE 模块，跳过 RoPE 应用。")
        return q, k

    cos, sin = rotary(q, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1)
    return q, k
    
def calc_attentions(seq: str, model, tokenizer, device, causal=True, block_rows=1024) -> List[float]:
    """
    只使用 Q/K 路径：
    - 在最后一层 self-attn 的 q_proj/k_proj 上挂钩抓取 Q/K（包含线性投影，尽量补 RoPE）
    - 用分块 softmax 计算所有头平均后的“按列求和”向量
    返回：长度为 L 的 Python list（float）
    """
    captured.clear()

    tokenizer.model_max_length = int(1e9)
    inputs = tokenizer(seq, return_tensors="pt", truncation=False)
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    if "position_ids" in inputs:
        position_ids = inputs["position_ids"]
    else:
        B, L = inputs["input_ids"].shape
        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L).to(device)

    last_attn = model.layers[-1].self_attn
    if last_attn is None:
        raise ValueError("未找到最后一层 self-attention 模块，无法挂钩 Q/K。")

    hooks = _attach_qk_hooks(last_attn, model)
    try:
        model.eval()
        with torch.no_grad():
            _ = model(**inputs)   # 触发前向以捕获 q/k
    finally:
        for h in hooks:
            try: h.remove()
            except Exception: pass

    q = captured.get("q_linear", None)
    k = captured.get("k_linear", None)
    if q is None or k is None:
        raise ValueError("未捕获到 Q/K（可能实现名不匹配或优化融合过深）。")

    # 尝试应用与前向一致的 RoPE
    rotary = _get_rotary(model)
    q, k = q.to(device), k.to(device)
    q, k = _apply_rope_if_possible(q, k, rotary, position_ids)

    # 因果 mask：Causal LM 通常为 True
    col_sums = attn_column_sums_rowwise_two_pass(q, k, causal=causal, block_rows=block_rows)  # [B,L]
    vec = col_sums[0]  # 单样本
    return vec.detach().cpu().float().numpy().tolist()


def calc_attentions_batch(
    seq_chunks: List[str],
    model,
    tokenizer,
    device,
    causal: bool = True,
    block_rows: int = 1024,
) -> List[List[float]]:
    """
    对一组等长的 seq_chunk 一次性 batch 打分，复用 calc_flash_attention_v2.py 里的 Q/K + two-pass 实现。
    要求：所有 seq_chunks 在 tokenizer 后的长度一致（这里我们依赖固定的 chunk_size 来保证）。
    返回：len(seq_chunks) 个 list，每个 list 长度为 L。
    """
    if not seq_chunks:
        return []

    # 清空上次捕获的 Q/K
    captured.clear()

    tokenizer.model_max_length = int(1e9)
    inputs = tokenizer(
        seq_chunks,
        return_tensors="pt",
        truncation=False,
        padding=False,   # 假定 chunk token 长度一致
    )
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # position_ids（和单条序列版本保持一致）
    if "position_ids" in inputs:
        position_ids = inputs["position_ids"]
    else:
        B, L = inputs["input_ids"].shape
        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L).to(device)

    # 挂钩到最后一层 self-attn 的 q_proj / k_proj
    last_attn = model.layers[-1].self_attn
    if last_attn is None:
        raise ValueError("未找到最后一层 self-attention 模块，无法挂钩 Q/K。")

    hooks = _attach_qk_hooks(last_attn, model)
    try:
        model.eval()
        with torch.no_grad():
            _ = model(**inputs)   # 触发前向以捕获 q/k
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    q = captured.get("q_linear", None)
    k = captured.get("k_linear", None)
    if q is None or k is None:
        raise ValueError("未捕获到 Q/K（可能实现名不匹配或优化融合过深）。")

    rotary = _get_rotary(model)
    q, k = q.to(device), k.to(device)
    q, k = _apply_rope_if_possible(q, k, rotary, position_ids)

    # two-pass FA，支持 batch 维度
    col_sums = attn_column_sums_rowwise_two_pass(
        q, k, causal=causal, block_rows=block_rows
    )  # [B, L]

    col_sums = col_sums.detach().cpu().float().numpy()
    return [col_sums[b].tolist() for b in range(col_sums.shape[0])]