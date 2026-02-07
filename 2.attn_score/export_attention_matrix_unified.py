#!/usr/bin/env python3
"""
Unified attention extraction script that automatically selects the appropriate
strategy based on sequence length:

1. Vanilla attention (L ≤ 4,096): Standard eager attention with full matrix materialization
2. FlashAttention-based (4,096 < L ≤ 131,072): Block-wise computation without full matrix
3. Chunked processing (L > 131,072): Sliding-window chunking with overlap handling

Usage:
    python export_attention_matrix_unified.py \
        --input_csv data.csv \
        --model_path /path/to/model \
        --vcf_file variants.vcf \
        --output_dir results/ \
        --strategy auto  # or vanilla/flash/chunked
"""

import argparse
import os
import math
import time
import json
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Set, Any
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from calc_flash_attention_v2 import calc_attentions


# Strategy thresholds
VANILLA_MAX_LENGTH = 4096
FLASH_MAX_LENGTH = 131072


def get_device(gpu_id: int = -1):
    """Get torch device based on GPU ID."""
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def load_vcf_indel_info(vcf_path: str) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Return (insertion_lengths, position_presence) from VCF."""
    insertion_lengths: Dict[int, int] = defaultdict(int)
    position_presence: Dict[int, int] = defaultdict(int)

    with open(vcf_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            pos = int(parts[1])
            ref = parts[3]
            alts = parts[4].split(',')

            position_presence[pos] = max(position_presence[pos], 1)

            for alt in alts:
                if len(alt) > len(ref):
                    ins_len = len(alt) - len(ref)
                    if ins_len > insertion_lengths[pos]:
                        insertion_lengths[pos] = ins_len
                elif len(alt) < len(ref):
                    del_len = len(ref) - len(alt)
                    for offset in range(del_len + 1):
                        position_presence[pos + offset] = max(position_presence[pos + offset], 1)

    return insertion_lengths, position_presence


def determine_position_columns(
    df: pd.DataFrame,
    insertion_lengths: Dict[int, int],
    position_presence: Dict[int, int],
) -> Dict[int, int]:
    """Return mapping of position -> occurrences across haplotypes."""
    base_positions: Set[int] = set(position_presence.keys())

    for row in df.itertuples(index=False):
        for attr in ('hap1_pos', 'hap2_pos'):
            pos_str = getattr(row, attr)
            if not isinstance(pos_str, str) or not pos_str:
                continue
            for p in pos_str.split(';'):
                if p:
                    base_positions.add(int(p))

    for pos in insertion_lengths.keys():
        base_positions.add(pos)

    return {pos: 1 for pos in sorted(base_positions)}


def build_column_list(pos_occurrences: Dict[int, int]) -> List[str]:
    """Build sorted list of position column names."""
    columns: List[str] = []
    for pos in sorted(pos_occurrences.keys()):
        columns.append(f"pos_{pos}")
    return columns


def analyze_sequence_lengths(df: pd.DataFrame) -> Tuple[int, int, int]:
    """Analyze sequence lengths in the dataset.
    
    Returns:
        max_len: Maximum sequence length
        avg_len: Average sequence length
        count: Number of valid sequences
    """
    max_len = 0
    total_len = 0
    count = 0
    
    for _, row in df.iterrows():
        for col in ['hap1_seq', 'hap2_seq']:
            seq = row.get(col)
            if isinstance(seq, str) and len(seq) > 0:
                seq_len = len(seq)
                max_len = max(max_len, seq_len)
                total_len += seq_len
                count += 1
    
    avg_len = total_len // count if count > 0 else 0
    
    return max_len, avg_len, count


def determine_strategy(max_len: int, strategy: str) -> str:
    """Determine which attention computation strategy to use.
    
    Args:
        max_len: Maximum sequence length in the dataset
        strategy: User-specified strategy ('auto', 'vanilla', 'flash', 'chunked')
    
    Returns:
        Selected strategy name
    """
    if strategy != 'auto':
        return strategy
    
    if max_len <= VANILLA_MAX_LENGTH:
        return 'vanilla'
    elif max_len <= FLASH_MAX_LENGTH:
        return 'flash'
    else:
        return 'chunked'


########################################################################
## Strategy 1: Vanilla Attention (L ≤ 4,096)
########################################################################

captured_attentions: Dict[str, torch.Tensor] = {}


def get_attention_hook(name: str):
    """Hook to capture attention weights from model output."""
    def hook(module, inputs, outputs):
        try:
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                captured_attentions[name] = outputs[1].detach().cpu()
            elif hasattr(module, 'attention_weights') and module.attention_weights is not None:
                captured_attentions[name] = module.attention_weights.detach().cpu()
            elif hasattr(module, 'attn_probs') and module.attn_probs is not None:
                captured_attentions[name] = module.attn_probs.detach().cpu()
        except Exception:
            pass
    return hook


def calc_attentions_vanilla(seq: str, model, tokenizer, device) -> List[float]:
    """Vanilla attention computation with full matrix materialization.
    
    Used for sequences with L ≤ 4,096 tokens.
    """
    captured_attentions.clear()
    tokenizer.model_max_length = int(1e9)
    inputs = tokenizer(seq, return_tensors="pt", truncation=False)
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    if "last_self_attn" not in captured_attentions:
        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            captured_attentions["last_self_attn"] = outputs.attentions[-1].detach().cpu()
        else:
            raise ValueError("Failed to extract attention weights from model")

    # Average over heads and sum over rows (column sums)
    attn = captured_attentions["last_self_attn"].mean(dim=1)[0].sum(dim=0)
    return attn.detach().cpu().float().numpy().tolist()


########################################################################
## Strategy 2: FlashAttention-based (4,096 < L ≤ 131,072)
########################################################################

def calc_attentions_flash(
    seq: str, 
    model, 
    tokenizer, 
    device, 
    block_rows: int = 1024,
    causal: bool = True
) -> List[float]:
    """FlashAttention-based computation without full matrix materialization.
    
    Used for sequences with 4,096 < L ≤ 131,072 tokens.
    Uses block-wise query processing and streaming softmax.
    """
    return calc_attentions(seq, model, tokenizer, device, causal, block_rows)


########################################################################
## Strategy 3: Chunked Processing (L > 131,072)
########################################################################

def calculate_seq_chunks(seq_len: int, chunk_size: int, overlap: int) -> int:
    """Calculate number of chunks needed for a sequence."""
    if seq_len <= chunk_size:
        return 1
    else:
        effective_chunk_size = chunk_size - overlap
        return math.ceil((seq_len - overlap) / effective_chunk_size)


def process_sequence_chunk(
    sample: str,
    seq: str, 
    pos_list: List[int],
    model, 
    tokenizer, 
    device,
    chunk_size: int,
    overlap: int,
    chunk_idx: int,
    total_chunks: int,
    block_rows: int = 8192,
    causal: bool = True,
) -> Tuple[str, Dict[str, float], float]:
    """Process a single chunk of a sequence.
    
    Discards half of overlap region at chunk boundaries to mitigate edge effects.
    """
    start_pos = 0 if chunk_idx == 0 else chunk_idx * (chunk_size - overlap)
    end_pos = min(start_pos + chunk_size, len(seq))
    
    if chunk_idx == total_chunks - 1:
        end_pos = len(seq)
        start_pos = max(0, end_pos - chunk_size)
    
    if not pos_list:
        return sample, {}, 0.0
    start_idx = min(start_pos, len(pos_list) - 1)
    end_idx = min(end_pos, len(pos_list))
    
    if start_idx >= end_idx:
        return sample, {}, 0.0
    
    try:
        seq_chunk = seq[start_pos:end_pos]
        
        # Timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        
        attn_scores = calc_attentions(seq_chunk, model, tokenizer, device, causal, block_rows)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        duration = t1 - t0
        
        current_pos_list = pos_list[start_idx:end_idx]
        L_chunk = len(current_pos_list)
        if L_chunk == 0:
            return sample, {}, duration

        if len(attn_scores) > L_chunk:
            attn_scores = attn_scores[:L_chunk]

        # Discard half of overlap at boundaries
        edge = overlap // 2
        if chunk_idx == 0:
            valid_start = 0
        else:
            valid_start = min(edge, L_chunk)

        if chunk_idx == total_chunks - 1:
            valid_end = L_chunk
        else:
            valid_end = max(0, L_chunk - edge)

        if valid_start >= valid_end:
            return sample, {}, duration

        sum_dict: Dict[int, float] = defaultdict(float)
        cnt_dict: Dict[int, int] = defaultdict(int)

        for local_i in range(valid_start, valid_end):
            pos = current_pos_list[local_i]
            att = attn_scores[local_i]
            sum_dict[pos] += float(att)
            cnt_dict[pos] += 1
        
        result: Dict[str, float] = {'sample': sample}
        for pos, s in sum_dict.items():
            col_name = f"pos_{pos}"
            result[col_name] = s / cnt_dict[pos]
        
        return sample, result, duration

    except Exception as e:
        print(f"Error processing chunk {chunk_idx} for sample {sample}: {str(e)}")
        return sample, {}, 0.0


def process_sample_chunked(
    row: pd.Series,
    model,
    tokenizer,
    device,
    chunk_size: int,
    overlap: int,
    hap_name: str
) -> Tuple[Dict[str, Any], Set[str], float, int]:
    """Process a sample using chunked strategy.
    
    Returns:
        result: Dictionary with attention scores
        columns: Set of column names
        total_time: Total computation time
        num_chunks: Number of chunks processed
    """
    sample = row['sample']
    seq_col = f'{hap_name}_seq'
    pos_col = f'{hap_name}_pos'
    
    seq = row.get(seq_col)
    pos_str = row.get(pos_col)
    
    if not isinstance(seq, str) or len(seq) == 0 or not isinstance(pos_str, str) or len(pos_str) == 0:
        return {'sample': sample}, set(), 0.0, 0
    
    pos_list = [int(p) for p in pos_str.split(';') if p]
    total_chunks = calculate_seq_chunks(len(seq), chunk_size, overlap)
    
    chunk_results = []
    total_time = 0.0

    for chunk_idx in range(total_chunks):
        _, chunk_result, chunk_duration = process_sequence_chunk(
            sample, seq, pos_list, model, tokenizer, device,
            chunk_size, overlap, chunk_idx, total_chunks,
            block_rows=8192, causal=True,
        )
        
        total_time += chunk_duration
        if chunk_result:
            chunk_results.append(chunk_result)
    
    merged_result = {'sample': sample}
    all_columns = set()
    
    for result in chunk_results:
        all_columns.update(result.keys())
    
    column_values = defaultdict(list)
    for result in chunk_results:
        for col, val in result.items():
            if col != 'sample':
                column_values[col].append(val)
    
    for col, values in column_values.items():
        if values:
            merged_result[col] = sum(values) / len(values)
    
    return merged_result, {col for col in all_columns if col != 'sample'}, total_time, total_chunks


########################################################################
## Main Processing Functions
########################################################################

def compute_haplotype_tables_vanilla(
    df: pd.DataFrame,
    model,
    tokenizer,
    device,
    pos_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute attention tables using vanilla strategy."""
    hap1_rows: List[Dict[str, Optional[float]]] = []
    hap2_rows: List[Dict[str, Optional[float]]] = []
    metadata_rows: List[Dict[str, int]] = []
    all_position_columns = set(pos_columns)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Vanilla attention"):
        sample = row['sample']
        original_label = int(row['sample_type'])
        metadata_rows.append({'sample': sample, 'sample_type': original_label})

        for hap_name, seq_col, pos_col, container in [
            ('hap1', 'hap1_seq', 'hap1_pos', hap1_rows),
            ('hap2', 'hap2_seq', 'hap2_pos', hap2_rows),
        ]:
            seq = row.get(seq_col)
            pos_str = row.get(pos_col)
            if not isinstance(seq, str) or len(seq) == 0:
                container.append({'sample': sample})
                continue
            if not isinstance(pos_str, str) or len(pos_str) == 0:
                container.append({'sample': sample})
                continue

            pos_list = [int(p) for p in pos_str.split(';')]
            attn = calc_attentions_vanilla(seq, model, tokenizer, device)
            length = min(len(pos_list), len(attn))

            row_dict: Dict[str, Optional[float]] = {'sample': sample}

            counters: Dict[int, int] = defaultdict(int)
            for pos, att in zip(pos_list[:length], attn[:length]):
                col_name = f"pos_{pos}"
                row_dict[col_name] = float(att)
                all_position_columns.add(col_name)

            container.append(row_dict)

    position_cols_sorted = ['sample'] + sorted(all_position_columns)

    hap1_df = pd.DataFrame(hap1_rows).reindex(columns=position_cols_sorted)
    hap2_df = pd.DataFrame(hap2_rows).reindex(columns=position_cols_sorted)
    metadata_df = pd.DataFrame(metadata_rows).drop_duplicates(subset='sample').reset_index(drop=True)

    return metadata_df, hap1_df, hap2_df


def compute_haplotype_tables_flash(
    df: pd.DataFrame,
    model,
    tokenizer,
    device,
    pos_columns: List[str],
    block_rows: int = 1024,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute attention tables using FlashAttention-based strategy."""
    hap1_rows: List[Dict[str, Optional[float]]] = []
    hap2_rows: List[Dict[str, Optional[float]]] = []
    metadata_rows: List[Dict[str, int]] = []
    all_position_columns = set(pos_columns)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="FlashAttention-based"):
        sample = row['sample']
        original_label = int(row['sample_type'])
        metadata_rows.append({'sample': sample, 'sample_type': original_label})

        for hap_name, seq_col, pos_col, container in [
            ('hap1', 'hap1_seq', 'hap1_pos', hap1_rows),
            ('hap2', 'hap2_seq', 'hap2_pos', hap2_rows),
        ]:
            seq = row.get(seq_col)
            pos_str = row.get(pos_col)
            if not isinstance(seq, str) or len(seq) == 0:
                container.append({'sample': sample})
                continue
            if not isinstance(pos_str, str) or len(pos_str) == 0:
                container.append({'sample': sample})
                continue

            pos_list = [int(p) for p in pos_str.split(';')]
            attn = calc_attentions_flash(seq, model, tokenizer, device, block_rows=block_rows)
            length = min(len(pos_list), len(attn))

            row_dict: Dict[str, Optional[float]] = {'sample': sample}

            for pos, att in zip(pos_list[:length], attn[:length]):
                col_name = f"pos_{pos}"
                row_dict[col_name] = float(att)
                all_position_columns.add(col_name)

            container.append(row_dict)

    position_cols_sorted = ['sample'] + sorted(all_position_columns)

    hap1_df = pd.DataFrame(hap1_rows).reindex(columns=position_cols_sorted)
    hap2_df = pd.DataFrame(hap2_rows).reindex(columns=position_cols_sorted)
    metadata_df = pd.DataFrame(metadata_rows).drop_duplicates(subset='sample').reset_index(drop=True)

    return metadata_df, hap1_df, hap2_df


def compute_haplotype_tables_chunked(
    df: pd.DataFrame,
    model,
    tokenizer,
    device,
    chunk_size: int,
    overlap: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Compute attention tables using chunked processing strategy."""
    hap1_rows: List[Dict[str, Optional[float]]] = []
    hap2_rows: List[Dict[str, Optional[float]]] = []
    metadata_rows: List[Dict[str, int]] = []
    all_position_columns = set()
    
    total_time = 0.0
    total_chunks = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunked processing"):
        sample = row['sample']
        original_label = int(row['sample_type'])
        metadata_rows.append({'sample': sample, 'sample_type': original_label})

        # Process hap1
        h1_res, h1_cols, h1_time, h1_chunks = process_sample_chunked(
            row, model, tokenizer, device, chunk_size, overlap, 'hap1'
        )
        hap1_rows.append(h1_res)
        all_position_columns.update(h1_cols)
        total_time += h1_time
        total_chunks += h1_chunks

        # Process hap2
        h2_res, h2_cols, h2_time, h2_chunks = process_sample_chunked(
            row, model, tokenizer, device, chunk_size, overlap, 'hap2'
        )
        hap2_rows.append(h2_res)
        all_position_columns.update(h2_cols)
        total_time += h2_time
        total_chunks += h2_chunks

    position_cols_sorted = ['sample'] + sorted([col for col in all_position_columns if col != 'sample'])

    hap1_df = pd.DataFrame(hap1_rows).reindex(columns=position_cols_sorted)
    hap2_df = pd.DataFrame(hap2_rows).reindex(columns=position_cols_sorted)
    metadata_df = pd.DataFrame(metadata_rows).drop_duplicates(subset='sample').reset_index(drop=True)

    stats = {
        'total_time': total_time,
        'total_chunks': total_chunks,
        'avg_time_per_chunk': total_time / total_chunks if total_chunks > 0 else 0.0
    }

    return metadata_df, hap1_df, hap2_df, stats


def collapse_haplotype_df(hap_df: pd.DataFrame, pos_occurrences: Dict[int, int]) -> pd.DataFrame:
    """Collapse haplotype dataframe to base positions."""
    base_positions = sorted(pos_occurrences.keys())
    cols = ['sample'] + [f"pos_{pos}" for pos in base_positions]
    return hap_df.reindex(columns=cols)


def main():
    workflow_start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Unified attention extraction with automatic strategy selection"
    )
    parser.add_argument('--input_csv', required=True, help='Input CSV file')
    parser.add_argument('--model_path', required=True, help='Path to model')
    parser.add_argument('--vcf_file', required=True, help='VCF file for indel information')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum samples to process')
    parser.add_argument(
        '--strategy', 
        type=str, 
        default='auto',
        choices=['auto', 'vanilla', 'flash', 'chunked'],
        help='Attention computation strategy (default: auto)'
    )
    parser.add_argument('--block_rows', type=int, default=1024, help='Block size for flash attention')
    parser.add_argument('--chunk_size', type=int, default=8192, help='Chunk size for chunked strategy')
    parser.add_argument('--chunk_overlap', type=int, default=4096, help='Overlap size for chunked strategy')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID')

    args = parser.parse_args()

    print("=" * 80)
    print("Unified Attention Extraction")
    print("=" * 80)
    print(f"Input: {args.input_csv}")
    print(f"Model: {args.model_path}")
    print(f"VCF: {args.vcf_file}")
    print(f"Output: {args.output_dir}")
    
    device = get_device(args.gpu_id)
    print(f"Device: {device}")

    # Load data
    df = pd.read_csv(args.input_csv)
    if args.max_samples is not None and args.max_samples > 0:
        df = df.head(args.max_samples)
    
    print(f"Total samples: {len(df)}")

    # Analyze sequences
    max_seq_len, avg_seq_len, seq_count = analyze_sequence_lengths(df)
    print(f"Sequence statistics:")
    print(f"  Max length: {max_seq_len}")
    print(f"  Avg length: {avg_seq_len}")
    print(f"  Count: {seq_count}")

    # Determine strategy
    strategy = determine_strategy(max_seq_len, args.strategy)
    print(f"\nSelected strategy: {strategy.upper()}")
    
    if strategy == 'vanilla':
        print(f"  - Using vanilla attention (L ≤ {VANILLA_MAX_LENGTH})")
        print(f"  - Full attention matrix materialization")
    elif strategy == 'flash':
        print(f"  - Using FlashAttention-based computation ({VANILLA_MAX_LENGTH} < L ≤ {FLASH_MAX_LENGTH})")
        print(f"  - Block-wise processing with block_rows={args.block_rows}")
    else:
        print(f"  - Using chunked processing (L > {FLASH_MAX_LENGTH})")
        print(f"  - Chunk size: {args.chunk_size}, Overlap: {args.chunk_overlap}")

    # Load VCF info
    insertion_lengths, position_presence = load_vcf_indel_info(args.vcf_file)
    pos_occurrences = determine_position_columns(df, insertion_lengths, position_presence)
    pos_columns = build_column_list(pos_occurrences)
    print(f"Position columns: {len(pos_columns)}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.model_max_length = int(1e9)

    if strategy == 'vanilla':
        # Vanilla uses AutoModelForCausalLM with eager attention
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            output_attentions=True,
            trust_remote_code=True,
        ).to(device)
        
        # Register hook for vanilla attention
        hook = None
        try:
            target_layer = None
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                target_layer = model.model.layers[-1].self_attn
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
                target_layer = model.transformer.layers[-1].self_attn
            if target_layer is not None:
                hook = target_layer.register_forward_hook(get_attention_hook("last_self_attn"))
        except Exception:
            pass

        metadata_df, hap1_df, hap2_df = compute_haplotype_tables_vanilla(
            df, model, tokenizer, device, pos_columns
        )
        
        if hook is not None:
            hook.remove()
        
        stats = {}

    else:
        # Flash and chunked strategies use AutoModel with flash_attention_2
        model = AutoModel.from_pretrained(
            args.model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)

        if strategy == 'flash':
            metadata_df, hap1_df, hap2_df = compute_haplotype_tables_flash(
                df, model, tokenizer, device, pos_columns, args.block_rows
            )
            stats = {}
        else:  # chunked
            metadata_df, hap1_df, hap2_df, stats = compute_haplotype_tables_chunked(
                df, model, tokenizer, device, args.chunk_size, args.chunk_overlap
            )

    # Save results
    print("\nSaving results...")
    meta_path = os.path.join(args.output_dir, 'metadata.csv')
    metadata_df.to_csv(meta_path, index=False)

    hap1_collapsed = collapse_haplotype_df(hap1_df, pos_occurrences)
    hap2_collapsed = collapse_haplotype_df(hap2_df, pos_occurrences)

    if strategy == 'chunked':
        # Use parquet for large datasets
        hap1_path = os.path.join(args.output_dir, 'hap1_attention_collapsed.parquet')
        hap2_path = os.path.join(args.output_dir, 'hap2_attention_collapsed.parquet')
        hap1_collapsed.to_parquet(hap1_path, index=False)
        hap2_collapsed.to_parquet(hap2_path, index=False)
    else:
        # Use CSV for smaller datasets
        hap1_path = os.path.join(args.output_dir, 'hap1_attention_collapsed.csv')
        hap2_path = os.path.join(args.output_dir, 'hap2_attention_collapsed.csv')
        hap1_collapsed.to_csv(hap1_path, index=False)
        hap2_collapsed.to_csv(hap2_path, index=False)

    print(f"Saved results:")
    print(f"  {meta_path}")
    print(f"  {hap1_path}")
    print(f"  {hap2_path}")

    # Timing statistics
    workflow_end_time = time.time()
    total_workflow_time = workflow_end_time - workflow_start_time

    timing_stats = {
        "strategy": strategy,
        "total_workflow_time_seconds": total_workflow_time,
        "total_samples": len(df),
        "max_sequence_length": max_seq_len,
        "avg_sequence_length": avg_seq_len,
    }

    if strategy == 'chunked':
        timing_stats.update({
            "total_attention_time_seconds": stats.get('total_time', 0.0),
            "total_chunks": stats.get('total_chunks', 0),
            "avg_time_per_chunk_ms": stats.get('avg_time_per_chunk', 0.0) * 1000,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
        })

    stats_path = os.path.join(args.output_dir, 'timing_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(timing_stats, f, indent=4)

    print(f"\nTiming statistics:")
    print(f"  Total workflow time: {total_workflow_time:.2f}s")
    if strategy == 'chunked':
        print(f"  Total chunks processed: {stats.get('total_chunks', 0)}")
        print(f"  Avg time per chunk: {stats.get('avg_time_per_chunk', 0.0) * 1000:.2f}ms")
    print(f"  Statistics saved to: {stats_path}")

    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
