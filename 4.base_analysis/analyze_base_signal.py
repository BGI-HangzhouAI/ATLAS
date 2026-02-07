
import os
import argparse
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from sklearn.preprocessing import quantile_transform
from scipy import stats
import math
from scipy.stats import mannwhitneyu, wasserstein_distance, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
from itertools import combinations

# import pyBigWig
HG38_SIZES = {"chr11": 135_086_622,}


def load_data(path: str, **kwargs) -> pd.DataFrame:
    """
    根据文件扩展名自动选择合适的 pandas 读取函数。

    支持：
    - .csv / .csv.gz / .csv.bz2 / .csv.zip → read_csv
    - .tsv / .tsv.gz / .tsv.bz2 / .tsv.zip → read_csv(sep='\\t')
    - .parquet → read_parquet
    - .feather / .ft → read_feather
    - .pkl / .pickle → read_pickle
    - .json → read_json
    - .jsonl / .ndjson → read_json(lines=True)

    其他参数通过 **kwargs 透传给对应的 read_XXX。
    """
    lower = path.lower()

    # CSV
    if lower.endswith((".csv", ".csv.gz", ".csv.bz2", ".csv.zip")):
        return pd.read_csv(path, **kwargs)

    # TSV
    if lower.endswith((".tsv", ".tsv.gz", ".tsv.bz2", ".tsv.zip")):
        return pd.read_csv(path, sep="\t", **kwargs)

    # Parquet
    if lower.endswith(".parquet"):
        return pd.read_parquet(path, **kwargs)

    # Feather
    if lower.endswith((".feather", ".ft")):
        return pd.read_feather(path, **kwargs)

    # Pickle
    if lower.endswith((".pkl", ".pickle")):
        return pd.read_pickle(path, **kwargs)

    # JSON / JSONL
    if lower.endswith(".json"):
        return pd.read_json(path, **kwargs)

    if lower.endswith((".jsonl", ".ndjson")):
        return pd.read_json(path, lines=True, **kwargs)

    raise ValueError(f"Unsupported file extension for path: {path}")
    
    
def get_score_file(the_dir, pattern):
    return glob(f"{the_dir}/{pattern}")[0]


# 计算组内组间差异效应，包括多种方法，后面都用conhen_d
def compute_separability_scores(matrix, groups, min_n=10, dropna=True):
    uniq = sorted(groups.unique().tolist())
    pairs = [(a,b) for i,a in enumerate(uniq) for b in uniq[i+1:]]
    
    positions = matrix.columns
    out_rows = []
    
    group_to_df = {g: matrix.loc[groups == g] for g in uniq}
    
    for pos in positions:
        # 提前取出每组该位点向量（含 NaN）；按需清洗
        col_per_group = {}
        for g in uniq:
            v = group_to_df[g][pos].to_numpy()
            if dropna:
                v = v[np.isfinite(v)]
            col_per_group[g] = v

        for gA, gB in pairs:
            a = col_per_group[gA]
            b = col_per_group[gB]

            nA, nB = a.size, b.size
            if nA < min_n or nB < min_n:
                out_rows.append({
                    "position": pos, "groupA": gA, "groupB": gB,
                    "n_A": nA, "n_B": nB,
                    "mean_A": np.nan, "mean_B": np.nan,
                    "std_A": np.nan, "std_B": np.nan,
                    "auc": np.nan, "cliffs_delta": np.nan,
                    "wasserstein": np.nan,
                    "cohen_d": np.nan, "hedges_g": np.nan,
                })
                continue

            meanA = float(np.mean(a)); meanB = float(np.mean(b))
            stdA  = float(np.std(a, ddof=1)) if nA > 1 else np.nan
            stdB  = float(np.std(b, ddof=1)) if nB > 1 else np.nan

            # Mann–Whitney U -> AUC
            U, _ = mannwhitneyu(a, b, alternative="two-sided")
            auc = U / (nA * nB)  # P(a < b) + 1/2 P(a == b)

            # Cliff's delta
            cliffs = 2.0 * auc - 1.0

            # Wasserstein distance
            wdist = wasserstein_distance(a, b)

            # Cohen's d (B - A) / pooled std
            if (nA > 1) and (nB > 1) and np.isfinite(stdA) and np.isfinite(stdB):
                sp_num = (nA - 1) * (stdA ** 2) + (nB - 1) * (stdB ** 2)
                sp_den = (nA + nB - 2)
                s_pool = np.sqrt(sp_num / sp_den) if sp_den > 0 and sp_num >= 0 else np.nan
            else:
                s_pool = np.nan
            cohen_d = (meanB - meanA) / s_pool if (s_pool is not None and np.isfinite(s_pool) and s_pool > 0) else np.nan

            # Hedges' g：对 Cohen's d 的无偏修正
            # 修正因子 J = 1 - 3/(4*(nA+nB) - 9)
            N = nA + nB
            if np.isfinite(cohen_d) and N > 3:
                J = 1.0 - 3.0 / (4.0 * N - 9.0)
                hedges_g = cohen_d * J
            else:
                hedges_g = np.nan

            out_rows.append({
                "position": pos, "groupA": gA, "groupB": gB,
                "n_A": nA, "n_B": nB,
                "mean_A": meanA, "mean_B": meanB,
                "std_A": stdA, "std_B": stdB,
                "auc": float(auc),
                "cliffs_delta": float(cliffs),
                "wasserstein": float(wdist),
                "cohen_d": float(cohen_d),
                "hedges_g": float(hedges_g),
            })

    return pd.DataFrame(out_rows)


# 差异attention分析主要逻辑
def bh(p):
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(p)
    q = np.full_like(p, np.nan, dtype=float)

    if mask.sum() == 0:
        return q

    pvals = p[mask]
    n = len(pvals)

    order = np.argsort(pvals)
    ranked = pvals[order]
    ranks = np.arange(1, n+1)

    # BH step: p * n / rank
    qvals = ranked * n / ranks

    # Reverse cumulative minimum
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]

    qvals = np.clip(qvals, 0, 1)

    q_corrected = np.empty_like(qvals)
    q_corrected[order] = qvals

    q[mask] = q_corrected
    return q

def differential_attention(matrix, groups, group_pairs=None, min_n=10, skip_del=False):
    if group_pairs is None:
        uniq = sorted(groups.unique().tolist())
        group_pairs = [(a,b) for i,a in enumerate(uniq) for b in uniq[i+1:]]
    cols = matrix.columns
    # 删除剂量（单倍型级别：NA=1, 非NA=0）
    del_flag = matrix.isna().astype(int)
    
    # 1) 连续注意力：rank test（忽略NA）
    rows = []
    for gA, gB in group_pairs:
        print(f"Compare {gA} vs. {gB} for continuous value rank test ...")
        maskA = (groups == gA).values
        maskB = (groups == gB).values
        XA = matrix.values[maskA, :]
        XB = matrix.values[maskB, :]

        meanA = np.nanmean(XA, axis=0)
        meanB = np.nanmean(XB, axis=0)
        baseMean = np.nanmean(matrix.values, axis=0)
        
        global_min = np.nanmin([np.nanmin(meanA), np.nanmin(meanB)])
        shift = (-global_min + 1e-8) if np.isfinite(global_min) and global_min <= 0 else 0.0
        log2FC = np.log2((meanB + shift) / (meanA + shift + 1e-8))
        delta = meanB - meanA

        p_cont = np.empty(len(cols), dtype=float)
        nA_eff = np.sum(~np.isnan(XA), axis=0)
        nB_eff = np.sum(~np.isnan(XB), axis=0)

        for j in range(len(cols)):
            a = XA[:, j]; a = a[np.isfinite(a)]
            b = XB[:, j]; b = b[np.isfinite(b)]
            if len(a) >= min_n and len(b) >= min_n:
                _, p = stats.mannwhitneyu(a, b, alternative='two-sided')
            else:
                p = np.nan
            p_cont[j] = p
        q_cont = bh(p_cont)

        part_cont = pd.DataFrame({
            "hap_part": "continuous",
            "groupA": gA, "groupB": gB,
            "position": cols,
            "baseMean": baseMean,
            "mean_A": meanA, "mean_B": meanB,
            "log2FC": log2FC, "delta": delta,
            "nA_nonNA": nA_eff, "nB_nonNA": nB_eff,
            "pvalue": p_cont, "padj": q_cont
        })
        rows.append(part_cont)
        
    # 2) 删除富集（比较 NA 比例）
    if not skip_del:
        rows_del = []
        uniq = sorted(groups.unique().tolist())
        for gA, gB in group_pairs:
            print(f"Compare {gA} vs. {gB} for missing value test ...")
            maskA = (groups == gA).values
            maskB = (groups == gB).values
            if np.sum(maskA) < min_n or np.sum(maskB) < min_n:
                continue
            
            dA = del_flag.values[maskA, :]  # 1=deleted, 0=present
            dB = del_flag.values[maskB, :]

            p_del = np.empty(len(cols), dtype=float)
            or_est = np.empty(len(cols), dtype=float)
            for j in range(len(cols)):
                a_del = int(dA[:, j].sum()); a_pres = int((~np.isnan(matrix.values[maskA, j])).sum()) - (len(dA[:, j]) - (len(dA[:, j]) - a_del))
                # 更安全：直接用样本计数
                nA = dA.shape[0]; a_pres = nA - a_del

                b_del = int(dB[:, j].sum()); nB = dB.shape[0]; b_pres = nB - b_del

                table = np.array([[a_del, a_pres],
                                [b_del, b_pres]])
                # 若计数都>=5，用卡方；否则 Fisher
                if (table < 5).any():
                    _, p = stats.fisher_exact(table)
                else:
                    _, p = stats.chi2_contingency(table)[:2]
                p_del[j] = p
                # 粗略OR（加0.5连续性修正）
                or_est[j] = ((a_del + 0.5)/(a_pres + 0.5)) / ((b_del + 0.5)/(b_pres + 0.5))

            q_del = bh(p_del)
            part_del = pd.DataFrame({
                "hap_part": "deletion_enrichment",
                "groupA": gA, "groupB": gB,
                "position": cols,
                "OR_deleted": or_est,
                "pvalue": p_del, "padj": q_del,
                "nA": dA.shape[0], "nB": dB.shape[0]
            })
            rows_del.append(part_del)
        
    out = pd.concat(rows, ignore_index=True)
    out_del = None
    if not skip_del:
        out_del = pd.concat(rows_del, ignore_index=True)
    return out, out_del
    

# 绘制attention log2FC随位置变化曲线，并标注真值位点
def plot_lfc_with_truth_by_pairs(
    diff_df: pd.DataFrame,
    score_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    group_pairs,
    lfc_quantile_thresh: float = 0.90,
    padj_thresh: float = 0.05,
    cohen_thresh: float = 0.5,
    ignore_pos: list = None,
    pos_col: str = "position",
    groupA_col: str = "groupA",
    groupB_col: str = "groupB",
    truth_pos_col: str = "pos",
    truth_id_col: str = "ID",
    figsize=(18, 4),
    truth_fontsize: int = 10,
    unify_ylim: bool = False,
    show_name = True,
):
    """
    对指定的 group 对（group_pairs）逐个绘制 log2FC 随位置的曲线，标注真值位点并高亮显著点。
    - 显著性规则：|log2FC| ≥ quantile(|log2FC|, lfc_quantile_thresh) 且 padj < padj_thresh
    - 标题会显示显著点计数
    - 每个对比单独一张图（返回 [(fig, ax), ...]）

    参数
    ----
    diff_df: 差异分析结果，至少包含列 [position, log2FC, padj, groupA, groupB]
             position 可为 'pos_XXXX' 字符串或整数坐标
    truth_df: 真值表，至少包含 [truth_pos_col]；如果有 truth_id_col 会显示 ID
    group_pairs: 需要绘图的组对列表，例如 [(0,1), (0,2), (1,2)]
    lfc_quantile_thresh: 显著性使用的 |log2FC| 分位阈值（0~1）
    padj_thresh: 显著性使用的 FDR 阈值
    truth_fontsize: 真值 ID 标注字体大小
    unify_ylim: 若为 True，会在所有子图绘制前先扫描各对比的 log2FC 取全局 y 轴范围

    返回
    ----
    figs_axes: list of (fig, ax)
    """
    df = diff_df.copy()
    df = df.merge(score_df[["position", "groupA", "groupB", "cohen_d"]], how="inner", 
                  on=["position", "groupA", "groupB"],)

    # --- position 统一为 int（兼容 "pos_XXXX"） ---
    if df[pos_col].dtype == object:
        df[pos_col] = df[pos_col].str.replace("^pos_", "", regex=True)
    df[pos_col] = df[pos_col].astype(int)
    
    if ignore_pos is not None:
        if not isinstance(ignore_pos, list):
            raise ValueError(f"ignore_pos should be a list, but received {ignore_pos}")
        df = df[~df[pos_col].isin(ignore_pos)]

    # --- 全域 x 轴范围（统一 x 轴，便于比较） ---
    min_pos = int(df[pos_col].min())
    max_pos = int(df[pos_col].max())
    all_positions = pd.DataFrame({pos_col: np.arange(min_pos, max_pos + 1, dtype=int)})

    # --- 真值 ---
    if truth_df is not None:
        truth = truth_df.copy()
        truth[truth_pos_col] = truth[truth_pos_col].astype(int)
        has_id = truth_id_col in truth.columns
        in_range_truth = truth[(truth[truth_pos_col] >= min_pos) & (truth[truth_pos_col] <= max_pos)]

    # --- 若需要统一 y 轴范围，先预扫 ---
    global_ymin, global_ymax = None, None
    if unify_ylim:
        ymin_list = []
        ymax_list = []
        for gA, gB in group_pairs:
            sub = df[(df[groupA_col] == gA) & (df[groupB_col] == gB)][[pos_col, "log2FC", "padj", "cohen_d"]]
            if sub.empty:
                continue
            merged = all_positions.merge(sub, on=pos_col, how="left")
            lfc = merged["log2FC"].fillna(0.0).values
            if lfc.size:
                ymin_list.append(np.nanmin(lfc))
                ymax_list.append(np.nanmax(lfc))
        if ymin_list and ymax_list:
            # 留点边距
            global_ymin = min(ymin_list) - 0.05 * (max(ymax_list) - min(ymin_list))
            global_ymax = max(ymax_list) + 0.05 * (max(ymax_list) - min(ymin_list))

    figs_axes = []

    for (gA, gB) in group_pairs:
        sub = df[(df[groupA_col] == gA) & (df[groupB_col] == gB)][[pos_col, "log2FC", "padj", "cohen_d"]]
        y_cur_min = np.nanmin(sub["log2FC"])
        y_cur_max = np.nanmax(sub["log2FC"])
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        if sub.empty:
            ax.set_title(f"Comparison {gA} vs {gB} | No data", fontsize=12)
            ax.set_xlabel("Position"); ax.set_ylabel("log2FC")
            figs_axes.append((fig, ax))
            continue

        merged = all_positions.merge(sub, on=pos_col, how="left")
        merged["log2FC"] = merged["log2FC"].fillna(0.0)
        merged["padj"] = merged["padj"].fillna(1.0)

        # 显著性：分位阈值 + FDR
        sig_count = 0
        if lfc_quantile_thresh is not None and padj_thresh is not None:
            abs_lfc = merged["log2FC"].abs()
            q_thr = abs_lfc.quantile(lfc_quantile_thresh) if abs_lfc.max() > 0 else 0.0
            abs_cohen = merged["cohen_d"].abs()
            sig_mask = (abs_lfc >= q_thr) & (merged["padj"] < padj_thresh) & (abs_cohen > cohen_thresh)
            sig_count = int(sig_mask.sum())

        # 主曲线
        ax.plot(merged[pos_col], merged["log2FC"], color="tab:blue", linewidth=1.0, alpha=0.9)
        # 填充正/负区域
        ax.fill_between(merged[pos_col], merged["log2FC"], 0, where=(merged["log2FC"] > 0), alpha=0.25)
        ax.fill_between(merged[pos_col], merged["log2FC"], 0, where=(merged["log2FC"] < 0), alpha=0.25)

        # 高亮显著点
        if sig_count > 0:
            ax.scatter(
                merged.loc[sig_mask, pos_col],
                merged.loc[sig_mask, "log2FC"],
                s=14, marker="o", edgecolors="k", linewidths=0.6, zorder=3
            )

        # 真值标注（竖线 + ID）
        if truth_df is not None:
            for _, r in in_range_truth.iterrows():
                xpos = int(r[truth_pos_col])
                ax.axvline(x=xpos, color="darkred", linestyle="--", linewidth=0.9, alpha=0.85, zorder=1)
                if has_id and show_name:
                    ymin, ymax = ax.get_ylim()
                    if unify_ylim:
                        ymin, ymax = global_ymin, global_ymax
                    ax.text(
                        xpos,
                        ymin + 0.90 * (ymax - ymin),  # 顶部 90% 处
                        str(r[truth_id_col]),
                        rotation=90, fontsize=truth_fontsize, ha="right", va="top",
                        color="darkred"
                    )

        # 统一 y 轴范围（可选）
        if unify_ylim and (global_ymin is not None) and (global_ymax is not None):
            ax.set_ylim(global_ymin, global_ymax)

        ax.set_xlim(min_pos, max_pos)
        ax.set_xlabel("Position")
        ax.set_ylabel("log2FC")
        ax.grid(alpha=0.2, linestyle=":")
        
        if lfc_quantile_thresh is not None and padj_thresh is not None:
            ax.set_title(
                f"Comparison {gA} vs {gB} | significant: {sig_count} "
                f"( |lfc|≥Q{lfc_quantile_thresh:.2f}, FDR<{padj_thresh} )",
                fontsize=16
            )
        else:
            ax.set_title(
                f"Comparison {gA} vs {gB}", fontsize=16
            )
            
        plt.tight_layout()
        figs_axes.append((fig, ax))

    return figs_axes


def evaluate_performance(
    diff_df: pd.DataFrame, 
    truth_df: pd.DataFrame, 
    group_pair: tuple,
    left_bp: int = 0, 
    right_bp: int = 50,
    pos_col: str = "position",
    score_col: str = "log2FC"
):
    """
    Calculates AUC/AUPRC along with signal concentration metrics (SNR, FRiW, Weighted Distance).
    """
    # 1. Filter data for the current group pair
    gA, gB = group_pair
    df = diff_df[(diff_df["groupA"] == gA) & (diff_df["groupB"] == gB)].copy()
    
    if df.empty:
        return None

    # 2. Process coordinate column
    if df[pos_col].dtype == object:
        df[pos_col] = df[pos_col].str.replace("^pos_", "", regex=True)
    df[pos_col] = df[pos_col].astype(int)
    
    # 3. Ground Truth Labeling & Distance Calculation
    # Initialize mask and distance array (for Weighted Distance metric)
    positions = df[pos_col].values
    is_true = np.zeros(len(positions), dtype=bool)
    
    # Initialize min_dists with infinity. This will store the distance to the *nearest* variant.
    min_dists = np.full(len(positions), np.inf)

    # Get all ground truth positions
    true_positions = truth_df["pos"].astype(int).values
    
    # Iterate to build mask and calculate distances
    # Note: If true_positions is very large (>10k), consider using KDTree for distance, 
    # but for typical variant lists, this loop is efficient enough.
    for t_pos in true_positions:
        # A. Define Window Logic (Labeling)
        start = t_pos - left_bp
        end = t_pos + right_bp
        mask = (positions >= start) & (positions <= end)
        is_true |= mask
        
        # B. Distance Logic (Metric: Spatial Accuracy)
        # Update the minimum distance to any variant found so far
        dist_to_this_variant = np.abs(positions - t_pos)
        min_dists = np.minimum(min_dists, dist_to_this_variant)
        
    df["is_true_region"] = is_true.astype(int)
    
    # 4. Prepare Scores
    # Use absolute log2FC as the score (magnitude of change)
    y_true = df["is_true_region"].values
    y_score = df[score_col].abs().fillna(0).values
    
    # Basic Counts
    pos_count = is_true.sum()
    neg_count = (~is_true).sum()
    
    # --- NEW METRICS CALCULATION ---
    
    # Metric 1: Signal-to-Noise Ratio (SNR)
    # Mean signal in True regions / Mean signal in False regions
    mean_signal_true = y_score[is_true].mean() if pos_count > 0 else 0
    mean_signal_false = y_score[~is_true].mean() if neg_count > 0 else 0
    # Avoid division by zero
    snr = mean_signal_true / mean_signal_false if mean_signal_false > 0 else np.nan

    # Metric 2: Fraction of Signal in Windows (FRiW)
    # Total signal in True regions / Total signal across whole sequence
    total_signal = y_score.sum()
    signal_in_true = y_score[is_true].sum()
    friw = signal_in_true / total_signal if total_signal > 0 else 0

    # Metric 3: Signal-Weighted Mean Distance
    # Average distance of signal from the nearest variant. Lower is better (tighter signal).
    # sum(score_i * dist_i) / sum(score_i)
    if total_signal > 0:
        weighted_dist = np.sum(y_score * min_dists) / total_signal
    else:
        weighted_dist = np.nan

    # 5. Calculate Classification Metrics (AUC/AUPRC)
    if len(np.unique(y_true)) < 2:
        roc_auc = np.nan
        auprc = np.nan
    else:
        roc_auc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
    
    return {
        "groupA": gA, 
        "groupB": gB,
        # Classification Metrics (Rank-based)
        "AUC": roc_auc,
        "AUPRC": auprc,
        # Concentration Metrics (Magnitude & Spatial)
        "SNR": snr,                   # Higher is better (>1.5 is good)
        "FRiW": friw,                 # Higher is better (e.g. 0.05 - 0.2 depending on peak width)
        "Weighted_Dist": weighted_dist, # Lower is better (signal is closer to variants)
        # Metadata
        "True_Window_Left": left_bp,
        "True_Window_Right": right_bp,
        "Positive_Base_Count": pos_count,
        "Mean_Signal_True": mean_signal_true,
        "Mean_Signal_Background": mean_signal_false
    }
    

def evaluate_performance_normalized(
    diff_df: pd.DataFrame, 
    truth_df: pd.DataFrame, 
    group_pair: tuple,
    left_bp: int = 0, 
    right_bp: int = 50,
    pos_col: str = "position",
    score_col: str = "log2FC"
):
    """
    Length-normalized metrics for evaluating signal concentration near variants.
    """
    # ... [keep your existing filtering and labeling code through line ~50] ...
    
    # 1. Filter data for the current group pair
    gA, gB = group_pair
    df = diff_df[(diff_df["groupA"] == gA) & (diff_df["groupB"] == gB)].copy()
    
    if df.empty:
        return None

    # 2. Process coordinate column
    if df[pos_col].dtype == object:
        df[pos_col] = df[pos_col].str.replace("^pos_", "", regex=True)
    df[pos_col] = df[pos_col].astype(int)
    
    # 3. Ground Truth Labeling & Distance Calculation
    # Initialize mask and distance array (for Weighted Distance metric)
    positions = df[pos_col].values
    is_true = np.zeros(len(positions), dtype=bool)
    
    # Initialize min_dists with infinity. This will store the distance to the *nearest* variant.
    min_dists = np.full(len(positions), np.inf)

    # Get all ground truth positions
    true_positions = truth_df["pos"].astype(int).values
    
    # Iterate to build mask and calculate distances
    # Note: If true_positions is very large (>10k), consider using KDTree for distance, 
    # but for typical variant lists, this loop is efficient enough.
    for t_pos in true_positions:
        # A. Define Window Logic (Labeling)
        start = t_pos - left_bp
        end = t_pos + right_bp
        mask = (positions >= start) & (positions <= end)
        is_true |= mask
        
        # B. Distance Logic (Metric: Spatial Accuracy)
        # Update the minimum distance to any variant found so far
        dist_to_this_variant = np.abs(positions - t_pos)
        min_dists = np.minimum(min_dists, dist_to_this_variant)
        
    df["is_true_region"] = is_true.astype(int)
    
    # 4. Prepare Scores
    # Use absolute log2FC as the score (magnitude of change)
    y_true = df["is_true_region"].values
    y_score = df[score_col].abs().fillna(0).values
    
    pos_count = is_true.sum()
    neg_count = (~is_true).sum()
    total_positions = len(positions)
    n_variants = len(true_positions)
    
    mean_signal_true = y_score[is_true].mean() if pos_count > 0 else 0
    mean_signal_false = y_score[~is_true].mean() if neg_count > 0 else 0
    
    # ========== LENGTH-NORMALIZED METRICS ==========
    
    # 1. SNR remains the same - it's already a ratio of means
    #    But we can add a "scaled" version that accounts for sample size uncertainty
    snr = mean_signal_true / mean_signal_false if mean_signal_false > 0 else np.nan
    
    # 2. Expected FRiW under null (if signal were uniformly distributed)
    #    Normalize FRiW by what we'd expect by chance
    total_signal = y_score.sum()
    signal_in_true = y_score[is_true].sum()
    friw = signal_in_true / total_signal if total_signal > 0 else 0
    
    expected_friw = pos_count / total_positions  # fraction of positions in true windows
    friw_enrichment = friw / expected_friw if expected_friw > 0 else np.nan  # fold enrichment
    
    # 3. Normalized Weighted Distance
    #    Divide by sequence length or window size to make it relative
    window_size = left_bp + right_bp + 1
    seq_length = positions.max() - positions.min() + 1
    
    if total_signal > 0:
        weighted_dist = np.sum(y_score * min_dists) / total_signal
        # Normalize by the "expected" distance if signal were uniform
        # For uniform distribution, expected distance to nearest of k variants 
        # in length L is approximately L / (2k)
        expected_dist = seq_length / (2 * n_variants) if n_variants > 0 else seq_length
        normalized_weighted_dist = weighted_dist / expected_dist  # <1 means better than random
    else:
        weighted_dist = np.nan
        normalized_weighted_dist = np.nan
    
    # 4. NEW: Precision at fixed recall (length-independent by design)
    #    What fraction of top-k predictions are in true regions?
    #    where k = number of true positions (or some multiple)
    k = pos_count  # or use: k = n_variants * window_size
    if k > 0 and len(y_score) >= k:
        top_k_indices = np.argsort(y_score)[-k:]
        precision_at_k = is_true[top_k_indices].mean()
    else:
        precision_at_k = np.nan
    
    # 5. NEW: Cohen's d (effect size) - inherently normalized
    std_true = y_score[is_true].std() if pos_count > 1 else 0
    std_false = y_score[~is_true].std() if neg_count > 1 else 0
    pooled_std = np.sqrt(((pos_count - 1) * std_true**2 + (neg_count - 1) * std_false**2) 
                         / (pos_count + neg_count - 2)) if (pos_count + neg_count) > 2 else 0
    cohens_d = (mean_signal_true - mean_signal_false) / pooled_std if pooled_std > 0 else np.nan
    
    # 6. NEW: Rank-based metric - completely length independent
    #    Mean percentile rank of true positions
    ranks = np.argsort(np.argsort(y_score))  # ranks from 0 to n-1
    percentile_ranks = ranks / (len(ranks) - 1) if len(ranks) > 1 else ranks
    mean_rank_true = percentile_ranks[is_true].mean() if pos_count > 0 else np.nan
    
    # 7. Calculate Classification Metrics (AUC/AUPRC)
    if len(np.unique(y_true)) < 2:
        roc_auc = np.nan
        auprc = np.nan
    else:
        roc_auc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
    
    return {
        "groupA": gA, 
        "groupB": gB,
        # Original metrics (for reference)
        "AUC": roc_auc,
        "AUPRC": auprc,
        "SNR": snr,
        "FRiW": friw,
        "Weighted_Dist": weighted_dist,
        # LENGTH-NORMALIZED metrics
        "FRiW_Enrichment": friw_enrichment,      # >1 means enriched vs random; length-independent
        "Normalized_Weighted_Dist": normalized_weighted_dist,  # <1 means better than random
        "Precision_at_K": precision_at_k,        # length-independent
        "Cohens_d": cohens_d,                    # effect size; length-independent  
        "Mean_Rank_True": mean_rank_true,        # 0.5 = random, 1.0 = perfect; length-independent
        # Metadata
        "Seq_Length": seq_length,
        "N_Variants": n_variants,
        "Expected_FRiW": expected_friw,
    }


# 控制参数
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HBB attention differential analysis pipeline."
    )

    # ====== 基础路径设置 ======
    parser.add_argument(
        "--hbb_attn_dir",
        type=str,
        required=True,
        help="Path to directory containing HBB attention matrices (hap1/hap2 collapsed).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory to save intermediate and final results.",
    )
    parser.add_argument(
        "--figure_dir",
        type=str,
        required=True,
        help="Directory to save plots and diagnostic figures.",
    )

    # ====== 参数控制 ======
    parser.add_argument(
        "--ignore_pos",
        type=str,
        default=None,
        help="Comma-separated positions to ignore, e.g. '5226248:5226268,5229150:5229170,5229177'. "
             "If None, all positions are included."
    )
    parser.add_argument(
        "--min_n",
        type=int,
        default=10,
        help="Minimum sample number to do significant test and deletion test"
    )
    parser.add_argument(
        "--true-sites",
        type=str,
        default=None,
        help="determine which true site will be loaded"
    )

    args = parser.parse_args()

    # ====== 目录初始化 ======
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    # ====== 解析 ignore_pos ======
    if args.ignore_pos is not None:
        try:
            parts = []
            for token in args.ignore_pos.split(","):
                if ":" in token:
                    s, e = token.split(":")
                    parts.extend(range(int(s), int(e)))
                else:
                    parts.append(int(token))
            args.ignore_pos = parts
        except Exception:
            print("[WARNING] Failed to parse ignore_pos; keep all pos instead.")
            args.ignore_pos = None
    else:
        args.ignore_pos = None

    return args
    
    
if __name__ == "__main__":
    args = parse_args()
    
    HBB_ATTN_DIR = args.hbb_attn_dir
    RESULTS_DIR = args.results_dir
    FIGURE_DIR = args.figure_dir
    
    IGNORE_POS = args.ignore_pos
    MIN_N = args.min_n
    
    SKIP_DEL = False
    # PAIRS = [(1,2),(1,3),(2,3)]
    PAIRS = [(1,2)]
    LFC_QUANTILE_THRESH = 0.95
    PADJ_THRESH = 0.01
    
    # --------------------------- 准备真值 --------------------------- #
    print("准备真值 ...")
    true_roi = None
    if args.true_sites is not None:
        true_roi = pd.read_csv(f"{args.true_sites}")
    
    
    # --------------------------- 加载注意力分数 --------------------------- #
    print("加载注意力分数 ...")
    # 1400例样本的
    samples = load_data(f"{HBB_ATTN_DIR}/metadata.csv", index_col=0)
    hap1_attn_matrix_collapsed = load_data(get_score_file(HBB_ATTN_DIR, "hap1*collapsed*"))
    hap1_attn_matrix_collapsed.set_index("sample", inplace=True)
    hap1_attn_matrix_collapsed = hap1_attn_matrix_collapsed.abs()
    hap2_attn_matrix_collapsed = load_data(get_score_file(HBB_ATTN_DIR, "hap2*collapsed*"))
    hap2_attn_matrix_collapsed.set_index("sample", inplace=True)
    hap2_attn_matrix_collapsed = hap2_attn_matrix_collapsed.abs()
    
    valid_cols = [col for col in hap1_attn_matrix_collapsed.columns if not hap1_attn_matrix_collapsed[col].isnull().all()]
    hap1_attn_matrix_collapsed = hap1_attn_matrix_collapsed[valid_cols]
    hap2_attn_matrix_collapsed = hap2_attn_matrix_collapsed[valid_cols]

    # 纯合样本的 placeholder
    samples_homo = pd.DataFrame()
    hap1_attn_matrix_collapsed_homo = pd.DataFrame()
    hap2_attn_matrix_collapsed_homo = pd.DataFrame()

    samples = pd.concat([samples, samples_homo], ignore_index=False)
    hap1 = pd.concat([hap1_attn_matrix_collapsed, hap1_attn_matrix_collapsed_homo], ignore_index=False)
    hap2 = pd.concat([hap2_attn_matrix_collapsed, hap2_attn_matrix_collapsed_homo], ignore_index=False)

    # reorder the samples 
    # 纯合 3->0, 没病携带 0->1, 轻症 1->2, 重症 2->3
    labelmap = {3:0, 0:1, 1:2, 2:3}
    samples["sample_type_reorder"] = samples["sample_type"].apply(lambda x: labelmap[x])
    
    print(samples)
    has_enough_sample = (samples.value_counts("sample_type_reorder") > args.min_n).all()
    if not has_enough_sample:
        SKIP_DEL = True
        LFC_QUANTILE_THRESH = None
        PADJ_THRESH = None
    
    
    # --------------------------- 计算组间差异对组内差异的大小 --------------------------- #
    print("计算组间差异对组内差异的大小 ...")
    samples = samples.loc[hap1.index]
    groups = samples["sample_type_reorder"]
    hap1_scores = compute_separability_scores(hap1, groups, min_n=MIN_N, dropna=True)

    samples = samples.loc[hap2.index]
    groups = samples["sample_type_reorder"]
    hap2_scores = compute_separability_scores(hap2, groups, min_n=MIN_N, dropna=True)
    
    hap1_scores.to_csv(f"{RESULTS_DIR}/hap1_separability_scores.csv", index=False)
    hap2_scores.to_csv(f"{RESULTS_DIR}/hap2_separability_scores.csv", index=False)
    
    
    # --------------------------- 差异attention分析 --------------------------- #
    print("差异attention分析 ...")
    samples = samples.loc[hap1.index]
    hap2 = hap2.loc[hap1.index]
    groups = samples["sample_type_reorder"]
    res_hap1_cont, res_hap1_del = differential_attention(hap1, groups, group_pairs=PAIRS, skip_del=SKIP_DEL)
    res_hap2_cont, res_hap2_del = differential_attention(hap2, groups, group_pairs=PAIRS, skip_del=SKIP_DEL)
    
    # 保存完整结果
    res_hap1_cont.to_csv(f"{RESULTS_DIR}/hap1_continuous_attention_differential.csv", index=False)
    res_hap2_cont.to_csv(f"{RESULTS_DIR}/hap2_continuous_attention_differential.csv", index=False)
    
    # 显著删除位点的结果
    if not SKIP_DEL:
        res_hap1_del = res_hap1_del[(res_hap1_del.padj < 0.01) & (res_hap1_del.groupA == 1)][["groupA", "groupB", "position", "OR_deleted", "padj"]]
        res_hap2_del = res_hap2_del[(res_hap2_del.padj < 0.01) & (res_hap2_del.groupA == 1)][["groupA", "groupB", "position", "OR_deleted", "padj"]]
        res_hap1_del.to_csv(f"{RESULTS_DIR}/hap1_deletion_enrichment_significant.csv", index=False)
        res_hap2_del.to_csv(f"{RESULTS_DIR}/hap2_deletion_enrichment_significant.csv", index=False)
        
    
    # --------------------------- 绘制各个碱基位置上的log2FC与真值 --------------------------- #
    print("绘制各个碱基位置上的log2FC与真值 ...")
    pairs = PAIRS  
    figs_axes = plot_lfc_with_truth_by_pairs(
        diff_df=res_hap1_cont,
        score_df=hap1_scores,
        truth_df=true_roi,                 # true_roi 需含列 pos，若含 ID 将显示
        group_pairs=pairs,
        lfc_quantile_thresh=LFC_QUANTILE_THRESH,          # 绝对log2FC分位阈值
        padj_thresh=PADJ_THRESH,                  # FDR阈值
        ignore_pos=IGNORE_POS,
        cohen_thresh=0,
        truth_fontsize=10,                 # 真值ID更大字号
        unify_ylim=False,                    # 可选：所有图统一y轴范围
        figsize=(24, 4),
        show_name=True,
    )
    for i, (fig, ax) in enumerate(figs_axes):
        gA, gB = pairs[i]
        fig.savefig(f"{FIGURE_DIR}/hap1_log2FC_with_8truth_{gA}vs{gB}.png", dpi=300)
        
    figs_axes = plot_lfc_with_truth_by_pairs(
        diff_df=res_hap1_cont,
        score_df=hap1_scores,
        truth_df=true_roi,                 # true_roi 需含列 pos，若含 ID 将显示
        group_pairs=pairs,
        lfc_quantile_thresh=LFC_QUANTILE_THRESH,          # 绝对log2FC分位阈值
        padj_thresh=PADJ_THRESH,                  # FDR阈值
        ignore_pos=IGNORE_POS,
        cohen_thresh=0.5,
        truth_fontsize=10,                 # 真值ID更大字号
        unify_ylim=False,                    # 可选：所有图统一y轴范围
        figsize=(24, 4),
        show_name=True,
    )
    for i, (fig, ax) in enumerate(figs_axes):
        gA, gB = pairs[i]
        fig.savefig(f"{FIGURE_DIR}/hap1_log2FC_with_8truth_{gA}vs{gB}_cohen.png", dpi=300)
        
    figs_axes = plot_lfc_with_truth_by_pairs(
        diff_df=res_hap1_cont,
        score_df=hap1_scores,
        truth_df=true_roi,                 # true_roi 需含列 pos，若含 ID 将显示
        group_pairs=pairs,
        lfc_quantile_thresh=LFC_QUANTILE_THRESH,          # 绝对log2FC分位阈值
        padj_thresh=PADJ_THRESH,                  # FDR阈值
        ignore_pos=IGNORE_POS,
        cohen_thresh=0.8,
        truth_fontsize=10,                 # 真值ID更大字号
        unify_ylim=False,                    # 可选：所有图统一y轴范围
        figsize=(24, 4),
        show_name=True,
    )
    for i, (fig, ax) in enumerate(figs_axes):
        gA, gB = pairs[i]
        fig.savefig(f"{FIGURE_DIR}/hap1_log2FC_with_8truth_{gA}vs{gB}_highercohen.png", dpi=300)
        
    figs_axes = plot_lfc_with_truth_by_pairs(
        diff_df=res_hap2_cont,
        score_df=hap2_scores,
        truth_df=true_roi,                 
        group_pairs=pairs,
        lfc_quantile_thresh=LFC_QUANTILE_THRESH,          
        padj_thresh=PADJ_THRESH,     
        ignore_pos=IGNORE_POS,             
        cohen_thresh=0,
        truth_fontsize=10,                 
        unify_ylim=False,                    
        figsize=(24, 4),
        show_name=True,
    )
    for i, (fig, ax) in enumerate(figs_axes):
        gA, gB = pairs[i]
        fig.savefig(f"{FIGURE_DIR}/hap2_log2FC_with_8truth_{gA}vs{gB}.png", dpi=300)
        
    figs_axes = plot_lfc_with_truth_by_pairs(
        diff_df=res_hap2_cont,
        score_df=hap2_scores,
        truth_df=true_roi,                 
        group_pairs=pairs,
        lfc_quantile_thresh=LFC_QUANTILE_THRESH,          
        padj_thresh=PADJ_THRESH,     
        ignore_pos=IGNORE_POS,             
        cohen_thresh=0.5,
        truth_fontsize=10,                 
        unify_ylim=False,                    
        figsize=(24, 4),
        show_name=True,
    )
    for i, (fig, ax) in enumerate(figs_axes):
        gA, gB = pairs[i]
        fig.savefig(f"{FIGURE_DIR}/hap2_log2FC_with_8truth_{gA}vs{gB}_cohen.png", dpi=300)
        
    figs_axes = plot_lfc_with_truth_by_pairs(
        diff_df=res_hap2_cont,
        score_df=hap2_scores,
        truth_df=true_roi,                 
        group_pairs=pairs,
        lfc_quantile_thresh=LFC_QUANTILE_THRESH,          
        padj_thresh=PADJ_THRESH,     
        ignore_pos=IGNORE_POS,             
        cohen_thresh=0.8,
        truth_fontsize=10,                 
        unify_ylim=False,                    
        figsize=(24, 4),
        show_name=True,
    )
    for i, (fig, ax) in enumerate(figs_axes):
        gA, gB = pairs[i]
        fig.savefig(f"{FIGURE_DIR}/hap2_log2FC_with_8truth_{gA}vs{gB}_highercohen.png", dpi=300)
        
        
    # --------------------------------- 定量分析Log2FC打分效果 ------------------------------------
    if true_roi is None:
        print("No true sites provided; skipping performance evaluation.")
        sys.exit(0)
        
    print(f"计算性能指标 ...")
    metrics_rows = []
    
    # 对 Hap1 计算
    for TRUE_WIN in [3, 5, 10, 20]:
        for pair in PAIRS:
            m = evaluate_performance_normalized(
                res_hap1_cont, true_roi, pair, 
                left_bp=0, right_bp=TRUE_WIN,
                pos_col="position", score_col="log2FC"
            )
            if m:
                m["Haplotype"] = "Hap1"
                m["Window"] = TRUE_WIN
                metrics_rows.append(m)

        # 对 Hap2 计算
        for pair in PAIRS:
            m = evaluate_performance_normalized(
                res_hap2_cont, true_roi, pair, 
                left_bp=0, right_bp=TRUE_WIN,
                pos_col="position", score_col="log2FC"
            )
            if m:
                m["Haplotype"] = "Hap2"
                m["Window"] = TRUE_WIN
                metrics_rows.append(m)
            
    # 保存 Benchmark 结果
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_file = os.path.join(RESULTS_DIR, "chunk_performance_metrics_auc.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Performance metrics saved to: {metrics_file}")
        # print(metrics_df[["Haplotype", "groupA", "groupB", "Window", "AUC", "AUPRC"]])
    else:
        print("No metrics calculated (empty results).")
