import os
import sys
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from glob import glob

from sklearn import metrics
from sklearn.preprocessing import quantile_transform
from scipy import stats
import math
from scipy.stats import mannwhitneyu, wasserstein_distance, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
from itertools import combinations
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from collections import Counter


# ================== 字体设置 type1 ================== 
plt.rcParams['pdf.use14corefonts'] = True
font = {'family': 'Serif'}
plt.rc('font', **font)
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams.update({
    'font.size': 16,           # 全局基础字体大小
    'axes.titlesize': 18,      # 子图标题 (ax.set_title)
    'axes.labelsize': 16,      # 轴标签 (ax.set_xlabel/ylabel)
    'xtick.labelsize': 14,     # X轴刻度数字
    'ytick.labelsize': 14,     # Y轴刻度数字
    'legend.fontsize': 14,     # 图例文字
    'figure.titlesize': 20     # 整个大图的标题
})
# =====================================================


# import pyBigWig
HG38_SIZES = {"chr11": 135_086_622,}


def load_data(path: str, **kwargs) -> pd.DataFrame:
    """
    根据文件扩展名自动选择合适的 pandas 读取函数。
    """
    lower = path.lower()

    if lower.endswith((".csv", ".csv.gz", ".csv.bz2", ".csv.zip")):
        return pd.read_csv(path, **kwargs)
    if lower.endswith((".tsv", ".tsv.gz", ".tsv.bz2", ".tsv.zip")):
        return pd.read_csv(path, sep="\t", **kwargs)
    if lower.endswith(".parquet"):
        return pd.read_parquet(path, **kwargs)
    if lower.endswith((".feather", ".ft")):
        return pd.read_feather(path, **kwargs)
    if lower.endswith((".pkl", ".pickle")):
        return pd.read_pickle(path, **kwargs)
    if lower.endswith(".json"):
        return pd.read_json(path, **kwargs)
    if lower.endswith((".jsonl", ".ndjson")):
        return pd.read_json(path, lines=True, **kwargs)

    raise ValueError(f"Unsupported file extension for path: {path}")
    
    
def get_score_file(the_dir, pattern):
    return glob(f"{the_dir}/{pattern}")[0]


def plot_trace_and_clusters_combined(
    diff_df: pd.DataFrame,
    score_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    group_pairs: list,
    gwas_df: pd.DataFrame = None,            
    lfc_quantile_thresh: float = 0.90,
    padj_thresh: float = 0.05,
    cohen_thresh: float = 0.5,
    pos_col: str = "position",
    groupA_col: str = "groupA",
    groupB_col: str = "groupB",
    truth_pos_col: str = "pos",
    truth_id_col: str = "ID",
    fig_dir: str = "./plots",
    haplotype: str = "Haplotype",
    extend_bps: int = 10,
    extend_strand: str = '+',
    figsize: tuple = (15, 4),
    title: str = None,  # NEW: Custom title parameter
):
    """
    Plots two-panel figure with shared x-axis:
    1. Top: Raw Log2FC Trace + Truth Lines + Cluster Boxes
    2. Bottom: Cluster Score Bar Plot
    """
    os.makedirs(fig_dir, exist_ok=True)
    
    # --- Data Prep ---
    df_main = diff_df.copy()
    if score_df is not None:
        cols_to_merge = ["position", "groupA", "groupB", "cohen_d"]
        if "cohen_d" not in df_main.columns:
            df_main = df_main.merge(score_df[cols_to_merge], how="left", on=["position", "groupA", "groupB"])
    else:
        if "cohen_d" not in df_main.columns:
            df_main["cohen_d"] = 0 

    # Normalize Position Column
    if df_main[pos_col].dtype == object:
        df_main[pos_col] = df_main[pos_col].str.replace("^pos_", "", regex=True)
    df_main[pos_col] = pd.to_numeric(df_main[pos_col], errors='coerce').fillna(0).astype(int)
    
    # Define Global X-Range
    min_pos = int(df_main[pos_col].min())
    max_pos = int(df_main[pos_col].max())
    all_positions = pd.DataFrame({pos_col: np.arange(min_pos, max_pos + 1, dtype=int)})

    in_range_gwas = pd.DataFrame()
    if gwas_df is not None and not gwas_df.empty:
        # 确保列名匹配，假设 CSV 里叫 'POS'，如果不确定可以做个映射
        gwas_pos_col = 'POS' if 'POS' in gwas_df.columns else 'pos'
        gwas_id_col = 'ID' if 'ID' in gwas_df.columns else 'id'
        
        if gwas_pos_col in gwas_df.columns:
            # 筛选在当前绘图区间内的 GWAS 位点
            in_range_gwas = gwas_df[
                (gwas_df[gwas_pos_col] >= min_pos) & 
                (gwas_df[gwas_pos_col] <= max_pos)
            ].copy()

    # Truth Prep
    if truth_df is not None and not truth_df.empty:
        truth = truth_df.copy()
        has_id = truth_id_col in truth.columns and truth_df[truth_id_col].notna().all()
        truth[truth_pos_col] = pd.to_numeric(truth[truth_pos_col], errors='coerce').fillna(-1).astype(int)
        in_range_truth = truth[(truth[truth_pos_col] >= min_pos) & (truth[truth_pos_col] <= max_pos)]
    else:
        in_range_truth = pd.DataFrame()
        has_id = False

    for (gA, gB) in group_pairs:
        # Filter Data
        sub_trace = df_main[(df_main[groupA_col] == gA) & (df_main[groupB_col] == gB)].copy()
        sub_sig = sig_df.copy() 

        if sub_trace.empty:
            continue

        merged = all_positions.merge(sub_trace, on=pos_col, how="left")
        merged["log2FC"] = merged["log2FC"].fillna(0.0)
        merged["padj"] = merged["padj"].fillna(1.0)
        merged["cohen_d"] = merged["cohen_d"].fillna(0.0)

        # Sig Calculation for highlighting on trace
        abs_lfc = merged["log2FC"].abs()
        q_thr = abs_lfc.quantile(lfc_quantile_thresh) if abs_lfc.max() > 0 else 0.0
        sig_mask = (abs_lfc >= q_thr) & (merged["padj"] < padj_thresh) & (merged["cohen_d"].abs() > cohen_thresh)
        sig_count = int(sig_mask.sum())

        # Cluster setup
        unique_clusters = sub_sig[sub_sig["cluster"] != -1]["cluster"].unique()
        raw_colors = plt.cm.tab10.colors + plt.cm.Set3.colors
        cluster_colors = {cid: raw_colors[i % len(raw_colors)] for i, cid in enumerate(unique_clusters)}

        # Calculate Cluster Bounds and get scores
        cluster_info = []
        for cid in unique_clusters:
            c_data = sub_sig[sub_sig["cluster"] == cid]
            
            # Get score from cluster_summary if available
            score = np.nan
            if cluster_summary is not None and not cluster_summary.empty:
                score_row = cluster_summary[cluster_summary["cluster_id"] == cid]
                if not score_row.empty:
                    score = score_row["total_score"].values[0]
            
            cluster_info.append({
                "id": cid,
                "start": c_data["position"].min() - extend_bps if extend_strand == '+' else c_data["position"].min(),
                "end": c_data["position"].max() + extend_bps if extend_strand == '-' else c_data["position"].max(),
                "center": c_data["position"].median(),
                "score": score,
                "color": cluster_colors[cid]
            })

        # --- Two-Panel Plot with Shared X-axis ---
        fig, (ax1, ax2) = plt.subplots(
            2, 1, 
            figsize=figsize, 
            sharex=True,  # Share x-axis
            gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.01}  # Minimal gap
        )
        
        # ==================== AX1: Trace ====================
        # Grey/Colored Boxes for Clusters (background)
        for info in cluster_info:
            ax1.axvspan(info["start"], info["end"], color="gray", alpha=0.2, zorder=0)
            # Also draw on ax2 for alignment reference
            ax2.axvspan(info["start"], info["end"], color="gray", alpha=0.2, zorder=0)

        # Main Trace
        ax1.plot(merged[pos_col], merged["log2FC"], color="tab:blue", linewidth=0.8, alpha=0.9, zorder=2)
        ax1.fill_between(merged[pos_col], merged["log2FC"], 0, where=(merged["log2FC"] > 0), color="tab:blue", alpha=0.2)
        ax1.fill_between(merged[pos_col], merged["log2FC"], 0, where=(merged["log2FC"] < 0), color="tab:blue", alpha=0.2)

        # Highlight Significant Points
        if sig_count > 0:
            ax1.scatter(merged.loc[sig_mask, pos_col], merged.loc[sig_mask, "log2FC"],
                        s=12, marker="o", edgecolors="k", linewidths=0.5, zorder=3, alpha=0.8)

        if not in_range_gwas.empty:
            gwas_pos_c = 'POS' if 'POS' in in_range_gwas.columns else 'pos'
            gwas_id_c = 'ID' if 'ID' in in_range_gwas.columns else 'id'
            
            for _, r in in_range_gwas.iterrows():
                xpos = int(r[gwas_pos_c])
                # 画虚线，颜色用 tab:orange 或者 gold 区别于 Truth 的 darkred
                ax1.axvline(x=xpos, color="tab:orange", linestyle="-.", linewidth=1.2, alpha=0.9, zorder=4)
                # 也在下方的 Cluster Score 图中画出参考线
                ax2.axvline(x=xpos, color="tab:orange", linestyle="-.", linewidth=0.8, alpha=0.4, zorder=1)
                
                # 如果有 rsID，标注在图的下方或者中间，避免和 Truth 的标签重叠
                if gwas_id_c in r:
                    ymin, ymax = ax1.get_ylim()
                    # 放在 y 轴 85% 的高度，颜色对应
                    ax1.text(xpos, ymin + 0.85 * (ymax - ymin), str(r[gwas_id_c]),
                             rotation=90, fontsize=12, ha="right", va="top", 
                             color="tab:orange", fontweight='bold')

        # Truth Lines on both panels
        if not in_range_truth.empty:
            for _, r in in_range_truth.iterrows():
                xpos = int(r[truth_pos_col])
                ax1.axvline(x=xpos, color="darkred", linestyle="--", linewidth=1.2, alpha=0.85, zorder=4)
                ax2.axvline(x=xpos, color="darkred", linestyle="--", linewidth=1.0, alpha=0.5, zorder=4)
                if truth_id_col in r and has_id:
                    ymin, ymax = ax1.get_ylim()
                    ax1.text(xpos, ymax * 0.95, str(r[truth_id_col]),
                             rotation=90, fontsize=8, ha="right", va="top", color="darkred", fontweight='bold')

        ax1.set_ylabel("log2FC")
        
        # Use custom title if provided, otherwise use default
        if title:
            plot_title = f"[{haplotype}] {gA} vs {gB} | {title}"
        else:
            plot_title = f"[{haplotype}] {gA} vs {gB} | Differential Attention Trace"
        ax1.set_title(plot_title)
        
        ax1.grid(True, alpha=0.2, linestyle=":")
        ax1.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        # ==================== AX2: Cluster Score Scatter Plot ====================
        if cluster_info:
            centers = [info["center"] for info in cluster_info]
            scores = [info["score"] if not np.isnan(info["score"]) else 0 for info in cluster_info]
            colors = [info["color"] for info in cluster_info]
            
            # Plot scatter
            ax2.scatter(centers, scores, c=colors, s=100, edgecolors='black', 
                        linewidths=0.8, alpha=0.9, zorder=5)
        
        ax2.set_ylabel("Cluster\nScore")
        ax2.set_xlabel("Genomic Position (bp)")
        ax2.grid(True, alpha=0.2, linestyle=":", axis='y')
        
        # Set y-axis limit for scores
        valid_scores = [info["score"] for info in cluster_info if not np.isnan(info["score"])]
        if valid_scores:
            ax2.set_ylim(0, max(valid_scores) * 1.25)
        
        # Hide x-axis label on top panel (shared with bottom)
        ax1.tick_params(labelbottom=False)

        # Set x-axis limits
        ax1.set_xlim(min_pos, max_pos)
        ax2.set_xlim(min_pos, max_pos)
        
        # Final Polish
        plt.tight_layout()
        fname = f"{fig_dir}/{haplotype}_Combined_group{gA}vs{gB}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        fname = f"{fig_dir}/{haplotype}_Combined_group{gA}vs{gB}.pdf"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  🖼️ Saved combined plot: {fname}")


def plot_haplotype_comparison(
    diff_df_hap1: pd.DataFrame,
    diff_df_hap2: pd.DataFrame,
    score_df_hap1: pd.DataFrame,
    score_df_hap2: pd.DataFrame,
    truth_df: pd.DataFrame,
    sig_df_hap1: pd.DataFrame,
    sig_df_hap2: pd.DataFrame,
    cluster_summary_hap1: pd.DataFrame,
    cluster_summary_hap2: pd.DataFrame,
    group_pair: tuple,
    gwas_df: pd.DataFrame = None,
    lfc_quantile_thresh: float = 0.90,
    padj_thresh: float = 0.05,
    cohen_thresh: float = 0.5,
    pos_col: str = "position",
    groupA_col: str = "groupA",
    groupB_col: str = "groupB",
    truth_pos_col: str = "pos",
    truth_id_col: str = "ID",
    fig_dir: str = "./plots",
    extend_bps: int = 10,
    extend_strand: str = '+',
    figsize: tuple = (20, 8),
    title: str = None,
):
    """
    Plots a 4-panel comparison figure stacking hap1 and hap2 results.
    
    Layout:
    - Row 1: Hap1 log2FC Trace
    - Row 2: Hap1 Cluster Scores
    - Row 3: Hap2 log2FC Trace
    - Row 4: Hap2 Cluster Scores
    
    All panels share the same x-axis (genomic position).
    """
    os.makedirs(fig_dir, exist_ok=True)
    
    gA, gB = group_pair
    
    # --- Helper function to process data ---
    def process_haplotype_data(diff_df, score_df, sig_df, cluster_summary):
        df_main = diff_df.copy()
        if score_df is not None:
            cols_to_merge = ["position", "groupA", "groupB", "cohen_d"]
            if "cohen_d" not in df_main.columns:
                df_main = df_main.merge(score_df[cols_to_merge], how="left", on=["position", "groupA", "groupB"])
        else:
            if "cohen_d" not in df_main.columns:
                df_main["cohen_d"] = 0
        
        if df_main[pos_col].dtype == object:
            df_main[pos_col] = df_main[pos_col].str.replace("^pos_", "", regex=True)
        df_main[pos_col] = pd.to_numeric(df_main[pos_col], errors='coerce').fillna(0).astype(int)
        
        sub_trace = df_main[(df_main[groupA_col] == gA) & (df_main[groupB_col] == gB)].copy()
        sub_sig = sig_df.copy() if sig_df is not None else pd.DataFrame()
        
        cluster_info = []
        if sub_sig is not None and not sub_sig.empty:
            unique_clusters = sub_sig[sub_sig["cluster"] != -1]["cluster"].unique()
            raw_colors = plt.cm.tab10.colors + plt.cm.Set3.colors
            cluster_colors = {cid: raw_colors[i % len(raw_colors)] for i, cid in enumerate(unique_clusters)}
            
            for cid in unique_clusters:
                c_data = sub_sig[sub_sig["cluster"] == cid]
                score = np.nan
                if cluster_summary is not None and not cluster_summary.empty:
                    score_row = cluster_summary[cluster_summary["cluster_id"] == cid]
                    if not score_row.empty:
                        score = score_row["total_score"].values[0]
                
                cluster_info.append({
                    "id": cid,
                    "start": c_data["position"].min() - extend_bps if extend_strand == '+' else c_data["position"].min(),
                    "end": c_data["position"].max() + extend_bps if extend_strand == '-' else c_data["position"].max(),
                    "center": c_data["position"].median(),
                    "score": score,
                    "color": cluster_colors[cid]
                })
        
        return sub_trace, sub_sig, cluster_info
    
    # Process both haplotypes
    trace_hap1, sig_hap1, clusters_hap1 = process_haplotype_data(
        diff_df_hap1, score_df_hap1, sig_df_hap1, cluster_summary_hap1
    )
    trace_hap2, sig_hap2, clusters_hap2 = process_haplotype_data(
        diff_df_hap2, score_df_hap2, sig_df_hap2, cluster_summary_hap2
    )
    
    if trace_hap1.empty and trace_hap2.empty:
        print("  ⚠️ No data for either haplotype")
        return
    
    # Define global x-range
    all_positions_list = []
    if not trace_hap1.empty:
        all_positions_list.extend(trace_hap1[pos_col].values)
    if not trace_hap2.empty:
        all_positions_list.extend(trace_hap2[pos_col].values)
    
    min_pos = int(min(all_positions_list))
    max_pos = int(max(all_positions_list))
    all_positions = pd.DataFrame({pos_col: np.arange(min_pos, max_pos + 1, dtype=int)})
    
    # Merge with all positions
    if not trace_hap1.empty:
        merged_hap1 = all_positions.merge(trace_hap1, on=pos_col, how="left")
        merged_hap1["log2FC"] = merged_hap1["log2FC"].fillna(0.0)
        merged_hap1["padj"] = merged_hap1["padj"].fillna(1.0)
        merged_hap1["cohen_d"] = merged_hap1["cohen_d"].fillna(0.0)
    else:
        merged_hap1 = all_positions.copy()
        merged_hap1["log2FC"] = 0.0
        
    if not trace_hap2.empty:
        merged_hap2 = all_positions.merge(trace_hap2, on=pos_col, how="left")
        merged_hap2["log2FC"] = merged_hap2["log2FC"].fillna(0.0)
        merged_hap2["padj"] = merged_hap2["padj"].fillna(1.0)
        merged_hap2["cohen_d"] = merged_hap2["cohen_d"].fillna(0.0)
    else:
        merged_hap2 = all_positions.copy()
        merged_hap2["log2FC"] = 0.0
    
    # Truth prep
    if truth_df is not None and not truth_df.empty:
        truth = truth_df.copy()
        has_id = truth_id_col in truth.columns and truth_df[truth_id_col].notna().all()
        truth[truth_pos_col] = pd.to_numeric(truth[truth_pos_col], errors='coerce').fillna(-1).astype(int)
        in_range_truth = truth[(truth[truth_pos_col] >= min_pos) & (truth[truth_pos_col] <= max_pos)]
    else:
        in_range_truth = pd.DataFrame()
        has_id = False
        
    in_range_gwas = pd.DataFrame()
    gwas_pos_col = 'POS'
    gwas_id_col = 'ID'
    
    if gwas_df is not None and not gwas_df.empty:
        # 简单兼容列名大小写
        if 'POS' in gwas_df.columns: gwas_pos_col = 'POS'
        elif 'pos' in gwas_df.columns: gwas_pos_col = 'pos'
        
        if 'ID' in gwas_df.columns: gwas_id_col = 'ID'
        elif 'id' in gwas_df.columns: gwas_id_col = 'id'
        
        if gwas_pos_col in gwas_df.columns:
            in_range_gwas = gwas_df[
                (gwas_df[gwas_pos_col] >= min_pos) & 
                (gwas_df[gwas_pos_col] <= max_pos)
            ].copy()
    
    # --- Create 4-panel Figure ---
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1, 2, 1], 'hspace': 0.05}
    )
    
    # 定义一个内部辅助函数来画 GWAS 线，避免重复代码
    def plot_gwas_lines(ax, add_labels=False):
        if not in_range_gwas.empty:
            for _, r in in_range_gwas.iterrows():
                xpos = int(r[gwas_pos_col])
                # 绘制垂直虚线 (橙色)
                ax.axvline(x=xpos, color="tab:orange", linestyle="-.", linewidth=1.0, alpha=0.7, zorder=1)
                
                # 添加 Label (仅在 add_labels=True 时，通常是 Trace 图)
                if add_labels and gwas_id_col in r:
                    label_txt = str(r[gwas_id_col])
                    if label_txt and label_txt.lower() != 'nan' and label_txt != '.':
                        ymin, ymax = ax.get_ylim()
                        # 放在 y 轴高度的 85% 处，避免跟 Truth (通常在 95%) 重叠
                        ax.text(xpos, ymin + 0.85 * (ymax - ymin), label_txt,
                                rotation=90, fontsize=8, ha="right", va="top", 
                                color="tab:orange", fontweight='bold', zorder=10)
    
    # ==================== AX1: Hap1 Trace ====================
    for info in clusters_hap1:
        ax1.axvspan(info["start"], info["end"], color="gray", alpha=0.2, zorder=0)
    
    ax1.plot(merged_hap1[pos_col], merged_hap1["log2FC"], color="tab:blue", linewidth=0.8, alpha=0.9, zorder=2)
    ax1.fill_between(merged_hap1[pos_col], merged_hap1["log2FC"], 0, 
                     where=(merged_hap1["log2FC"] > 0), color="tab:blue", alpha=0.2)
    ax1.fill_between(merged_hap1[pos_col], merged_hap1["log2FC"], 0, 
                     where=(merged_hap1["log2FC"] < 0), color="tab:blue", alpha=0.2)
    
    # Highlight Significant Points for Hap1
    if sig_hap1 is not None and not sig_hap1.empty:
        sig_positions = sig_hap1["position"].values
        sig_lfc = []
        for pos in sig_positions:
            match = merged_hap1[merged_hap1[pos_col] == pos]
            if not match.empty:
                sig_lfc.append(match["log2FC"].values[0])
            else:
                sig_lfc.append(np.nan)
        valid_mask = ~np.isnan(sig_lfc)
        ax1.scatter(np.array(sig_positions)[valid_mask], np.array(sig_lfc)[valid_mask],
                    s=12, marker="o", edgecolors="k", linewidths=0.5, zorder=3, alpha=0.8)
    
    if not in_range_truth.empty:
        for _, r in in_range_truth.iterrows():
            xpos = int(r[truth_pos_col])
            ax1.axvline(x=xpos, color="darkred", linestyle="--", linewidth=1.0, alpha=0.7, zorder=4)
            if truth_id_col in r and has_id:
                ymin, ymax = ax1.get_ylim()
                ax1.text(xpos, ymax * 0.95, str(r[truth_id_col]),
                         rotation=90, fontsize=8, ha="right", va="top", color="darkred", fontweight='bold')
    
    ax1.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax1.set_ylabel("log2FC")
    
    if title:
        plot_title = f"Haplotype Comparison | Group {gA} vs {gB} | {title}"
    else:
        plot_title = f"Haplotype Comparison | Group {gA} vs {gB}"
    ax1.set_title(plot_title, fontweight='bold')
    ax1.grid(True, alpha=0.2, linestyle=":")
    
    # Add Hap1 label
    ax1.text(0.01, 0.95, "Hap1", transform=ax1.transAxes, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plot_gwas_lines(ax1, add_labels=True)
    
    # ==================== AX2: Hap1 Cluster Scores ====================
    for info in clusters_hap1:
        ax2.axvspan(info["start"], info["end"], color="gray", alpha=0.2, zorder=0)
    
    if not in_range_truth.empty:
        for _, r in in_range_truth.iterrows():
            xpos = int(r[truth_pos_col])
            ax2.axvline(x=xpos, color="darkred", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
    
    if clusters_hap1:
        centers = [info["center"] for info in clusters_hap1]
        scores = [info["score"] if not np.isnan(info["score"]) else 0 for info in clusters_hap1]
        colors = [info["color"] for info in clusters_hap1]
        ax2.scatter(centers, scores, c=colors, s=100, edgecolors='black', linewidths=0.8, alpha=0.9, zorder=5)
    
    ax2.set_ylabel("Cluster\nScore")
    ax2.grid(True, alpha=0.2, linestyle=":", axis='y')
    
    valid_scores_h1 = [info["score"] for info in clusters_hap1 if not np.isnan(info["score"])]
    if valid_scores_h1:
        ax2.set_ylim(0, max(valid_scores_h1) * 1.25)
        
    plot_gwas_lines(ax2, add_labels=False)
    
    # ==================== AX3: Hap2 Trace ====================
    for info in clusters_hap2:
        ax3.axvspan(info["start"], info["end"], color="gray", alpha=0.2, zorder=0)
    
    ax3.plot(merged_hap2[pos_col], merged_hap2["log2FC"], color="tab:blue", linewidth=0.8, alpha=0.9, zorder=2)
    ax3.fill_between(merged_hap2[pos_col], merged_hap2["log2FC"], 0, 
                     where=(merged_hap2["log2FC"] > 0), color="tab:blue", alpha=0.2)
    ax3.fill_between(merged_hap2[pos_col], merged_hap2["log2FC"], 0, 
                     where=(merged_hap2["log2FC"] < 0), color="tab:blue", alpha=0.2)
    
    # Highlight Significant Points for Hap2
    if sig_hap2 is not None and not sig_hap2.empty:
        sig_positions = sig_hap2["position"].values
        sig_lfc = []
        for pos in sig_positions:
            match = merged_hap2[merged_hap2[pos_col] == pos]
            if not match.empty:
                sig_lfc.append(match["log2FC"].values[0])
            else:
                sig_lfc.append(np.nan)
        valid_mask = ~np.isnan(sig_lfc)
        ax3.scatter(np.array(sig_positions)[valid_mask], np.array(sig_lfc)[valid_mask],
                    s=12, marker="o", edgecolors="k", linewidths=0.5, zorder=3, alpha=0.8)
    
    if not in_range_truth.empty:
        for _, r in in_range_truth.iterrows():
            xpos = int(r[truth_pos_col])
            ax3.axvline(x=xpos, color="darkred", linestyle="--", linewidth=1.0, alpha=0.7, zorder=4)
    
    ax3.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax3.set_ylabel("log2FC")
    ax3.grid(True, alpha=0.2, linestyle=":")
    
    # Add Hap2 label
    ax3.text(0.01, 0.95, "Hap2", transform=ax3.transAxes, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plot_gwas_lines(ax3, add_labels=True)
    
    # ==================== AX4: Hap2 Cluster Scores ====================
    for info in clusters_hap2:
        ax4.axvspan(info["start"], info["end"], color="gray", alpha=0.2, zorder=0)
    
    if not in_range_truth.empty:
        for _, r in in_range_truth.iterrows():
            xpos = int(r[truth_pos_col])
            ax4.axvline(x=xpos, color="darkred", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
    
    if clusters_hap2:
        centers = [info["center"] for info in clusters_hap2]
        scores = [info["score"] if not np.isnan(info["score"]) else 0 for info in clusters_hap2]
        colors = [info["color"] for info in clusters_hap2]
        ax4.scatter(centers, scores, c=colors, s=100, edgecolors='black', linewidths=0.8, alpha=0.9, zorder=5)
    
    ax4.set_ylabel("Cluster\nScore")
    ax4.set_xlabel("Genomic Position (bp)")
    ax4.grid(True, alpha=0.2, linestyle=":", axis='y')
    
    valid_scores_h2 = [info["score"] for info in clusters_hap2 if not np.isnan(info["score"])]
    if valid_scores_h2:
        ax4.set_ylim(0, max(valid_scores_h2) * 1.25)
        
    plot_gwas_lines(ax4, add_labels=False)
    
    # Hide x-axis labels on upper panels
    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)
    ax3.tick_params(labelbottom=False)
    
    # Set x-axis limits
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(min_pos, max_pos)
    
    plt.tight_layout()
    fname = f"{fig_dir}/Haplotype_Comparison_group{gA}vs{gB}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    fname = f"{fig_dir}/Haplotype_Comparison_group{gA}vs{gB}.pdf"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  🖼️ Saved haplotype comparison plot: {fname}")


def cluster_significant_cohen_positions(
    diff_df: pd.DataFrame,
    score_df: pd.DataFrame,
    truth_df: pd.DataFrame = None,
    gwas_df: pd.DataFrame = None,
    haplotype: str = "hap1",
    group_pair: tuple = (0, 3),
    adjp_threshold: float = 0.01,
    lfc_quantile: float = 0.95,
    cohen_threshold: float = 0.2,
    eps: float = 20,
    eps_percentile: int = None,
    min_samples: int = 5,
    weighted_distance: bool = True,
    position_weight: float = 1.0,
    pvalue_weight: float = 0.5,
    lfc_weight: float = 0.3,
    plot_diagnostics: bool = True,
    extend_bps: int = 10,
    extend_strand: str = '+',
    fig_dir: str = "./cluster_picture/",
    plot_mode: str = "combined",
    title: str = None,  # NEW: Custom title parameter
) -> pd.DataFrame:
    
    # --- 1. Data Loading & Cleaning ---
    print(f"[{haplotype}] Converting position format...")
    df_processed = diff_df.copy()
    
    if score_df is not None:
         cols = ["position", "groupA", "groupB", "cohen_d"]
         if "cohen_d" not in df_processed.columns:
             df_processed = df_processed.merge(score_df[cols], how="inner", on=["position", "groupA", "groupB"])

    if df_processed["position"].dtype == object:
        df_processed["position"] = df_processed["position"].str.replace("^pos_", "", regex=True)
    df_processed["position"] = pd.to_numeric(df_processed["position"], errors='coerce')
    df_processed = df_processed.dropna(subset=["position"])

    global_min_pos = int(df_processed["position"].min())
    global_max_pos = int(df_processed["position"].max())
    
    print(f"[{haplotype}] Filtering significant sites...")
    mask = (df_processed["groupA"] == group_pair[0]) & (df_processed["groupB"] == group_pair[1])
    df = df_processed[mask].copy()
    
    if df.empty:
        print(f"  ⚠️ Warning: No data for pair {group_pair}")
        return pd.DataFrame(), pd.DataFrame()
    
    try:
        lfc_thr = df["log2FC"].abs().quantile(lfc_quantile)
    except:
        print(f"  ⚠️ Warning: Could not calculate log2FC quantile")
        return pd.DataFrame(), pd.DataFrame()
    
    if cohen_threshold is not None:
        sig_mask = (df["padj"] < adjp_threshold) & (df["log2FC"].abs() >= lfc_thr) & (df["cohen_d"].abs() >= cohen_threshold)
    else:
        sig_mask = (df["padj"] < adjp_threshold) & (df["log2FC"].abs() >= lfc_thr)
        
    sig_df = df[sig_mask].copy()
    
    if sig_df.empty:
        print(f"  ⚠️ Warning: No significant sites found.")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"  ✅ Found {len(sig_df)} significant sites.")

    # --- 2. Clustering Preparation ---
    positions = sig_df["position"].values
    log2FC_abs = np.abs(sig_df["log2FC"].values)
    p_strength = -np.log10(sig_df["padj"].values + 1e-100)
    
    if weighted_distance:
        pos_scaled = StandardScaler().fit_transform(positions.reshape(-1, 1)).flatten()
        p_scaled = StandardScaler().fit_transform(p_strength.reshape(-1, 1)).flatten()
        lfc_scaled = StandardScaler().fit_transform(log2FC_abs.reshape(-1, 1)).flatten()
        features = np.vstack([
            position_weight * pos_scaled,
            pvalue_weight * p_scaled,
            lfc_weight * lfc_scaled
        ]).T
    else:
        features = positions.reshape(-1, 1)
    
    # --- 3. Determine EPS ---
    if eps is None:
        neigh = NearestNeighbors(n_neighbors=min_samples)
        neigh.fit(features)
        distances, _ = neigh.kneighbors(features)
        k_distances = np.sort(distances[:, -1])
        eps = np.percentile(k_distances, eps_percentile)
        print(f"Auto-calculated eps: {eps:.4f} ({eps_percentile}th percentile)")
    else:
        print(f"Using specified eps: {eps:.4f}")
    
    # --- 4. DBSCAN Clustering ---
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = dbscan.fit_predict(features)
    sig_df["cluster"] = cluster_labels
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"  ✅ Clustering done: {n_clusters} valid clusters.")
    
    # --- 5. Scoring Clusters ---
    print("  🎯 Scoring clusters...")
    cluster_scores = []
    
    for cluster_id in sorted(sig_df["cluster"].unique()):
        if cluster_id == -1:
            continue
        
        cluster_data = sig_df[sig_df["cluster"] == cluster_id]
        n_points = len(cluster_data)
        if n_points == 0: continue
        
        center_pos = cluster_data["position"].median()
        
        top_n_10 = min(10, n_points)
        mean_abs_lfc_top10 = cluster_data["log2FC"].abs().nlargest(top_n_10).mean()
        mean_p_strength_top_10 = (-np.log10(cluster_data["padj"])).nlargest(top_n_10).mean()
        
        if n_points > 1:
            positions_sorted = np.sort(cluster_data["position"].values)
            diffs = np.diff(positions_sorted)
            mean_spacing = np.mean(diffs[diffs > 0]) if np.any(diffs > 0) else 0.1
        else:
            mean_spacing = 1.0

        score = (
            np.log(n_points) + 
            np.log(100 / (mean_spacing + 1e-9)) +
            mean_abs_lfc_top10 * 30 +
            mean_p_strength_top_10 * 0.5
        )
        
        cluster_scores.append({
            "cluster_id": int(cluster_id),
            "n_points": n_points,
            "center_position": int(center_pos),
            "start_position": int(cluster_data["position"].min()),
            "end_position": int(cluster_data["position"].max()),
            "mean_abs_log2FC_top10": mean_abs_lfc_top10,
            "mean_p_strength_top10": mean_p_strength_top_10,
            "mean_spacing": mean_spacing,
            "total_score": score,
            "direction": "positive" if cluster_data["log2FC"].mean() > 0 else "negative"
        })
    
    if not cluster_scores:
        print(f"  ⚠️ Warning: No valid clusters to score.")
        return pd.DataFrame(), sig_df
    
    cluster_summary = pd.DataFrame(cluster_scores)
    cluster_summary = cluster_summary.sort_values("total_score", ascending=False).reset_index(drop=True)
    
    # --- 6. Plotting ---
    if plot_diagnostics:
        print("  🎨 Plotting diagnostic trace...")
        
        plot_trace_and_clusters_combined(
            diff_df=diff_df,
            score_df=score_df,
            truth_df=truth_df,
            sig_df=sig_df,
            cluster_summary=cluster_summary,
            group_pairs=[group_pair],
            lfc_quantile_thresh=lfc_quantile,
            padj_thresh=adjp_threshold,
            cohen_thresh=cohen_threshold if cohen_threshold else 0,
            fig_dir=fig_dir,
            haplotype=haplotype,
            extend_bps=extend_bps,
            extend_strand=extend_strand,
            title=title,
            gwas_df=gwas_df,
        )
    
    # Print Top 5
    print(f"\n  🏆 Top-5 Cluster Summary:")
    print(cluster_summary.head(5)[["cluster_id", "n_points", "center_position", "direction", "total_score"]].to_string())
    
    return cluster_summary, sig_df
    

def match_clusters_to_truth(
    all_sig_dfs: dict,
    summary_df: pd.DataFrame,
    true_roi: pd.DataFrame,
    score_percent: float = None,
    min_n_points: int = None,
    min_density: float = None,
    extension_bp: int = 10,
    direction: str = '-', 
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Matches clusters to ground truth variants and calculates Precision/Recall.
    """
    if direction not in ['+', '-']:
        raise ValueError(f"DIRECTION parameter must be '+' or '-', received '{direction}'")
    
    truth_matching_results = []
    metrics_list = []
    
    for key, sig_df in all_sig_dfs.items():
        try:
            parts = key.split('_')
            haplotype = parts[0]
            groups_str = parts[-1] 
            groupA, groupB = map(int, groups_str.split('vs'))
        except Exception as e:
            print(f"Skipping key '{key}' due to parsing error: {e}")
            continue
        
        print(f"\n[{haplotype}] Processing match for {groupA}vs{groupB}...")
        
        n_clusters_total = 0
        n_clusters_tp = 0
        
        total_variants_pool = set(true_roi['ID'].unique())
        n_variants_total = len(total_variants_pool)
        
        found_truth_ids_this_pair = set()
        
        group_summary = summary_df[
            (summary_df['haplotype'] == haplotype) & 
            (summary_df['groupA'] == groupA) & 
            (summary_df['groupB'] == groupB)
        ]
        
        score_threshold = float('-inf')
        if not group_summary.empty and score_percent is not None:
            if not (0 < score_percent < 1):
                raise ValueError(f"score_percent must be between 0 and 1, got {score_percent}")
            high_score_quantile = 1 - score_percent
            score_threshold = group_summary['total_score'].quantile(high_score_quantile)
            print(f"  Score Threshold (Top {score_percent*100:.1f}%): {score_threshold:.4f}")

        if not group_summary.empty:
            for _, cluster_row in group_summary.iterrows():
                cluster_id = cluster_row['cluster_id']
                n_points = cluster_row['n_points']
                score = cluster_row['total_score']
                start_pos = cluster_row['start_position']
                end_pos = cluster_row['end_position']
                
                cluster_span = end_pos - start_pos
                if cluster_span == 0: cluster_span = 1 
                density = n_points / cluster_span
                
                n_pass = min_n_points is None or n_points >= min_n_points
                m_pass = min_density is None or density >= min_density
                s_pass = score >= score_threshold
                
                if not (n_pass and m_pass and s_pass):
                    continue
                
                n_clusters_total += 1
                
                if direction == '+':
                    match_start = start_pos - extension_bp
                    match_end = end_pos
                else:
                    match_start = start_pos
                    match_end = end_pos + extension_bp
                
                in_range_mask = (
                    (true_roi['pos'] >= match_start) & 
                    (true_roi['pos'] <= match_end)
                )
                matched_truth = true_roi[in_range_mask].copy()
                
                base_info = {
                    'haplotype': haplotype,
                    'groupA': groupA,
                    'groupB': groupB,
                    'cluster_id': cluster_id,
                    'n_points': n_points,
                    'density': density,
                    'total_score': score,
                    'cluster_start': start_pos,
                    'cluster_end': end_pos,
                    'match_window_start': match_start,
                    'match_window_end': match_end,
                }

                if not matched_truth.empty:
                    n_clusters_tp += 1
                    found_truth_ids_this_pair.update(matched_truth['ID'].tolist())
                    
                    for _, truth_row in matched_truth.iterrows():
                        record = base_info.copy()
                        record.update({
                            'match_status': 'True Positive',
                            'truth_id': truth_row['ID'],
                            'truth_position': truth_row['pos'],
                            'truth_cpra': truth_row['cpra'] if 'cpra' in truth_row else None,
                            'dist_truth_to_cluster_center': truth_row['pos'] - cluster_row['center_position']
                        })
                        truth_matching_results.append(record)
                
                else:
                    record = base_info.copy()
                    record.update({
                        'match_status': 'False Positive',
                        'truth_id': None,
                        'truth_position': None,
                        'truth_cpra': None,
                        'dist_truth_to_cluster_center': None
                    })
                    truth_matching_results.append(record)
        
        else:
            print("  ⚠️  No clusters found. Precision is 0, Recall is 0.")

        missed_variants = true_roi[~true_roi['ID'].isin(found_truth_ids_this_pair)]
        
        if not missed_variants.empty:
            for _, truth_row in missed_variants.iterrows():
                truth_matching_results.append({
                    'haplotype': haplotype,
                    'groupA': groupA,
                    'groupB': groupB,
                    'match_status': 'False Negative',
                    'truth_id': truth_row['ID'],
                    'truth_position': truth_row['pos'],
                    'truth_cpra': truth_row['cpra'] if 'cpra' in truth_row else None,
                    'cluster_id': None, 'n_points': None, 'density': None, 
                    'total_score': None, 'cluster_start': None, 'cluster_end': None, 
                    'dist_truth_to_cluster_center': None
                })
        
        n_variants_found = len(found_truth_ids_this_pair)
        
        precision = n_clusters_tp / n_clusters_total if n_clusters_total > 0 else 0.0
        recall = n_variants_found / n_variants_total if n_variants_total > 0 else 0.0
        
        f1_score = 0.0
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
            
        metrics_list.append({
            'haplotype': haplotype,
            'groupA': groupA,
            'groupB': groupB,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1_score,
            'Total_Clusters': n_clusters_total,
            'TP_Clusters': n_clusters_tp,
            'FP_Clusters': n_clusters_total - n_clusters_tp,
            'Total_Variants': n_variants_total,
            'Found_Variants': n_variants_found,
            'Missed_Variants': n_variants_total - n_variants_found
        })
        
        print(f"  📊 Metrics: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}")

    truth_match_df = pd.DataFrame()
    metrics_df = pd.DataFrame()

    if truth_matching_results:
        truth_match_df = pd.DataFrame(truth_matching_results)
        truth_match_df['matching_timestamp'] = pd.Timestamp.now()
        
        cols_priority = [
            'haplotype', 'groupA', 'groupB', 'match_status', 
            'cluster_id', 'truth_id', 'truth_position', 'total_score'
        ]
        cols = [c for c in cols_priority if c in truth_match_df.columns] + \
               [c for c in truth_match_df.columns if c not in cols_priority]
        truth_match_df = truth_match_df[cols]

    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        
    return truth_match_df, metrics_df


def parse_pairs(pairs_str: str) -> list:
    """
    Parse pairs string like "1-2,1-3,2-3" into list of tuples [(1,2), (1,3), (2,3)]
    """
    pairs = []
    for pair in pairs_str.split(','):
        pair = pair.strip()
        if '-' in pair:
            a, b = pair.split('-')
            pairs.append((int(a.strip()), int(b.strip())))
        elif 'vs' in pair.lower():
            a, b = pair.lower().split('vs')
            pairs.append((int(a.strip()), int(b.strip())))
    return pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HBB attention differential analysis pipeline."
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory to save intermediate and final results.",
    )
    parser.add_argument(
        "--cluster_dir",
        type=str,
        required=True,
        help="Directory to save cluster results.",
    )
    parser.add_argument(
        "--figure_dir",
        type=str,
        required=True,
        help="Directory to save plots and diagnostic figures.",
    )
    parser.add_argument(
        "--true-sites",
        type=str,
        default=None,
        help="determine which true site will be loaded"
    )
    parser.add_argument(
        "--score-percent",
        type=float,
        default=None,
        help="Top X%% score threshold for matching clusters to truth.",
    )
    parser.add_argument(
        "--strand",
        type=str,
        default='+',
        help="Direction to extend clusters for matching ('+' or '-').",
    )
    parser.add_argument(
        "--plot-mode",
        type=str,
        default="combined",
        choices=["combined", "two_panel"],
        help="Plot mode: 'combined' (single plot with dual y-axis) or 'two_panel' (separate track).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the plots (replaces 'Differential Attention Trace').",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default="1-2,1-3,2-3",
        help="Group pairs to analyze, format: '1-2,1-3,2-3' or '1vs2,1vs3,2vs3'. Default: '1-2,1-3,2-3'",
    )
    parser.add_argument(
        "--haplotype-comparison",
        action="store_true",
        help="Generate haplotype comparison plots (hap1 vs hap2 aligned).",
    )
    parser.add_argument(
        "--gwas-file",
        type=str,
        default=None,
        help="Path to GWAS result CSV file (must contain 'POS' column).",
    )

    args = parser.parse_args()
    os.makedirs(args.cluster_dir, exist_ok=True)

    return args
    
    
if __name__ == "__main__":
    args = parse_args()
    
    RESULTS_DIR = args.results_dir
    CLUSTER_DIR = args.cluster_dir
    FIGURE_DIR = args.figure_dir
    PLOT_MODE = args.plot_mode
    TITLE = args.title
    PAIRS = parse_pairs(args.pairs)
    HAPLOTYPE_COMPARISON = args.haplotype_comparison
    
    print(f"📋 Analyzing pairs: {PAIRS}")
    
    # --------------------------- 准备数据 --------------------------- #
    print("准备cluster数据 ...")
    res_hap1_cont = pd.read_csv(f"{RESULTS_DIR}/hap1_continuous_attention_differential.csv")
    res_hap2_cont = pd.read_csv(f"{RESULTS_DIR}/hap2_continuous_attention_differential.csv")
    hap1_scores = pd.read_csv(f"{RESULTS_DIR}/hap1_separability_scores.csv")
    hap2_scores = pd.read_csv(f"{RESULTS_DIR}/hap2_separability_scores.csv")
    
    LEN = res_hap1_cont.shape[0] / 1000
    LFC_QUANTILE_THRESH = 1 - ((4*0.05)/(LEN))*math.log10(LEN/4+9)
    PADJ_THRESH = 0.01
    
    # --------------------------- 准备真值 --------------------------- #
    print("准备真值 ...")
    true_roi = None
    if args.true_sites is not None:
        true_roi = pd.read_csv(f"{args.true_sites}")
        
    # --------------------------- [NEW] 准备 GWAS 数据 --------------------------- #
    print("准备 GWAS 数据 ...")
    gwas_data = None
    if args.gwas_file is not None:
        try:
            # 读取 GWAS 文件
            gwas_data = load_data(args.gwas_file)
            print(f"  ✅ 成功加载 GWAS 数据，共 {len(gwas_data)} 行")
            
            # 简单的列名检查
            if 'POS' not in gwas_data.columns and 'pos' not in gwas_data.columns:
                print("  ⚠️ 警告：GWAS 文件中未找到 'POS' 或 'pos' 列，绘图时可能无法显示 GWAS 结果。")
                
        except Exception as e:
            print(f"  ❌ 加载 GWAS 文件失败: {e}")
    
    # --------------------------- 聚类分析寻找致病位点 --------------------------- #
    print("聚类分析寻找致病位点...")

    all_cluster_summaries = []
    all_cluster_summaries_success = []
    all_sig_dfs = {}
    all_candidates = []

    os.makedirs(CLUSTER_DIR, exist_ok=True)

    for haplotype, diff_cont in [("hap1", res_hap1_cont), ("hap2", res_hap2_cont)]:
        print(f"\n{'='*80}")
        print(f"开始分析 {haplotype} 的所有组别对")
        print(f"{'='*80}")
    
        for gA, gB in PAIRS:
            print(f"\n{'-'*60}")
            print(f"[{haplotype}] 处理组对: {gA} vs {gB}")
            print(f"{'-'*60}")
        
            mask_pair = (diff_cont["groupA"] == gA) & (diff_cont["groupB"] == gB)
            if not mask_pair.any():
                print(f"  ⚠️  警告：该组别对不存在数据，跳过")
                all_cluster_summaries.append({
                    "haplotype": haplotype,
                    "groupA": gA,
                    "groupB": gB,
                    "status": "skipped_no_data",
                    "n_significant": 0,
                    "n_clusters": 0,
                    "top_score": np.nan
                })
                continue

            try:
                cluster_summary, sig_df = cluster_significant_cohen_positions(
                    diff_df=diff_cont,
                    score_df=hap1_scores if haplotype=="hap1" else hap2_scores,
                    truth_df=true_roi,
                    gwas_df=gwas_data,
                    haplotype=haplotype,
                    group_pair=(gA, gB),
                    adjp_threshold=PADJ_THRESH,
                    lfc_quantile=LFC_QUANTILE_THRESH,
                    cohen_threshold=0.2,
                    eps=20,
                    min_samples=5,
                    weighted_distance=False,
                    plot_diagnostics=True,
                    fig_dir=CLUSTER_DIR,
                    plot_mode=PLOT_MODE,
                    title=TITLE,  # Pass custom title,
                    extend_strand=args.strand,
                )
            
                if cluster_summary.empty:
                    print(f"  ⚠️  警告：聚类后未找到有效簇")
                    all_cluster_summaries.append({
                        "haplotype": haplotype,
                        "groupA": gA,
                        "groupB": gB,
                        "status": "skipped_no_clusters",
                        "n_significant": len(sig_df) if not sig_df.empty else 0,
                        "n_clusters": 0,
                        "top_score": np.nan
                    })
                
                    if not sig_df.empty:
                        sig_df.to_csv(
                            f"{CLUSTER_DIR}/{haplotype}_sig_positions_no_clusters_{gA}vs{gB}.csv",
                            index=False
                        )
                    continue
            
                print(f"  ✅ 成功：发现 {len(cluster_summary)} 个聚类")

                cluster_summary["haplotype"] = haplotype
                cluster_summary["groupA"] = gA
                cluster_summary["groupB"] = gB
                meta_cols = ['haplotype', 'groupA', 'groupB']
                other_cols = [col for col in cluster_summary.columns if col not in meta_cols]
                cluster_summary = cluster_summary[meta_cols + other_cols]
                all_cluster_summaries_success.append(cluster_summary)

                all_cluster_summaries.append({
                    "haplotype": haplotype,
                    "groupA": gA,
                    "groupB": gB,
                    "status": "success",
                    "n_significant": len(sig_df),
                    "n_clusters": len(cluster_summary),
                    "top_score": cluster_summary["total_score"].max()
                })
                all_sig_dfs[f"{haplotype}_{gA}vs{gB}"] = sig_df
            
            except Exception as e:
                print(f"  ❌ 错误：聚类分析失败 - {str(e)}")
                import traceback
                traceback.print_exc()
            
                all_cluster_summaries.append({
                    "haplotype": haplotype,
                    "groupA": gA,
                    "groupB": gB,
                    "status": "error",
                    "error_message": str(e),
                    "n_significant": 0,
                    "n_clusters": 0,
                    "top_score": np.nan
                })
                continue

    print(f"\n{'='*80}")
    print("聚类分析完成，汇总结果...")
    print(f"{'='*80}")

    summary_df = pd.DataFrame(all_cluster_summaries)
    cluster_summary_success_df = pd.concat(all_cluster_summaries_success, ignore_index=True) if all_cluster_summaries_success else pd.DataFrame()

    summary_df["analysis_timestamp"] = pd.Timestamp.now()
    cluster_summary_success_df["analysis_timestamp"] = pd.Timestamp.now()

    summary_path = f"{CLUSTER_DIR}/cluster_analysis_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n📊 聚类分析汇总表已保存: {summary_path}")

    cluster_summary_success_path = f"{CLUSTER_DIR}/cluster_analysis_summary_success.csv"
    cluster_summary_success_df.to_csv(cluster_summary_success_path, index=False)
    print(f"\n📊 成功聚类结果汇总表已保存: {cluster_summary_success_path}")

    # --------------------------- 生成 Haplotype 对比图 --------------------------- #
    if HAPLOTYPE_COMPARISON:
        print(f"\n{'='*80}")
        print("生成 Haplotype 对比图...")
        print(f"{'='*80}")
        
        for gA, gB in PAIRS:
            print(f"\n生成 group {gA} vs {gB} 的 haplotype 对比图...")
            
            sig_df_hap1 = all_sig_dfs.get(f"hap1_{gA}vs{gB}")
            sig_df_hap2 = all_sig_dfs.get(f"hap2_{gA}vs{gB}")
            
            cluster_summary_hap1 = cluster_summary_success_df[
                (cluster_summary_success_df["haplotype"] == "hap1") &
                (cluster_summary_success_df["groupA"] == gA) &
                (cluster_summary_success_df["groupB"] == gB)
            ] if not cluster_summary_success_df.empty else pd.DataFrame()
            
            cluster_summary_hap2 = cluster_summary_success_df[
                (cluster_summary_success_df["haplotype"] == "hap2") &
                (cluster_summary_success_df["groupA"] == gA) &
                (cluster_summary_success_df["groupB"] == gB)
            ] if not cluster_summary_success_df.empty else pd.DataFrame()
            
            if sig_df_hap1 is not None or sig_df_hap2 is not None:
                plot_haplotype_comparison(
                    diff_df_hap1=res_hap1_cont,
                    diff_df_hap2=res_hap2_cont,
                    score_df_hap1=hap1_scores,
                    score_df_hap2=hap2_scores,
                    truth_df=true_roi,
                    sig_df_hap1=sig_df_hap1,
                    sig_df_hap2=sig_df_hap2,
                    cluster_summary_hap1=cluster_summary_hap1,
                    cluster_summary_hap2=cluster_summary_hap2,
                    gwas_df=gwas_data,
                    group_pair=(gA, gB),
                    fig_dir=CLUSTER_DIR,
                    title=TITLE,  # Pass custom title
                    extend_strand=args.strand,
                )
            else:
                print(f"  ⚠️ 跳过：没有找到 group {gA} vs {gB} 的 sig_df 数据")

    # --------------------------- 高置信位点聚类匹配真值 --------------------------- #
    print("开始匹配真值位点...")
    if true_roi is None or true_roi.empty:
        print("⚠️ 警告：未提供真值位点文件，跳过真值匹配步骤。")
        sys.exit(0)
        
    matched_df, metrics_df = match_clusters_to_truth(all_sig_dfs, cluster_summary_success_df, true_roi, 
                                                     score_percent=None, min_n_points=None, 
                                                     min_density=None, extension_bp=10, direction=args.strand)
    matched_path = f"{CLUSTER_DIR}/cluster_truth_matching_results.csv"
    matched_df.to_csv(matched_path, index=False)
    metrics_path = f"{CLUSTER_DIR}/cluster_truth_matching_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✅ 真值匹配结果已保存: {matched_path}")
    print("分析流程全部完成!")