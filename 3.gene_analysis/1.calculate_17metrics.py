import pandas as pd
import numpy as np
import os
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from scipy.stats import entropy, iqr, skew, kurtosis
from scipy.signal import find_peaks

root_path = "/mnt/zzbnew/peixunban/changan/try2/AS_result"
save_path = "/mnt/zzbnew/peixunban/changan/try2/plot_resulets/"

os.makedirs(save_path, exist_ok=True)

# ========== 定义辅助函数 ==========
def get_percent_mean(x, percent=0.05, top=True):
    """获取前/后 percent 百分比数据的均值"""
    x_clean = x.dropna()
    if len(x_clean) == 0:
        return np.nan
    n = max(1, int(len(x_clean) * percent))
    if top:
        return x_clean.nlargest(n).mean()
    else:
        return x_clean.nsmallest(n).mean()

def analyze_peaks(x):
    """分析峰值：峰值数量、峰值密度、峰值均值"""
    x_clean = x.dropna().values
    if len(x_clean) < 3:
        return pd.Series([0, 0, np.nan])
    
    # 找峰值（高度至少为均值）
    peaks, _ = find_peaks(x_clean, height=x_clean.mean())
    peak_count = len(peaks)
    peak_density = peak_count / len(x_clean) if len(x_clean) > 0 else 0
    peak_mean = x_clean[peaks].mean() if peak_count > 0 else np.nan
    
    return pd.Series([peak_count, peak_density, peak_mean])

def calc_entropy(x):
    """计算香农熵"""
    x_clean = x[x > 0].dropna()
    if len(x_clean) == 0:
        return 0
    probs = x_clean / x_clean.sum()
    return entropy(probs)

# 获取基因文件夹
genes = sorted([
    d for d in os.listdir(root_path)
    if os.path.isdir(os.path.join(root_path, d)) and d != "1.plot"
])
print("找到基因文件夹：", genes)

results = []

for gene in genes:
    gene_dir = os.path.join(root_path, gene)
    file_hap1 = os.path.join(gene_dir, "hap1_attention_collapsed.csv")#定义使用基因的hap1_attention_collapsed.csv文件
    file_meta = os.path.join(gene_dir, "metadata.csv")

    if not (os.path.exists(file_hap1) and os.path.exists(file_meta)):
        print(f"⚠️ 跳过 {gene}（缺少 CSV 文件）")
        continue

    # 读取数据
    df = pd.read_csv(file_hap1)
    metadata = pd.read_csv(file_meta)

    # ========== 计算所有统计指标 ==========
    value_cols = df.columns[1:]
    data_values = df[value_cols]
    
    result = pd.DataFrame({"sample": df["sample"]})
    
    # 基础统计量
    result["mean"] = data_values.mean(axis=1)
    result["max"] = data_values.max(axis=1)
    result["std"] = data_values.std(axis=1)
    result["cv"] = result["std"] / result["mean"]
    result["median"] = data_values.median(axis=1)
    result["iqr"] = data_values.apply(lambda x: iqr(x, nan_policy='omit'), axis=1)
    result["percentile_90"] = data_values.quantile(0.9, axis=1)
    result["percentile_10"] = data_values.quantile(0.1, axis=1)
    
    # 众数
    result["mode"] = data_values.apply(
        lambda x: stats.mode(x.dropna(), keepdims=True, nan_policy='omit')[0][0] 
        if not x.dropna().empty else np.nan, axis=1
    )
    
    # 偏度与峰度
    result["skewness"] = data_values.apply(lambda x: skew(x, nan_policy='omit'), axis=1)
    result["kurtosis"] = data_values.apply(lambda x: kurtosis(x, nan_policy='omit'), axis=1)
    
    # 自定义复杂指标
    result["top5_percent_mean"] = data_values.apply(lambda x: get_percent_mean(x, 0.05, True), axis=1)
    result["low5_percent_mean"] = data_values.apply(lambda x: get_percent_mean(x, 0.05, False), axis=1)
    result[["peak_count", "peak_density", "peak_mean"]] = data_values.apply(analyze_peaks, axis=1)
    result["shannon_entropy"] = data_values.apply(calc_entropy, axis=1)

    # 合并 metadata 并映射分组
    result = result.merge(metadata, on="sample", how="left")
    result["group"] = result["sample_type"].map({0: "control", 2: "case"})

    # ========== 对所有指标做 U 检验 ==========
    metrics = [
        "mean", "max", "std", "cv", "median", "mode", "iqr", "skewness", "kurtosis",
        "top5_percent_mean", "low5_percent_mean", "percentile_90", "percentile_10",
        "peak_count", "peak_density", "peak_mean", "shannon_entropy"
    ]
    
    p_values = [gene]

    for metric in metrics:
        case_data = result[result["group"] == "case"][metric].dropna()
        ctrl_data = result[result["group"] == "control"][metric].dropna()
        if len(case_data) > 0 and len(ctrl_data) > 0:
            _, p_val = stats.mannwhitneyu(case_data, ctrl_data, alternative='two-sided')
        else:
            p_val = np.nan
        p_values.append(p_val)

    results.append(p_values)
    print(f"✓ 完成 {gene}")

# ========== 生成结果表 ==========
columns = ["gene"] + [f"{m}_p" for m in metrics]
df_out = pd.DataFrame(results, columns=columns)

# ========== BH correction（多重检验校正）==========
print("\n开始进行 BH correction...")
for metric in metrics:
    p_col = f"{metric}_p"
    # 过滤掉 NaN 值
    valid_mask = ~df_out[p_col].isna()
    valid_pvals = df_out.loc[valid_mask, p_col]
    
    if len(valid_pvals) > 0:
        # 进行 BH correction
        rejected, corrected_pvals, _, _ = multipletests(
            valid_pvals, 
            alpha=0.05, 
            method='fdr_bh'
        )
        # 创建新列存储校正后的 p 值
        df_out.loc[valid_mask, f"{metric}_p_corrected"] = corrected_pvals
        df_out.loc[valid_mask, f"{metric}_significant"] = rejected
    else:
        df_out[f"{metric}_p_corrected"] = np.nan
        df_out[f"{metric}_significant"] = False

# 按 std_p 排序
df_out = df_out.sort_values("std_p")

# 保存结果
out_file = os.path.join(save_path, "17stats_pvalues_with_BH_correction.csv") #保存结果文件路径
df_out.to_csv(out_file, index=False)
print(f"\n🎉 结果已保存至：{out_file}")

# ========== 输出统计摘要 ==========
print("\n" + "="*80)
print("统计摘要")
print("="*80)
print(f"总基因数: {len(df_out)}")

for metric in metrics:
    p_col = f"{metric}_p"
    p_corr_col = f"{metric}_p_corrected"
    sig_col = f"{metric}_significant"
    
    # 原始 p < 0.05 的基因数
    uncorrected_sig = df_out[df_out[p_col] < 0.05].shape[0]
    # BH 校正后显著的基因数
    corrected_sig = df_out[df_out[sig_col] == True].shape[0]
    
    print(f"\n{metric.upper()}:")
    print(f"  原始 P < 0.05: {uncorrected_sig} 个基因")
    print(f"  BH 校正后显著: {corrected_sig} 个基因")

# ========== 输出显著基因列表 ==========
print("\n" + "="*80)
print("BH 校正后显著基因列表")
print("="*80)

for metric in metrics:
    sig_col = f"{metric}_significant"
    p_col = f"{metric}_p"
    p_corr_col = f"{metric}_p_corrected"
    
    sig_genes = df_out[df_out[sig_col] == True].sort_values(p_corr_col)
    
    if not sig_genes.empty:
        print(f"\n🔹 {metric.upper()} 显著基因 ({len(sig_genes)} 个):")
        for _, row in sig_genes.iterrows():
            print(f"  {row['gene']:<20}  原始p = {row[p_col]:.2e}  校正p = {row[p_corr_col]:.2e}")
    else:
        print(f"\n🔹 {metric.upper()}: 无显著基因")

print("\n" + "="*80)
print("前10个基因（按 std_p 排序）：")
print("="*80)
print(df_out.head(10).to_string(index=False))