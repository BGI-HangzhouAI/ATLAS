import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from scipy.stats import entropy, iqr, skew, kurtosis
from scipy.signal import find_peaks

# ===== 路径设置 =====
hbb_path = "./AS_result/HBB"
save_path = "./plot_results/"

os.makedirs(save_path, exist_ok=True)

gene = "HBB"

# ===== 文件路径 =====
file_hap1 = os.path.join(hbb_path, "hap1_attention_collapsed.csv")
file_meta = os.path.join(hbb_path, "metadata.csv")

if not (os.path.exists(file_hap1) and os.path.exists(file_meta)):
    print(f"⚠️ 文件不存在：请检查路径")
    exit()

# ===== 辅助函数 =====
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

# ===== 1. 读取文件 =====
df = pd.read_csv(file_hap1)
metadata = pd.read_csv(file_meta)

# ===== 2. 计算统计指标 =====
value_cols = df.columns[1:]  # 第 1 列之后为注意力值
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

# 处理除以0的情况
result.loc[result["mean"] == 0, "cv"] = np.nan

# ===== 3. 合并 metadata =====
result = result.merge(metadata, on="sample", how="left")

# ===== 4. 定义分组 =====
result["group"] = result["sample_type"].map({0: "control", 2: "case"})

# ===== 5. 定义所有指标 =====
metrics = [
    "mean", "max", "std", "cv", "median", "mode", "iqr", "skewness", "kurtosis",
    "top5_percent_mean", "low5_percent_mean", "percentile_90", "percentile_10",
    "peak_count", "peak_density", "peak_mean", "shannon_entropy"
]

# ===== 6. U 检验 =====
p_values = {}
for metric in metrics:
    case_vals = result[result["group"] == "case"][metric].dropna()
    ctrl_vals = result[result["group"] == "control"][metric].dropna()
    if len(case_vals) > 0 and len(ctrl_vals) > 0:
        U, p = stats.mannwhitneyu(case_vals, ctrl_vals, alternative='two-sided')
    else:
        p = np.nan
    p_values[metric] = p

# ===== 7. 绘制箱线图 =====
# 创建 5x4 的子图布局（17个指标）
fig, axes = plt.subplots(3, 6, figsize=(20, 25))
axes = axes.flat  # 展平为一维数组

titles_map = {
    "mean": "Mean", "max": "Max", "std": "Standard Deviation", 
    "cv": "Coefficient of Variation", "median": "Median", "mode": "Mode",
    "iqr": "Interquartile Range", "skewness": "Skewness", "kurtosis": "Kurtosis",
    "top5_percent_mean": "Top 5% Mean", "low5_percent_mean": "Low 5% Mean",
    "percentile_90": "90th Percentile", "percentile_10": "10th Percentile",
    "peak_count": "Peak Count", "peak_density": "Peak Density", 
    "peak_mean": "Peak Mean", "shannon_entropy": "Shannon Entropy"
}

colors = {"control": "#3E7EDF", "case": "#f01414"}

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    
    data = [
        result[result["group"] == "case"][metric].dropna(),
        result[result["group"] == "control"][metric].dropna()
    ]

    bp = ax.boxplot(data, labels=["case", "control"], patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5))

    for patch, color in zip(bp['boxes'], [colors["case"], colors["control"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title(titles_map[metric], fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)

    # ===== p 值标注（显著为红色） =====
    p = p_values[metric]
    if not pd.isna(p):
        y_max = max([d.max() if len(d) > 0 else float('-inf') for d in data])
        color = "red" if p < 0.05 else "black"
        ax.text(1.5, y_max, f"p = {p:.2e}", ha="center", fontsize=9, color=color)

# 隐藏多余的子图
for idx in range(len(metrics), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle(f"{gene}: Distribution of 17 Metrics by Sample Type", 
             fontsize=16, fontweight="bold", y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

# ===== 保存图片 =====
out_file = os.path.join(save_path, f"{gene}_17metrics_hap1_box.pdf")
plt.savefig(out_file, dpi=200, bbox_inches='tight')
plt.close()

print(f"✓ 完成 {gene}，保存到 {out_file}")
print("🎉 HBB基因处理完成！")

# 打印p值结果
print(f"\n{gene} 的统计检验结果：")
for metric in metrics:
    p = p_values[metric]
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"{metric:20s}: p = {p:.2e} {sig}")