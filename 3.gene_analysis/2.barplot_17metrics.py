import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df = pd.read_csv('/mnt/zzbnew/peixunban/changan/try2/plot_resulets/5genes_17stats_pvalues_with_BH_correction.csv')

# 提取基因名（只保留 reverse_ 后面的部分）
def extract_gene_name(full_name):
    full_name_str = str(full_name)
    if 'reverse_' in full_name_str:
        return full_name_str.split('reverse_')[1]
    elif 'forward_' in full_name_str:
        return full_name_str.split('forward_')[1]
    else:
        return full_name_str

df['gene_short'] = df['gene'].apply(extract_gene_name)

# 定义17个统计指标
metrics = ['std_p_corrected', 'mean_p_corrected', 'max_p_corrected', 'cv_p_corrected', 
           'median_p_corrected', 'mode_p_corrected', 'iqr_p_corrected','skewness_p_corrected',
           'kurtosis_p_corrected','top5_percent_mean_p_corrected', 'low5_percent_mean_p_corrected','percentile_90_p_corrected',
           'percentile_10_p_corrected','peak_count_p_corrected','peak_density_p_corrected','peak_mean_p_corrected',  
           'shannon_entropy_p_corrected']

# 创建图形
fig, axes = plt.subplots(6, 3, figsize=(24, 22))
axes = axes.flatten()

# P值阈值
threshold = -np.log10(0.05)

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    
    # ✅ 只筛选 P < 0.05 的显著基因，并按P值从小到大排序
    current_data = df[df[metric] < 0.05].sort_values(by=metric, ascending=True).copy()
    
    # 如果没有显著基因，显示提示信息
    if len(current_data) == 0:
        ax.text(0.5, 0.5, f'No significant genes\n(P < 0.05)', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_xlabel('-log10(P-value)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Genes', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.upper()} P-value (Significant Genes)', fontsize=13, fontweight='bold')
        continue
    
    # 计算-log10(P值)
    neg_log10_values = -np.log10(current_data[metric])
    
    # ✅ 所有基因都是显著的（P < 0.05），统一用红色
    colors = ['#e74c3c'] * len(current_data)
    
    # 绘制水平柱状图
    bars = ax.barh(range(len(current_data)), neg_log10_values, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # 添加P=0.05的参考线
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label='P=0.05')
    
    ax.set_xlabel('-log10(P-value)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Genes', fontsize=11, fontweight='bold')
    ax.set_title(f'{metric.upper()} P-value ({len(current_data)} Significant Genes)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # ✅ 设置y轴标签为简化后的基因名称
    ax.set_yticks(range(len(current_data)))
    ax.set_yticklabels(current_data['gene_short'].values, fontsize=10)
    ax.invert_yaxis()  # P值最小的（最显著的）在上面
    
    # 调整x轴标签字体大小
    ax.tick_params(axis='x', labelsize=9)
    
    # 如果基因数量很多，调整y轴字体大小
    if len(current_data) > 20:
        ax.tick_params(axis='y', labelsize=8)
    elif len(current_data) > 30:
        ax.tick_params(axis='y', labelsize=6)

    
    # ✅ 特殊处理 HBB 基因：红色加粗
    for tick in ax.get_yticklabels():
        label = tick.get_text()
        if label == 'HBB':
            tick.set_color('red')           # 红色文字
            tick.set_weight('bold')         # 加粗
            tick.set_fontsize(12)           # 字号稍大（从10增加到12）
        else:
            tick.set_color('black')         # 其他基因保持黑色
            tick.set_weight('normal')       # 正常粗细

    # ✅ 隐藏多余的空白子图（第8和第9个）
    for i in range(len(metrics), len(axes)):
        axes[i].axis('off')


plt.tight_layout()

# 保存图片
plt.savefig('/mnt/zzbnew/peixunban/changan/try2/plot_resulets/significant_genes_BH_17pvalue.png', dpi=300, bbox_inches='tight')
plt.savefig('/mnt/zzbnew/peixunban/changan/try2/plot_resulets/significant_genes_BH_17pvalue.pdf', bbox_inches='tight')

plt.show()

print("图片已保存为 'significant_genes_BH_17pvalue.png'")

# ✅ 输出每个指标的显著基因数量
print("\n显著基因统计 (P < 0.05):")
print("="*50)
for metric in metrics:
    sig_count = len(df[df[metric] < 0.05])
    print(f"{metric.upper()}: {sig_count} 个显著基因")