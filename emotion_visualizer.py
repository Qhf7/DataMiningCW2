import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 情绪数据
emotions = [
    "sadness (悲伤)", 
    "hopelessness (绝望)", 
    "worthlessness (无价值感)", 
    "loneliness (孤独)", 
    "self hate (自我厌恶)", 
    "emptiness (空虚)", 
    "lack of energy (缺乏能量)", 
    "anger (愤怒)", 
    "suicide intent (自杀意图)", 
    "brain dysfunction/forget (大脑功能障碍)"
]

counts = [69, 54, 46, 37, 35, 31, 27, 24, 16, 12]
percentages = [20.4, 16.0, 13.6, 10.9, 10.4, 9.2, 8.0, 7.1, 4.7, 3.6]

# 创建专业的颜色映射
colors = ['#1f77b4', '#2a5caa', '#3676c8', '#4a8ddc', '#5fa3ed', 
          '#74baff', '#89d0ff', '#9ee6ff', '#b3fcff', '#c8ffff']

# 创建水平条形图
fig, ax = plt.figure(figsize=(14, 10)), plt.axes()
bars = ax.barh(emotions, counts, color=colors)

# 设置样式
ax.set_facecolor('#f5f5f5')
for spine in ax.spines.values():
    spine.set_visible(False)
    
# 添加数据标签
for i, (bar, count, percentage) in enumerate(zip(bars, counts, percentages)):
    ax.text(count + 1, bar.get_y() + bar.get_height()/2, 
            f"{count} ({percentage:.1f}%)", 
            va='center', ha='left', 
            fontsize=12, fontweight='bold', color='#333333')

# 添加标题和标签
ax.set_title("抑郁症数据集中的情绪分布分析", fontsize=22, pad=20, fontweight='bold', color='#333333')
ax.set_xlabel("出现次数", fontsize=16, labelpad=10, fontweight='bold', color='#555555')
ax.set_ylabel("情绪类型", fontsize=16, labelpad=10, fontweight='bold', color='#555555')

# 添加网格线
ax.grid(axis='x', linestyle='--', alpha=0.3, color='#888888')
ax.set_axisbelow(True)

# 调整刻度标签样式
ax.tick_params(axis='both', which='major', labelsize=13, colors='#555555')

# 添加信息框
info_text = (
    "数据集统计:\n"
    "总记录数: 80条\n"
    "总情绪标记数: 351个\n"
    "平均每条记录的情绪标记数: 4.39个"
)
plt.figtext(0.15, 0.02, info_text, fontsize=14, 
            bbox=dict(facecolor='#f0f0f0', alpha=0.8, boxstyle='round,pad=0.5',
                     edgecolor='#cccccc'), color='#333333', fontweight='bold')

# 保存条形图
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("emotion_distribution_chart.png", dpi=300, bbox_inches='tight')
print("条形图已保存为 emotion_distribution_chart.png")

# 创建饼图
plt.figure(figsize=(12, 10))

# 前5种情绪的数据
top_emotions = [e.split(' ')[0] for e in emotions[:5]]  # 仅使用英文部分作为标签
top_emotions_full = emotions[:5]
top_counts = counts[:5]
other_count = sum(counts[5:])

# 准备饼图数据
pie_labels = top_emotions + ["其他情绪"]
pie_counts = top_counts + [other_count]
pie_percentages = [count/sum(counts)*100 for count in pie_counts]

# 创建自定义颜色方案
pie_colors = ['#1f77b4', '#2a5caa', '#3676c8', '#4a8ddc', '#5fa3ed', '#e0e0e0']

# 突出显示最大的部分
explode = [0.05, 0, 0, 0, 0, 0]  # 突出显示第一部分

# 绘制饼图
wedges, texts, autotexts = plt.pie(
    pie_counts, 
    labels=None,
    autopct='%1.1f%%',
    startangle=90,
    colors=pie_colors,
    explode=explode,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True},
    shadow=True
)

# 调整自动百分比文本
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

# 添加图例
legend_labels = [f"{emotions[i].split('(')[1].strip(')').strip()} ({count})" for i, count in enumerate(top_counts)]
legend_labels.append(f"其他情绪 ({other_count})")

plt.legend(
    wedges, 
    legend_labels,
    title="情绪类型 (出现次数)",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=12,
    title_fontsize=14
)

plt.title("抑郁症数据集中的主要情绪分布", fontsize=20, pad=20, fontweight='bold')

# 添加注释
plt.annotate(
    f"数据来源: 抑郁症相关帖子分析\n总样本数: 80条记录",
    xy=(-0.1, -0.1), xycoords='axes fraction',
    fontsize=12, ha='left', va='bottom',
    bbox=dict(boxstyle="round,pad=0.5", fc="#f0f0f0", ec="#cccccc", alpha=0.8)
)

plt.tight_layout()
plt.savefig("emotion_pie_chart.png", dpi=300, bbox_inches='tight')
print("饼图已保存为 emotion_pie_chart.png")

# 堆叠柱状图 - 展示情绪共现关系
plt.figure(figsize=(14, 8))
ax = plt.axes()

# 定义情绪共现组合及其数量
emotion_pairs = [
    ("悲伤+绝望", 42), 
    ("悲伤+无价值感", 35),
    ("孤独+悲伤", 28),
    ("自我厌恶+无价值感", 25),
    ("空虚+悲伤", 22),
    ("缺乏能量+绝望", 19),
    ("愤怒+自我厌恶", 17),
    ("自杀意图+绝望", 14)
]

pair_labels = [pair[0] for pair in emotion_pairs]
pair_counts = [pair[1] for pair in emotion_pairs]

# 创建渐变颜色
gradient_colors = plt.cm.YlOrRd(np.linspace(0.3, 0.8, len(emotion_pairs)))

# 绘制柱状图
bars = ax.bar(pair_labels, pair_counts, color=gradient_colors, width=0.6, 
          edgecolor='white', linewidth=1.5)

# 设置样式
ax.set_facecolor('#f5f5f5')
for spine in ax.spines.values():
    spine.set_visible(False)

# 添加数据标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# 添加标题和标签
ax.set_title("抑郁症数据集中的情绪共现关系", fontsize=20, pad=20, fontweight='bold')
ax.set_xlabel("情绪组合", fontsize=16, labelpad=10, fontweight='bold')
ax.set_ylabel("共同出现次数", fontsize=16, labelpad=10, fontweight='bold')

# 添加网格线
ax.grid(axis='y', linestyle='--', alpha=0.3, color='#888888')
ax.set_axisbelow(True)

# 调整刻度标签
plt.xticks(rotation=30, ha='right', fontsize=12)
plt.yticks(fontsize=12)

# 添加注释
plt.annotate(
    "情绪共现数据显示抑郁症患者往往同时经历多种负面情绪",
    xy=(0.5, -0.15), xycoords='axes fraction',
    fontsize=13, ha='center', va='bottom',
    bbox=dict(boxstyle="round,pad=0.5", fc="#f0f0f0", ec="#cccccc", alpha=0.8)
)

plt.tight_layout()
plt.savefig("emotion_cooccurrence_chart.png", dpi=300, bbox_inches='tight')
print("共现关系图已保存为 emotion_cooccurrence_chart.png")

# 显示所有图表
plt.show() 