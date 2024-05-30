import matplotlib.pyplot as plt

# ccr-s数据
# test_years = ["2010-2012/2013", "2011-2013/2014", "2012-2014/2015", "2013-2015/2016"]
# train_samples = [382, 655, 947, 1058]
# test_samples = [312, 383, 480, 428]
# train_pos_neg_ratios = [1.616, 1.549, 1.519, 1.549]
# test_pos_neg_ratios = [1.4, 1.471, 1.652, 0.853]
# unseen_percentages = [0.4231, 0.2402, 0.1750, 0.2360]

# ccr-l数据
test_years = ["2010-2012/2013", "2011-2013/2014", "2012-2014/2015", "2013-2015/2016"]
train_samples = [1294, 2505, 3962, 4854]
test_samples = [1387, 1944, 2062, 975]
train_pos_neg_ratios = [1.562, 1.75, 1.95, 2.138]
test_pos_neg_ratios = [1.914, 2.273, 2.177, 1.257]
unseen_percentages = [0.5451, 0.1178, 0.0946, 0.1990]

# 创建图形和轴
fig, ax1 = plt.subplots(figsize=(6.5, 5))
ax2 = ax1.twinx()

# 定义柔和的颜色
color1 = '#002c53'  # 学术蓝
color2 = '#ffa510'  # 学术绿
color3 = '#0c84c6'  # 学术红
color4 = '#ffbd66'  # 学术紫
color5 = '#f74d4d'  # 学术黄

# 绘制柱状图
bar_width = 0.35
bars1 = ax1.bar([x - bar_width/2 for x in range(len(test_years))], train_samples, width=bar_width, color=color1, alpha=0.8, label='Training set samples')
bars2 = ax1.bar([x + bar_width/2 for x in range(len(test_years))], test_samples, width=bar_width, color=color2, alpha=0.8, label='Test set samples')

# 绘制折线图
line1, = ax2.plot(range(len(test_years)), train_pos_neg_ratios, marker='o', linestyle='-', color=color3, label='Training set pos/neg ratio')
line2, = ax2.plot(range(len(test_years)), test_pos_neg_ratios, marker='^', linestyle='-', color=color4, label='Test set pos/neg ratio')
line3, = ax2.plot(range(len(test_years)), unseen_percentages, marker='s', linestyle='-', color=color5, label='Unseen corporations ratio')

# 设置x轴刻度和标签
ax1.set_xticks(range(len(test_years)))
ax1.set_xticklabels(test_years)
ax1.set_xlabel('Training/testing years')

# 设置y轴标签
ax1.set_ylabel('Number of samples')
ax2.set_ylabel('Ratio')

# 添加图例
handles = [bars1, bars2, line1, line2, line3]
labels = [h.get_label() for h in handles]

# 将图例放在图的下面，两行显示
ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)

# 调整布局并显示图表
plt.tight_layout()
plt.show()