import pandas as pd

# 加载数据
data = pd.read_csv('all.csv')

# 定义映射字典
rating_dict = {'AAA': 1, 'AA': 1, 'A': 1, 'BBB': 1, 'BB': 0, 'B': 0, 'CCC': 0, 'CC': 0, 'C': 0, 'D': 0}

# 使用映射字典创建新列
data['binaryRating'] = data['Rating'].map(rating_dict)

# 保存新的数据集
data.to_csv('all.csv', index=False)