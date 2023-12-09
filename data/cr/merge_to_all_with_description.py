import pandas as pd

# 读取每个csv文件
test = pd.read_csv('test(with_description_col).csv')
train = pd.read_csv('train(with_description_col).csv')
val = pd.read_csv('val(with_description_col).csv')

# 将它们合并到一个DataFrame
combined = pd.concat([test, train, val])

# 将合并后的DataFrame保存为新的csv文件
combined.to_csv('all(with_description_col).csv', index=False)