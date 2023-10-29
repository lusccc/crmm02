import pandas as pd
from sklearn.model_selection import train_test_split

dataset = 'cr2'
# 加载数据
data = pd.read_csv(f'./{dataset}/all.csv', header=0)

# 先划分出80%的训练集+验证集和20%的测试集
train_val, test = train_test_split(data, test_size=0.2, random_state=42)

# 再从训练集+验证集中划分出75%的训练集和25%的验证集，这样验证集就占总数据的10%, note shuffle=False
train, val = train_test_split(train_val, test_size=0.125, shuffle=False)  # 0.125 x 0.8 = 0.1

# 保存训练集，验证集和测试集
train.to_csv(f'./{dataset}/train.csv', index=False)
val.to_csv(f'./{dataset}/val.csv', index=False)
test.to_csv(f'./{dataset}/test.csv', index=False)