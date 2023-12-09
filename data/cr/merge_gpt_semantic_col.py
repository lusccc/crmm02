import pandas as pd
# data_type = 'val'
# data_type = 'train'
data_type = 'test'
train_data = pd.read_csv(f'{data_type}.csv')
description_data = pd.read_csv(f'{data_type}_description.csv', header=0)

# 提取GPT_description列
gpt_description = description_data['GPT_description']

# 将GPT_description列添加到train_data的末尾
train_data['GPT_description'] = gpt_description

# 保存修改后的train_data到新的CSV文件
train_data.to_csv(f'{data_type}(with_description_col).csv', index=False)