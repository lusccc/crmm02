import glob

import pandas as pd
import numpy as np

# 假设你的Excel文件命名为 data1.xlsx, data2.xlsx, ..., data10.xlsx
file_names = glob.glob('./my_file/*.xlsx')

all_data = []

# 读取每个Excel文件，并将它们追加到all_data列表中
for file in file_names:
    df = pd.read_excel(file)
    all_data.append(df)

# 获取所有不同的模型名称
models = pd.concat(all_data)['model'].unique()

# 最终汇总的DataFrame列表
final_dfs = []

for model in models:
    # 筛选出每种模型的所有重复实验结果
    model_dfs = [df[df['model'] == model] for df in all_data]

    # 将同一模型的所有重复实验结果合并（按行）
    model_df_concat = pd.concat(model_dfs, ignore_index=True)

    # 计算平均值，忽略'Model'列
    avg_values = model_df_concat.mean(numeric_only=True)
    avg_values['model'] = f'{model} Average'  # 为平均行设置模型名称

    # 将平均值转换为DataFrame
    avg_df = pd.DataFrame([avg_values])

    # 将原始数据和平均数据合并
    model_final_df = pd.concat([model_df_concat, avg_df], ignore_index=True)

    # 将合并后的数据添加到最终的DataFrame列表中
    final_dfs.append(model_final_df)

# 将所有模型的数据合并为一个DataFrame
final_df = pd.concat(final_dfs, ignore_index=True)

# 创建一个Excel写入器
with pd.ExcelWriter('combined_results_with_averages.xlsx') as writer:
    # 将结果写入到名为 'Results with Averages' 的sheet中
    final_df.to_excel(writer, sheet_name='Results with Averages', index=False)