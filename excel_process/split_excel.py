import pandas as pd

# 读取Excel文件
df = pd.read_excel('hist_cr_multitask_rolling_window_#2013,2014,2015#_#2016#_clncpNoEnsemble_allrep.xlsx')

# 假设第一列包含epoch信息
# 找出所有不同的实验（假设每个实验的epoch是连续的）
experiments = df['epoch'].diff().ne(1).cumsum()

# 使用ExcelWriter保存到不同的sheet
with pd.ExcelWriter('sorted_experiment_results.xlsx', engine='openpyxl') as writer:
    for experiment, data in df.groupby(experiments):
        # 为每个实验创建一个sheet，可以根据需要自定义sheet名称
        sheet_name = f'Experiment_{experiment}'
        data.to_excel(writer, sheet_name=sheet_name, index=False)

print('实验结果已经被分割到不同的工作表中。')