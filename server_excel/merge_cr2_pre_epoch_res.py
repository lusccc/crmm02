import pandas as pd
import re

# 定义一个函数，从文件名中提取pretrain epoch
def extract_epoch(filename):
    match = re.search(r'pre_(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return None

# 所有Excel文件的列表
files = [
    "cr2_res_pre_100.xlsx", "cr2_res_pre_10.xlsx", "cr2_res_pre_120.xlsx",
    "cr2_res_pre_150.xlsx", "cr2_res_pre_170.xlsx", "cr2_res_pre_200.xlsx",
    "cr2_res_pre_20.xlsx", "cr2_res_pre_220.xlsx", "cr2_res_pre_250.xlsx",
    "cr2_res_pre_300.xlsx", "cr2_res_pre_50.xlsx", "cr2_res_pre_60.xlsx",
] +[f'cr2_res_pre_{i+1}.xlsx' for i in range(9)]

# 使用pandas的concat方法合并所有Excel文件
combined_df = pd.concat([
    pd.read_excel(file).assign(**{"pretrain epoch": extract_epoch(file)})
    for file in files
])

# 保存合并后的Excel文件
combined_df.to_excel("merge_cr2_pre_epoch_res.xlsx", index=False)