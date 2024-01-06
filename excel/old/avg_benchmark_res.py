import pandas as pd
import glob

year_window = '#2010,2011,2012#_#2013#'
# 获取所有Excel文件
files = glob.glob(f"cr_res_benchmark_rolling_window_{year_window}_*.xlsx")

# 创建空的DataFrame来存储所有数据
all_data = pd.DataFrame()

# 遍历所有文件
for file in files:
    # 读取Excel文件
    df = pd.read_excel(file)

    # 添加到all_data
    # all_data = all_data.append(df, ignore_index=True)
    all_data = pd.concat([all_data, df], ignore_index=True)

# 通过模型名称对数据进行分组，然后计算平均值
average_data = all_data.groupby('Model').mean()

# 将结果保存到新的Excel文件中
average_data.to_excel(f'average_benchmark_rolling_{year_window}_results.xlsx')
