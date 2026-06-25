import pandas as pd
import os

# 1. 读取原始 Excel 表格
input_file = "../scenario23_dev/scenario23.csv"
root_dir = "../scenario23_dev"


df = pd.read_csv(input_file)

# 2. 我们关心的列
columns = ['unit2_loc', 'unit2_height','unit1_pwr_60ghz']
value_column = ['unit1_beam_index','seq_index']

# 3. 用来存放结果的 DataFrame
result = pd.DataFrame()

# 4. 定义一个函数读取路径对应的数值
def read_txt(path):
    """读取 txt 文件中的数值"""
    full_path = os.path.join(root_dir, path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"{full_path} not found")
    arr = []
    with open(full_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            arr.append([float(x) for x in parts])
    return arr

# 5. 遍历每个列，对路径进行读取
for col in columns:
    if col in df.columns:
        print(f"正在读取列 {col} ...")
        result[col] = df[col].apply(read_txt)
    else:
        print(f"Excel中未找到列 {col}")

for col in value_column:
    if col in df.columns:
        result[col] = df[col]
    else:
        print(f"未找到列 {value_column}")

# 6. 保存结果到新的 Excel
output_file = "motion_data.xlsx"
result.to_excel(output_file, index=False)

print(f"数据已成功保存到: {output_file}")
