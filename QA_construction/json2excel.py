import pandas as pd
import json

# 读取 JSON 文件
with open('dataProcessed/wit_Complete/wit_init_1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 将 JSON 数据转换为 DataFrame
# 将 JSON 数据转化为 DataFrame
rows = []

for key, value in data.items():
    row = {'ID': key}
    row.update(value)
    rows.append(row)

df = pd.DataFrame(rows)
# 保存为 Excel 文件
df.to_excel('dataProcessed/wit_Complete/wit_init_1.xlsx', index=False)