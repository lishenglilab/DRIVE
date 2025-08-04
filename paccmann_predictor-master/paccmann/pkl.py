import pandas as pd

# 读取 pkl 文件
df = pd.read_pickle('smiles_language.pkl')

# 将 DataFrame 保存为 csv 文件
print(type(df))
print(dir(df))