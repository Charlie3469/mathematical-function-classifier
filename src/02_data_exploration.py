# 02_data_exploration.py
import pandas as pd

df_dataset = pd.read_csv(r'D:\Python\我的AI作品集\專案1_數學函數辨識\data\function_dataset.csv')
# 查看基本資料資訊
print(f"數據外型: {df_dataset.shape}")
print(f"前5筆資料:\n{df_dataset.head()}\n")
print(f"欄位:\n{df_dataset.columns.tolist()}")
print(f"資料型態:\n{df_dataset.dtypes}")
print(f"缺失值:\n{df_dataset.isnull().sum()}\n")

print(f"各函數類型數量:\n{df_dataset['label'].value_counts()}\n")       # 查看各類別數量
print(f"各函數類型比例:\n{df_dataset['label'].value_counts(normalize=True)}\n") # 查看比例
print(f"Noise 統計:\n{df_dataset['noise_sigma'].describe()}")           # 查看noise分布
print('='*120)


df_metadata = pd.read_csv(r'D:\Python\我的AI作品集\專案1_數學函數辨識\data\function_metadata.csv')
# 查看基本資料資訊
print(f"數據外型: {df_metadata.shape}")
print(f"前5筆資料:\n{df_metadata.head()}\n")
print(f"欄位:\n{df_metadata.columns.tolist()}")
print(f"資料型態:\n{df_metadata.dtypes}")
print(f"缺失值:\n{df_metadata.isnull().sum()}\n")
print(df_metadata.info())