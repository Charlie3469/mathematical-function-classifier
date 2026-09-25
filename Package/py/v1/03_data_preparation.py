# 03_data_preparation.py
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

df_dataset = pd.read_csv(f'{PROJECT_DIR}/data/v1/function_dataset.csv')

# 刪除不必要的數據
df1 = df_dataset.drop(columns=['id'])
df1 = df1.drop(columns=['noise'])
df1.to_csv(f'{PROJECT_DIR}/data/v1/function_dataset_cleaned.csv', index=False)
print(f"原始資料清理完成, 形狀為: {df1.shape}")

df_metadata = pd.read_csv(f'{PROJECT_DIR}/data/v1/function_metadata.csv')

# 進行數據清理
df_metadata.fillna(0, inplace=True)
df2 = df_metadata
df2.to_csv(f'{PROJECT_DIR}/data/v1/function_metadata.csv', index=False)
print(f"原始 metadata 資料清理完成, 形狀為: {df2.shape}")
