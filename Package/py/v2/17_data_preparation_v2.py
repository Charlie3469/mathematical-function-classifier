# 17_data_preparation_v2.py
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

df_dataset = pd.read_csv(f'{PROJECT_DIR}/data/v2/function_dataset_v2.csv')

# 刪除不必要的數據
df1 = df_dataset.drop(columns=['id'])
df1 = df1.drop(columns=['noise'])
df1.to_csv(f'{PROJECT_DIR}/data/v2/function_dataset_cleaned_v2.csv', index=False)
print(f"原始 V2 資料清理完成, 形狀為: {df1.shape}")

df_metadata = pd.read_csv(f'{PROJECT_DIR}/data/v2/function_metadata_v2.csv')

# 進行數據清理
df_metadata.fillna(0, inplace=True)
df2 = df_metadata
df2.to_csv(f'{PROJECT_DIR}/data/v2/function_metadata_v2.csv', index=False)
print(f"原始 V2 metadata 資料清理完成, 形狀為: {df2.shape}")
