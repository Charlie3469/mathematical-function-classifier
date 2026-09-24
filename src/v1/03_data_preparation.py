# 03_data_preparation.py
import pandas as pd

df_dataset = pd.read_csv('data/v1/function_dataset.csv')

# 刪除不必要的數據
df1 = df_dataset.drop(columns=['sample_id'])
df1 = df1.drop(columns=['noise_sigma'])
df1.to_csv('data/v1/function_dataset_cleaned.csv', index=False)
print(f"清理後dataset資料形狀: {df1.shape}")

df_metadata = pd.read_csv('data/v1/function_metadata.csv')

# 進行數據清理
df_metadata.fillna(0, inplace=True)
df2 = df_metadata
df2.to_csv('data/v1/function_metadata.csv', index=False)
print(f"清理後metadata資料形狀: {df2.shape}")
