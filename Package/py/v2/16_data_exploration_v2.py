# 16_data_exploration_v2.py
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

df_dataset = pd.read_csv(f'{PROJECT_DIR}/data/v2/function_dataset_v2.csv')

# 查看基本資料資訊
print(f"外型: {df_dataset.shape}\n")
print(df_dataset.info())
print(f"前5筆資料:\n{df_dataset.head()}\n")
print(f"欄位:\n{df_dataset.columns.tolist()}\n")
print(f"資料型態:\n{df_dataset.dtypes}\n")
print(f"缺失值:\n{df_dataset.isnull().sum()}\n")
print(f"各函數類型數量:\n{df_dataset['label'].value_counts()}\n")
print(f"各函數類型比例:\n{df_dataset['label'].value_counts(normalize=True)}\n")
print('-'*120)

df_metadata = pd.read_csv(f'{PROJECT_DIR}/data/v2/function_metadata_v2.csv')

# 查看基本資料資訊
print(f"外型: {df_metadata.shape}\n")
print(df_metadata.info())
print(f"前5筆資料:\n{df_metadata.head()}\n")
print(f"欄位:\n{df_metadata.columns.tolist()}\n")
print(f"資料型態:\n{df_metadata.dtypes}\n")
print(f"缺失值:\n{df_metadata.isnull().sum()}\n")
print(f"Noise 統計:\n{df_metadata['noise_sigma'].describe()}")