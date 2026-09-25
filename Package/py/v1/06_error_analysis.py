# 06_error_analysis.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

# 讀取原始資料
df = pd.read_csv(f'{DATA_DIR}/v1/function_dataset.csv')

# 定義模型特徵
X = df.drop(columns=['id', 'label', 'noise'])
y = df['label']
sample_id = df['id']        # sample_id 不拿去訓練模型

# 分割測試資料
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, sample_id, test_size=0.2, random_state=10, stratify=y)

# 載入最佳模型
best_model = joblib.load(f'{MODEL_DIR}/v1/best_model.joblib')

# 開始預測
y_pred = best_model.predict(X_test)

# 建立測試資料表
test_all_df = pd.DataFrame({'id': id_test.values,
                            'label': y_test.values,
                            'predicted_label': y_pred})
test_all_df['error'] = (test_all_df['label'] != test_all_df['predicted_label'])

# 讀取metadata檔案
metadata = pd.read_csv(f'{DATA_DIR}/v1/function_metadata.csv')
metadata = metadata[['id', 'param_1', 'param_2', 'param_3', 'param_4', 'noise_sigma']]

# 加入 metadata
test_metadata = metadata.copy()

# 透過樣本編號對應到 metadata
error_metadata = test_all_df.merge(metadata, on='id', how='left', validate='one_to_one')

# 篩選所有預測錯誤的資料
all_errors = test_all_df[test_all_df['error']].copy()
all_errors = all_errors.merge(test_metadata, on='id', how='left', validate='one_to_one')
print("----預測錯誤分析----")
print(f"測試樣本數: {len(test_all_df)}")
print(f"預測錯誤數量: {len(all_errors)}")
print(f"整體錯誤率: {test_all_df['error'].mean():.2%}\n")

error_summary = (all_errors.groupby(['label', 'predicted_label']).size()
                           .reset_index(name='error_count')
                           .sort_values('error_count', ascending=False))
print(f"錯誤類別統計:\n{error_summary.to_string(index=False)}")
