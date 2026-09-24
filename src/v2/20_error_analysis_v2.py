# 20_error_analysis_v2.py
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split

# 讀取原始資料
df = pd.read_csv('data/v2/function_dataset_v2.csv')

# 定義模型特徵
X = df.drop(columns=['sample_id', 'label', 'noise_sigma'])
y = df['label']
sample_id = df['sample_id']     # sample_id 不拿去訓練模型

# 分割測試資料
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, sample_id, test_size=0.2, random_state=10, stratify=y)

# 載入最佳模型
best_model = load('models/v2/best_model_v2.joblib')

# 開始預測
y_pred = best_model.predict(X_test)

# 建立測試資料表
test_all_df = pd.DataFrame({'sample_id': id_test.values,
                            'label': y_test.values,
                            'predicted_label': y_pred})
test_all_df['error'] = (test_all_df['label'] != test_all_df['predicted_label'])
print("----預測錯誤分析----")
print(f"測試樣本數: {len(test_all_df)}")
print(f"預測錯誤數量: {test_all_df['error'].sum()}")
print(f"整體錯誤率: {test_all_df['error'].mean():.2%}\n")

# 讀取metadata檔案
metadata = pd.read_csv('data/v2/function_metadata_v2.csv')
metadata = metadata[['sample_id', 'param_1', 'param_2', 'param_3', 'param_4', 'noise_sigma']]

# 加入 metadata
test_metadata = metadata.copy()

# 透過樣本編號對應到 metadata
error_metadata = test_all_df.merge(metadata, on='sample_id', how='left', validate='one_to_one')

# 找出 Sine/Cosine 之間的錯誤點
test_df = test_all_df[test_all_df['label'].isin(['sine', 'cosine'])].copy()
test_df = test_df.merge(test_metadata, on='sample_id', how='left', validate='one_to_one')
print(f"Sine被誤判成Cosine: {((test_df['label']=='sine') & (test_df['predicted_label']=='cosine')).sum()}")
sine_to_cosine = ((y_test == 'sine') & (y_pred == 'cosine'))
sine_cosine_metadata = error_metadata[(error_metadata['label']=='sine') & 
                                      (error_metadata['predicted_label']=='cosine')]
print("錯誤樣本:")
print(sine_cosine_metadata[['sample_id', 'label', 'predicted_label', 'param_1',
                            'param_2', 'param_3', 'param_4', 'noise_sigma']].to_string(index=False))
print(f"\nCosine被誤判成Sine: {((test_df['label']=='cosine') & (test_df['predicted_label']=='sine')).sum()}")
cosine_to_sine = ((y_test == 'cosine') & (y_pred == 'sine'))
cosine_sine_metadata = error_metadata[(error_metadata['label']=='cosine') & 
                                      (error_metadata['predicted_label']=='sine')]
print("錯誤樣本:")
print(cosine_sine_metadata[['sample_id', 'label', 'predicted_label', 'param_1',
                            'param_2', 'param_3', 'param_4', 'noise_sigma']].to_string(index=False))

# 處理其他錯誤的部分
others = test_all_df['error'].sum() - \
    ((test_df['label']=='sine') & (test_df['predicted_label']=='cosine')).sum() - \
    ((test_df['label']=='cosine') & (test_df['predicted_label']=='sine')).sum()
print(f"\n其他錯誤: {others}")
print("Sine / Cosine 的其他錯誤樣本:")
other_errors = test_df[(test_df['error']) &
    ~(((test_df['label'] == 'sine') & (test_df['predicted_label'] == 'cosine')) |
      ((test_df['label'] == 'cosine') & (test_df['predicted_label'] == 'sine')))]
print(other_errors[['sample_id', 'label', 'predicted_label', 
                    'param_1', 'param_2', 'param_3', 'param_4', 'noise_sigma']].to_string(index=False))
