# 07_error_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from sklearn.model_selection import train_test_split
plt.rcParams["font.family"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# 讀取原始資料
df = pd.read_csv(r'D:\Python\我的AI作品集\專案1_數學函數辨識\data\function_dataset.csv')

# 定義模型特徵
X = df.drop(columns=['sample_id', 'label', 'noise_sigma'])
y = df['label']
sample_id = df['sample_id']     # sample_id 不拿去訓練模型

# 分割測試資料
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, sample_id, test_size=0.2, random_state=10, stratify=y)

# 載入最佳模型
best_model = load(r'D:\Python\我的AI作品集\專案1_數學函數辨識\models\best_model.joblib')

# 開始預測
y_pred = best_model.predict(X_test)

# 找出 Sine/Cosine 之間的錯誤點
cosine_to_sine = ((y_test == 'cosine') & (y_pred == 'sine'))
cosine_to_sine_index = X_test.index[cosine_to_sine]
sine_to_cosine = ((y_test == 'sine') & (y_pred == 'cosine'))
sine_to_cosine_index = X_test.index[sine_to_cosine]
print(f"Cosine 被預測成 Sine 的錯誤: 有 {cosine_to_sine.sum()} 個")
print(f"Sine 被預測成 Cosine 的錯誤: 有 {sine_to_cosine.sum()} 個\n")

# 建立錯誤樣本集
error_df = pd.DataFrame({'sample_id': id_test.values, 
                         'label': y_test.values, 
                         'predicted_label': y_pred})
error_df = error_df[((error_df['label']=='cosine') &
                     (error_df['predicted_label']=='sine')) | 
                    ((error_df['label']=='sine') & 
                     (error_df['predicted_label']=='cosine'))]

# 讀取metadata檔案
metadata = pd.read_csv(r'D:\Python\我的AI作品集\專案1_數學函數辨識\data\function_metadata.csv')
metadata = metadata[['sample_id', 'param_1', 'param_2', 'param_3', 'param_4', 'noise_sigma']]

# 透過樣本編號對應到 metadata
error_metadata = error_df.merge(metadata, on='sample_id', how='left', validate='one_to_one')
print("錯誤樣本:")
print(error_metadata[['sample_id', 'label', 'predicted_label', 'param_1', 
                      'param_2', 'param_3', 'param_4', 'noise_sigma']].head(10).to_string(index=False))
print('-'*120)

# 分析兩種錯誤
cosine_sine_metadata = error_metadata[(error_metadata['label']=='cosine') & 
                                      (error_metadata['predicted_label']=='sine')]
sine_cosine_metadata = error_metadata[(error_metadata['label']=='sine') & 
                                      (error_metadata['predicted_label']=='cosine')]

# 顯示 Noise 統計
print(f"Cosine → Sine 的 noise_sigma:\n{cosine_sine_metadata['noise_sigma'].describe()}\n")
print(f"Sine → Cosine 的 noise_sigma:\n{sine_cosine_metadata['noise_sigma'].describe()}\n")

# 建立測試資料表
test_df = pd.DataFrame({'sample_id': id_test.values, 
                        'label': y_test.values, 
                        'predicted_label': y_pred})
test_df = test_df[test_df['label'].isin(['sine', 'cosine'])]
test_metadata = metadata.copy()     # 加入 metadata
test_df = test_df.merge(test_metadata, on='sample_id', how='left', validate='one_to_one')

# 將 noise_sigma 分成4個區間
bins = [0, 0.05, 0.10, 0.15, 0.20]        
labels = ['0.00 ~ 0.05', '0.05 ~ 0.10', '0.10 ~ 0.15', '0.15 ~ 0.20']
test_df['noise_range'] = pd.cut(test_df['noise_sigma'], bins=bins, labels=labels, include_lowest=True)
test_df['error'] = (test_df['label'] != test_df['predicted_label'])     # 判斷是否預測錯誤


# 計算每個 Noise 區間的錯誤率
noise_analysis = test_df.groupby('noise_range', observed=False).agg(total_samples=('error', 'count'),
                                                                    error_count=('error', 'sum'),
                                                                    error_rate=('error', 'mean'))
print(f"Noise Sigma 錯誤率分析:\n{noise_analysis.to_string(formatters={'error_rate': '{:.2%}'.format})}")

# 繪製圖表
xx = np.linspace(-5, 5, 100)            # Sample Points
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(1, 2, 1)
for index in cosine_to_sine_index:
    y_values = X_test.loc[index].values
    ax.plot(xx, y_values, alpha=0.5)
ax.set_xlabel("樣本點")
ax.set_ylabel("對應數值")
ax.set_title("Cosine 被預測成 Sine 的圖形類別")
ax.grid()

ax = fig.add_subplot(1, 2, 2)
for index in sine_to_cosine_index:
    y_values = X_test.loc[index].values
    ax.plot(xx, y_values, alpha=0.5)
ax.set_xlabel("樣本點")
ax.set_ylabel("對應數值")
ax.set_title("Sine 被預測成 Cosine 的圖形類別")
ax.grid()

plt.tight_layout()
plt.show()