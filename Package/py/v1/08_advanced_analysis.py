# 08_advanced_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

plt.rcParams["font.family"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(f'{DATA_DIR}/v1/function_dataset.csv')
X = df.drop(columns=['id', 'label', 'noise'])
y = df['label']
sample_id = df['id']

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, sample_id, test_size=0.2, random_state=10, stratify=y)

best_model = joblib.load(f'{MODEL_DIR}/v1/best_model.joblib')
y_pred = best_model.predict(X_test)

test_df = pd.DataFrame({'id': id_test.values, 
                        'label': y_test.values, 
                        'predicted_label': y_pred})
test_df = test_df[test_df['label'].isin(['sine', 'cosine'])].copy()
test_df['error'] = (test_df['label'] != test_df['predicted_label'])

# 讀取metadata檔案
metadata = pd.read_csv(f'{DATA_DIR}/v1/function_metadata.csv')
metadata = metadata[['id', 'param_1', 'param_2', 'param_3', 'param_4', 'noise_sigma']]
test_metadata = metadata.copy()

# 透過樣本編號對應到 metadata
error_metadata = test_df.merge(metadata, on='id', how='left', validate='one_to_one')
test_df = test_df.merge(test_metadata, on='id', how='left', validate='one_to_one')

# 找出 Sine/Cosine 之間的錯誤點
print("----以下是 Sine / Cosine 錯誤分析的結果----")
print(f"樣本數: {len(test_df)}")
print(f"預測錯誤: {test_df['error'].sum()}")
cosine_to_sine = ((y_test == 'cosine') & (y_pred == 'sine'))
cosine_to_sine_index = X_test.index[cosine_to_sine]
sine_to_cosine = ((y_test == 'sine') & (y_pred == 'cosine'))
sine_to_cosine_index = X_test.index[sine_to_cosine]
cosine_sine_metadata = error_metadata[(error_metadata['label']=='cosine') & 
                                      (error_metadata['predicted_label']=='sine')]
sine_cosine_metadata = error_metadata[(error_metadata['label']=='sine') & 
                                      (error_metadata['predicted_label']=='cosine')]
print(f"Sine被誤判成Cosine: {((test_df['label']=='sine') & (test_df['predicted_label']=='cosine')).sum()}")
print(sine_cosine_metadata[['id', 'label', 'predicted_label', 'param_1',
                            'param_2', 'param_3', 'param_4', 'noise_sigma']].to_string(index=False))
print(f"\nCosine被誤判成Sine: {((test_df['label']=='cosine') & (test_df['predicted_label']=='sine')).sum()}")
print(cosine_sine_metadata[['id', 'label', 'predicted_label', 'param_1',
                            'param_2', 'param_3', 'param_4', 'noise_sigma']].to_string(index=False))

# 其他錯誤
others = test_df['error'].sum() - \
    ((test_df['label']=='sine') & (test_df['predicted_label']=='cosine')).sum() - \
    ((test_df['label']=='cosine') & (test_df['predicted_label']=='sine')).sum()
print(f"\n其他錯誤: {others}")
other_errors = test_df[(test_df['error']) &
    ~(((test_df['label'] == 'sine') & (test_df['predicted_label'] == 'cosine')) |
      ((test_df['label'] == 'cosine') & (test_df['predicted_label'] == 'sine')))]
print(other_errors[['id', 'label', 'predicted_label', 'param_1', 
                    'param_2', 'param_3', 'param_4', 'noise_sigma']].to_string(index=False))

# 顯示 Noise 統計
print("Noise sigma:")
print(f"Sine → Cosine:\n{sine_cosine_metadata['noise_sigma'].describe()}\n")
print(f"Cosine → Sine:\n{cosine_sine_metadata['noise_sigma'].describe()}\n")
bins = [0, 0.02, 0.04, 0.06, 0.08, 0.10, np.inf]      # 將 noise_sigma 分成6個區間
labels = ['0.00 ~ 0.02',
          '0.02 ~ 0.04',
          '0.04 ~ 0.06',
          '0.06 ~ 0.08',
          '0.08 ~ 0.10',
          '0.10+']
test_df['noise_range'] = pd.cut(test_df['noise_sigma'], bins=bins, labels=labels, include_lowest=True)
test_df['error'] = (test_df['label'] != test_df['predicted_label'])     # 判斷是否預測錯誤

# 計算每個 Noise 區間的錯誤率
noise_analysis = test_df.groupby('noise_range', observed=False).agg(total_samples=('error', 'count'),
                                                                    error_count=('error', 'sum'),
                                                                    error_rate=('error', 'mean'))
print(f"錯誤率分析:\n{noise_analysis.to_string(formatters={'error_rate': '{:.2%}'.format})}")
print('='*100)

# 繪製圖表
xx = np.linspace(-5, 5, 100)            # Sample Points
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(2, 1, 1)
for index in cosine_to_sine_index:
    y_values = X_test.loc[index].values
    ax.plot(xx, y_values, alpha=0.5)
ax.set_xlabel("樣本點")
ax.set_ylabel("對應數值")
ax.set_title("Cosine 被預測成 Sine 的圖形類別")
ax.grid()

ax = fig.add_subplot(2, 1, 2)
for index in sine_to_cosine_index:
    y_values = X_test.loc[index].values
    ax.plot(xx, y_values, alpha=0.5)
ax.set_xlabel("樣本點")
ax.set_ylabel("對應數值")
ax.set_title("Sine 被預測成 Cosine 的圖形類別")
ax.grid()

plt.tight_layout()
plt.legend()
plt.savefig(f'{DATA_DIR}/v1/advanced_error_analysis.png')
plt.show()


# 建立測試資料表
test_df = pd.DataFrame({'id': id_test.values, 'label': y_test.values, 'predicted_label': y_pred})
test_df = test_df[test_df['label'].isin(['sine', 'cosine'])].copy()
test_df['error'] = (test_df['label'] != test_df['predicted_label'])
print("參數分析:")
print(f"樣本數: {len(test_df)}")
print(f"預測錯誤: {test_df['error'].sum()}")

metadata = pd.read_csv(f'{DATA_DIR}/v1/function_metadata.csv')
metadata = metadata[['id', 'param_1', 'param_2', 'param_3', 'param_4', 'noise_sigma']]
test_df = test_df.merge(metadata, on='id', how='left', validate='one_to_one')

# 正確 vs 錯誤的參數統計
parameters = ['param_1', 'param_2', 'param_3', 'param_4']
parameter_summary = []
for parameter in parameters:
    correct_data = test_df[test_df['error'] == False][parameter]
    error_data = test_df[test_df['error'] == True][parameter]
    parameter_summary.append({'Parameter': parameter,
                              'Correct Mean': correct_data.mean(),
                              'Correct Median': correct_data.median(),
                              'Error Mean': error_data.mean(),
                              'Error Median': error_data.median(),
                              'Mean Difference': (error_data.mean() - correct_data.mean())})
para_sum_df = pd.DataFrame(parameter_summary)

print("正確 vs 錯誤樣本參數比較:")
print(para_sum_df.to_string(index=False, formatters={'Correct Mean': '{:.3f}'.format,
                                                     'Correct Median': '{:.3f}'.format,
                                                     'Error Mean': '{:.3f}'.format,
                                                     'Error Median': '{:.3f}'.format,
                                                     'Mean Difference': '{:.3f}'.format}))
print("-"*100)

# 儲存參數統計比較
para_sum_df.to_html(f'{DATA_DIR}/v1/advanced_parameter_results.html')

# 分析每個參數的數值區間
print("各參數區間的錯誤率: ")
for parameter in parameters:
    # 使用四分位數分組, 每個區間大約包含相同數量的資料
    test_df[f'{parameter}_range'] = pd.qcut(test_df[parameter], q=4, duplicates='drop')
    analysis = test_df.groupby(f'{parameter}_range', observed=False).agg(
        total_samples=('error', 'count'),
        error_count=('error', 'sum'),
        error_rate=('error', 'mean')
    )
    print(analysis.to_string(formatters={'error_rate': '{:.2%}'.format}))
    print("\n")

# 繪製圖表(正確/錯誤樣本的參數平均值)
plot_data = para_sum_df[['Parameter', 'Correct Mean', 'Error Mean']]
plot_data = plot_data.set_index('Parameter')
plot_data.plot(kind='bar', figsize=(10, 6))
plt.xlabel('參數')
plt.ylabel('平均值')
plt.title('Sine / Cosine 的錯誤樣本參數平均值')
plt.xticks()
plt.grid()
plt.tight_layout()
plt.show()