# 07_parameter_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.model_selection import train_test_split
plt.rcParams["font.family"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv('data/v1/function_dataset.csv')
X = df.drop(columns=['sample_id', 'label', 'noise_sigma'])
y = df['label']
sample_id = df['sample_id']

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, sample_id, test_size=0.2, random_state=10, stratify=y)

best_model = load('models/v1/best_model.joblib')
y_pred = best_model.predict(X_test)

# 建立測試資料表
test_df = pd.DataFrame({'sample_id': id_test.values, 'label': y_test.values, 'predicted_label': y_pred})
test_df['error'] = (test_df['label'] != test_df['predicted_label'])

metadata = pd.read_csv('data/v1/function_metadata.csv')
metadata = metadata[['sample_id', 'param_1', 'param_2', 'param_3', 'param_4', 'noise_sigma']]
test_df = test_df.merge(metadata, on='sample_id', how='left', validate='one_to_one')

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

print("正確樣本 vs 錯誤樣本參數比較")
print(para_sum_df.to_string(index=False, formatters={'Correct Mean': '{:.3f}'.format,
                                                     'Correct Median': '{:.3f}'.format,
                                                     'Error Mean': '{:.3f}'.format,
                                                     'Error Median': '{:.3f}'.format,
                                                     'Mean Difference': '{:.3f}'.format}))
print("-"*100)

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

# 儲存參數統計比較
para_sum_df.to_html('data/v1/parameter_analysis.html')

# 繪製圖表(正確/錯誤樣本的參數平均值)
plot_data = para_sum_df[['Parameter', 'Correct Mean', 'Error Mean']]
plot_data = plot_data.set_index('Parameter')
plot_data.plot(kind='bar', figsize=(10, 6))
plt.xlabel('參數')
plt.ylabel('平均值')
plt.title('所有的錯誤樣本參數平均值')
plt.xticks()
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig('data/v1/error_analysis.png')
plt.show()