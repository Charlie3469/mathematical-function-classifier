# 10_feature_experiment.py  
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import load
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r'D:\Python\我的AI作品集\專案1_數學函數辨識\data\v1\function_dataset_cleaned.csv')
X = df.drop(columns=['label'])
y = df['label']

# 分割測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)

# 先將原本10個欄位拿去訓練模型
best_model = load(r'D:\Python\我的AI作品集\專案1_數學函數辨識\models\v1\best_model.joblib')
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("原始特徵:")
print(f"準確度: {accuracy:.3f}")
print(f"混淆矩陣:\n{confusion_matrix(y_test, y_pred)}\n")
print(f"分類報告:\n{classification_report(y_test, y_pred)}\n")

# ----建立數學特徵----
# 處理訓練集的部分
train_first_diff = np.diff(X_train.values, axis=1)          # 一階差分
train_second_diff = np.diff(X_train.values, n=2, axis=1)    # 二階差分
train_amplitude = X_train.max(axis=1) - X_train.min(axis=1) # 變化總幅度

train_x = np.linspace(-2, 2, X_train.shape[1]).reshape(-1, 1)
train_slopes = []                                           # 每一筆函數的整體迴歸斜率
for row in X_train.values:
    train_model = LinearRegression()
    train_model.fit(train_x, row)
    train_slopes.append(train_model.coef_[0])
train_slopes = np.array(train_slopes)  

train_first_diff_mean = train_first_diff.mean(axis=1)
train_first_diff_std = train_first_diff.std(axis=1)

train_second_diff_mean = train_second_diff.mean(axis=1)
train_second_diff_std = train_second_diff.std(axis=1)

train_math_features = pd.DataFrame({'slope': train_slopes,
                                    'amplitude': train_amplitude,
                                    'first_diff_mean': train_first_diff_mean,
                                    'first_diff_std': train_first_diff_std,
                                    'second_diff_mean': train_second_diff_mean,
                                    'second_diff_std': train_second_diff_std})
X_train_math = pd.concat([X_train.reset_index(drop=True), 
                          train_math_features.reset_index(drop=True)], axis=1)

# 處理測試集的部分
test_first_diff = np.diff(X_test.values, axis=1)            # 一階差分
test_second_diff = np.diff(X_test.values, n=2, axis=1)      # 二階差分
test_amplitude = X_test.max(axis=1) - X_test.min(axis=1)    # 變化總幅度

test_x = np.linspace(-2, 2, X_test.shape[1]).reshape(-1, 1)
test_slopes = []                                            # 每一筆函數的整體迴歸斜率
for row in X_test.values:
    test_model = LinearRegression()
    test_model.fit(test_x, row)
    test_slopes.append(test_model.coef_[0])
test_slopes = np.array(test_slopes)

test_first_diff_mean = test_first_diff.mean(axis=1)
test_first_diff_std = test_first_diff.std(axis=1)

test_second_diff_mean = test_second_diff.mean(axis=1)
test_second_diff_std = test_second_diff.std(axis=1)

test_math_features = pd.DataFrame({'slope': test_slopes,
                                   'amplitude': test_amplitude,
                                   'first_diff_mean': test_first_diff_mean,
                                   'first_diff_std': test_first_diff_std,
                                   'second_diff_mean': test_second_diff_mean,
                                   'second_diff_std': test_second_diff_std})
X_test_math = pd.concat([X_test.reset_index(drop=True),
                         test_math_features.reset_index(drop=True)], axis=1)


# 將結合後的數據再訓練模型一次
best_model_math = load(r'D:\Python\我的AI作品集\專案1_數學函數辨識\models\v1\best_model.joblib')
best_model_math.fit(X_train_math, y_train.reset_index(drop=True))
y_pred_math = best_model_math.predict(X_test_math)
math_accuracy = accuracy_score(y_test.reset_index(drop=True), y_pred_math)
print("新增一些數學特徵之後:")
print(f"準確度: {math_accuracy:.3f}")
print(f"混淆矩陣:\n{confusion_matrix(y_test.reset_index(drop=True), y_pred_math)}\n")
print(f"分類報告:\n{classification_report(y_test.reset_index(drop=True), y_pred_math)}")
print("-"*100)

# 比較結果
improvement = math_accuracy - accuracy
print("----實驗結果----")
print(f"原始特徵準確度: {accuracy:.3f}")
print(f"數學特徵準確度: {math_accuracy:.3f}")
print(f"比原本特徵進步了: {improvement:+.3f}")
print("結論:", end=" ")
if improvement > 0:
    print("加入數學特徵後, 準確度提升。")
elif improvement < 0:
    print("加入數學特徵後, 準確度下降。")
else:
    print("加入數學特徵後, 準確度沒有改變。")