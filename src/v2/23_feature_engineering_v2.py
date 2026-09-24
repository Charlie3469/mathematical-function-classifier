# 23_feature_engineering_v2.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/v2/function_dataset_cleaned_v2.csv')
X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)

# ----進行特徵工程----
# 將原始資料標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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

print("----以下是特徵工程的結果----")
print(f"原始最大值:\n{X_train.max(axis=0)}\n")              # 最大值
print(f"標準化後的最大值:\n{X_train_scaled.max(axis=0)}\n")
print(f"原始最小值:\n{X_train.min(axis=0)}\n")              # 最小值
print(f"標準化後的最小值:\n{X_train_scaled.min(axis=0)}\n")
print(f"原始平均值:\n{X_train.mean(axis=0)}\n")             # 原始平均值
print(f"原始標準差:\n{X_train.std(axis=0)}\n")              # 原始標準差

print(f"迴歸斜率:(取前5筆資料)\n{train_slopes[:5]}\n")      # 前5筆的迴歸斜率

print("經過一階差分處理後的結果:(取前5筆資料)")
train_first_diff_mean = train_first_diff.mean(axis=1)
train_first_diff_std = train_first_diff.std(axis=1)
print(f"平均值: {train_first_diff_mean[:5]}")
print(f"標準差: {train_first_diff_std[:5]}\n")

print("經過二階差分處理後的結果:(取前5筆資料)")
train_second_diff_mean = train_second_diff.mean(axis=1)
train_second_diff_std = train_second_diff.std(axis=1)
print(f"平均值: {train_second_diff_mean[:5]}")
print(f"標準差: {train_second_diff_std[:5]}\n")

print(f"變化幅度:(取前5筆資料)\n{train_amplitude[:5]}\n")

# 建立 X_train 的6個數學特徵(斜率, 總變化量, 一階差分, 二階差分)
train_math_features = pd.DataFrame({'slope': train_slopes,
                                    'amplitude': train_amplitude,
                                    'first_diff_mean': train_first_diff_mean,
                                    'first_diff_std': train_first_diff_std,
                                    'second_diff_mean': train_second_diff_mean,
                                    'second_diff_std': train_second_diff_std})
print(f"X_train數學特徵前5筆:\n{train_math_features.head()}\n")

# 加入原始特徵
X_train_math = pd.concat([X_train.reset_index(drop=True), 
                          train_math_features.reset_index(drop=True)], axis=1)
print(f"X_train_math 形狀為: {X_train_math.shape}")
print('-'*100)


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

# 一階差分統計
test_first_diff_mean = test_first_diff.mean(axis=1)
test_first_diff_std = test_first_diff.std(axis=1)
# 二階差分統計
test_second_diff_mean = test_second_diff.mean(axis=1)
test_second_diff_std = test_second_diff.std(axis=1)

# 建立 X_test 的6個數學特徵(斜率, 總變化量, 一階差分, 二階差分)
test_math_features = pd.DataFrame({'slope': test_slopes,
                                   'amplitude': test_amplitude,
                                   'first_diff_mean': test_first_diff_mean,
                                   'first_diff_std': test_first_diff_std,
                                   'second_diff_mean': test_second_diff_mean,
                                   'second_diff_std': test_second_diff_std})
print(f"X_test數學特徵前5筆:\n{test_math_features.head()}\n")

# 加入原始特徵
X_test_math = pd.concat([X_test.reset_index(drop=True),
                         test_math_features.reset_index(drop=True)], axis=1)
print(f"X_test_math 形狀為: {X_test_math.shape}")
print('='*100)

# 把訓練集跟測試集結合成一個大表格
X_math_merged = pd.concat([X_train_math.reset_index(drop=True), 
                           X_test_math.reset_index(drop=True)], axis=0)
print(f"把訓練數據跟測試數據合併之後, 形狀為: {X_math_merged.shape}\n")

# 將合併後的表格儲存成新的檔案
y_merged = pd.concat([y_train.reset_index(drop=True), 
                      y_test.reset_index(drop=True)], axis=0)
math_merged = pd.concat([y_merged, X_math_merged], axis=1)
math_merged.to_csv('data/v2/function_dataset_merged_v2.csv', index=False)

# 再進行標準化一次
scaler_after = StandardScaler()
X_train_math_scaled = scaler_after.fit_transform(X_train_math)
X_test_math_scaled = scaler_after.transform(X_test_math)
print("標準化之後:")
print(f"平均值:\n{X_train_math_scaled.mean(axis=0)}")
print(f"標準差:\n{X_train_math_scaled.std(axis=0)}")