# 26_model_tuning_v2.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"


# 讀取原始清理後資料
data = pd.read_csv(f'{DATA_DIR}/v2/function_dataset_cleaned_v2.csv')
X = data.drop(columns=['label'])
y = data['label']

# 分割訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)

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

# 調參前模型
baseline_model = joblib.load(f'{MODEL_DIR}/v2/best_model_v2.joblib')
baseline_model.fit(X_train_math, y_train.reset_index(drop=True))
baseline_pred = baseline_model.predict(X_test_math)
baseline_accuracy = accuracy_score(y_test.reset_index(drop=True), baseline_pred)
print("模型調參前")
print(f"準確度: {baseline_accuracy:.3f}\n")

# 設定 GridSearchCV
rf_model = RandomForestClassifier(random_state=10)
param_grid = {'n_estimators': [100, 150, 200],
              'max_depth': [None, 10, 20],
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1]}

# 開始 GridSearchCV
print("進行 GridSearchCV 調參...")
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_math, y_train.reset_index(drop=True))
print(f"最佳參數: {grid_search.best_params_}")
print(f"最佳CV準確度: {grid_search.best_score_:.4f}\n")

# 使用最佳模型預測
best_model = grid_search.best_estimator_
train_pred = best_model.predict(X_train_math)
test_pred = best_model.predict(X_test_math)
train_accuracy = accuracy_score(y_train.reset_index(drop=True), train_pred)
test_accuracy = accuracy_score(y_test.reset_index(drop=True), test_pred)
print("模型調參後")
print(f"訓練數據準確度: {train_accuracy:.3f}")
print(f"測試數據準確度: {test_accuracy:.3f}")
print('='*100)

# 比較調參前後
print("----實驗結果----")
print(f"調參前準確度: {baseline_accuracy:.4f}")
print(f"調參後準確度: {test_accuracy:.4f}")
improvement = test_accuracy - baseline_accuracy
print(f"改善幅度: {improvement:+.4f}")
print("結論:", end=" ")
if improvement > 0:
    print("模型調參後，測試準確度提升。")
elif improvement < 0:
    print("模型調參後，測試準確度下降。")
else:
    print("測試準確度沒有改變。\n")
print(f"混淆矩陣:\n{confusion_matrix(y_test.reset_index(drop=True), test_pred)}")
print(f"分類報告:\n{classification_report(y_test.reset_index(drop=True), test_pred)}")

# 儲存最佳模型
joblib.dump(best_model, f'{MODEL_DIR}/v2/best_model_tuned_v2.joblib')
print("V2 調參後最佳模型已儲存!")