# 12_model_tuning.py
import pandas as pd
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *

data = pd.read_csv(r'D:\Python\我的AI作品集\專案1_數學函數辨識\data\function_dataset_merged.csv')
X = data.drop(columns=['label'])
y = data['label']

# 分割數據
train_rows = int(len(data)*0.8)
X_train = X.iloc[:train_rows].copy()
X_test = X.iloc[train_rows:].copy()
y_train = y.iloc[:train_rows].copy()
y_test = y.iloc[train_rows:].copy()

model = load(r'D:\Python\我的AI作品集\專案1_數學函數辨識\models\best_model.joblib')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"準確度: {accuracy_score(y_test, y_pred):.4f}")

# 設定要調整的參數
param_grid = {'n_estimators': [100, 150, 200],
              'max_depth': [None, 10, 20],
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1]}

# 建立 GridSearchCV
rf = RandomForestClassifier(random_state=10, n_jobs=-1)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, 
                           scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print(f"最佳參數:\n{grid_search.best_params_}")
print(f"最佳CV準確度: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)
print(f"訓練數據準確度: {accuracy_score(y_train, train_pred):.4f}")
print(f"測試數據準確度: {accuracy_score(y_test, test_pred):.4f}")
improvement = accuracy_score(y_test, test_pred) - accuracy_score(y_test, y_pred)
print(f"原始 Random Forest 模型: {accuracy_score(y_test, y_pred):.4f}") 
print(f"調整後 Random Forest 模型: {accuracy_score(y_test, test_pred):.4f}") 
print(f"測試集改善幅度: {improvement:+.4f}") 

print("結論:")
if improvement > 0: 
    print("Model Tuning 後，測試準確度提升。") 
elif improvement < 0: 
    print("Model Tuning 後，測試準確度下降。") 
else: 
    print("Model Tuning 前後，測試準確度沒有改變。")

print(f"混淆矩陣:\n{confusion_matrix(y_test, test_pred)}")
print(f"分類報告:\n{classification_report(y_test, test_pred)}\n")
