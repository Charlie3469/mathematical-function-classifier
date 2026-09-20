# 13_ensemble.py
import pandas as pd
import joblib
from joblib import load
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *

# 讀取資料
data = pd.read_csv(r'D:\Python\我的AI作品集\專案1_數學函數辨識\data\function_dataset_merged.csv')
X = data.drop(columns=['label'])
y = data['label']

train_rows = int(len(data)*0.8)
X_train = X.iloc[:train_rows].copy()
X_test = X.iloc[train_rows:].copy()
y_train = y.iloc[:train_rows].copy()
y_test = y.iloc[train_rows:].copy()

best_model = load(r'D:\Python\我的AI作品集\專案1_數學函數辨識\models\best_model.joblib')
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
print(f"一開始測試數據準確度: {accuracy_score(y_test, y_pred):.4f}")

param_grid = {'n_estimators': [100, 150, 200],
              'max_depth': [None, 10, 20],
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1]}

rf = RandomForestClassifier(random_state=10, n_jobs=-1)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, 
                           scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_model_modified = grid_search.best_estimator_
train_pred = best_model_modified.predict(X_train)
test_pred = best_model_modified.predict(X_test)
print(f"調參後測試數據準確度: {accuracy_score(y_test, test_pred):.4f}")

# 定義基礎學習器 & 最終學習器
base_learners = [('DecisionTree', DecisionTreeClassifier(random_state=10)),
                 ('RandomForest', 
                  RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],
                                         random_state=10,
                                         max_depth=grid_search.best_params_['max_depth'],
                                         min_samples_split=grid_search.best_params_['min_samples_split']))]
final_estimator = SVC(random_state=10)

# 利用集成學習的方式訓練以上 base_learners 模型
st_model = StackingClassifier(estimators=base_learners, final_estimator=final_estimator)
st_model.fit(X_train, y_train)

ensemble_train_pred = st_model.predict(X_train)
ensemble_train_acc = accuracy_score(y_train, ensemble_train_pred)
print(f"Ensemble 訓練數據準確度: {ensemble_train_acc:.4f}")

ensemble_test_pred = st_model.predict(X_test)
ensemble_test_acc = accuracy_score(y_test, ensemble_test_pred)
print(f"Ensemble 測試數據準確度: {ensemble_test_acc:.4f}")
print(f"混淆矩陣:\n{confusion_matrix(y_test, ensemble_test_pred)}")
print(f"分類報告:\n{classification_report(y_test, ensemble_test_pred)}\n")

# 選出最佳模型
if ensemble_test_acc >= accuracy_score(y_test, y_pred) and ensemble_test_acc >= accuracy_score(y_test, test_pred):
    model = st_model
elif accuracy_score(y_test, y_pred) >= accuracy_score(y_test, test_pred) and accuracy_score(y_test, y_pred) >= ensemble_test_acc:
    model = best_model
else:
    model = best_model_modified
print(f"最佳模型: {model}\n")

# 儲存最終模型
model_package = {'model': model, 
                 'feature_names': X_train.columns.tolist()}
joblib.dump(model_package, r'D:\Python\我的AI作品集\專案1_數學函數辨識\models\final_model.joblib')
print("已儲存最終模型!")