# 27_ensemble_v2.py
import pandas as pd
import joblib
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *

# 讀取資料
data = pd.read_csv(r'D:\Python\我的AI作品集\專案1_數學函數辨識\data\v2\function_dataset_merged_v2.csv')
X = data.drop(columns=['label'])
y = data['label']

train_rows = int(len(data)*0.8)
X_train = X.iloc[:train_rows].copy()
X_test = X.iloc[train_rows:].copy()
y_train = y.iloc[:train_rows].copy()
y_test = y.iloc[train_rows:].copy()

baseline_model = joblib.load(r'D:\Python\我的AI作品集\專案1_數學函數辨識\models\v2\best_model_v2.joblib')
baseline_model.fit(X_train, y_train)
baseline_train_pred = baseline_model.predict(X_train)
baseline_test_pred = baseline_model.predict(X_test)
baseline_train_acc = accuracy_score(y_train, baseline_train_pred)
baseline_test_acc = accuracy_score(y_test, baseline_test_pred)

tuned = joblib.load(r'D:\Python\我的AI作品集\專案1_數學函數辨識\models\v2\best_model_tuned_v2.joblib')
tuned.fit(X_train, y_train)
tuned_train_pred = tuned.predict(X_train)
tuned_test_pred = tuned.predict(X_test)
tuned_train_acc = accuracy_score(y_train, tuned_train_pred)
tuned_test_acc = accuracy_score(y_test, tuned_test_pred)

# 定義學習器
base_learners = [('DecisionTree', DecisionTreeClassifier(random_state=10)),
                 ('RandomForest', tuned)]
final_estimator = make_pipeline(StandardScaler(), 
                                SVC(random_state=10))

# 建立 Stacking 模型
st_model = StackingClassifier(estimators=base_learners, final_estimator=final_estimator)
st_model.fit(X_train, y_train)
ensemble_train_pred = st_model.predict(X_train)
ensemble_test_pred = st_model.predict(X_test)
ensemble_train_acc = accuracy_score(y_train, ensemble_train_pred)
ensemble_test_acc = accuracy_score(y_test, ensemble_test_pred)
print("Ensemble 模型評估:")
print(f"訓練數據準確度: {ensemble_train_acc:.4f}")
print(f"測試數據準確度: {ensemble_test_acc:.4f}\n")
print(f"混淆矩陣:\n{confusion_matrix(y_test, ensemble_test_pred)}\n")
print(f"分類報告:\n{classification_report(y_test, ensemble_test_pred)}\n")

# 選出測試集表現最好的模型
print("----實驗結果----")
print(f"模型調參前: {baseline_test_acc:.4f}")
print(f"模型調參後: {tuned_test_acc:.4f}")
print(f"Stacking Ensemble: {ensemble_test_acc:.4f}\n")

model_results = {'模型調參前': (baseline_test_acc, baseline_model),
                 '模型調參後': (tuned_test_acc, tuned),
                 'Stacking Ensemble': (ensemble_test_acc, st_model)}
best_model_name = max(model_results, key=lambda name: model_results[name][0])
best_accuracy, final_model = model_results[best_model_name]
print(f"最佳模型: {best_model_name} 模型")
print(f"準確度: {best_accuracy:.3f}\n")

# 儲存最終模型
model_package = {'model': final_model,
                 'feature_names': X_train.columns.tolist()}
joblib.dump(model_package, r'D:\Python\我的AI作品集\專案1_數學函數辨識\models\v2\final_model_v2.joblib')
print("已儲存 V2 最終模型!")