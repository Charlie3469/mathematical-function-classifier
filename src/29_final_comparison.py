# 29_final_comparison.py
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn.base import clone

VERSION_INFO = {"V1": {"sample_points": 100,
                       "raw_features": 100,
                       "feature_count": 106},
                "V2": {"sample_points": 10,
                       "raw_features": 10,
                       "feature_count": 16}}
LABEL_NAMES = {"linear": "Linear",
               "quadratic": "Quadratic",
               "cubic": "Cubic",
               "exponential": "Exponential",
               "logarithmic": "Logarithmic",
               "sine": "Sine",
               "cosine": "Cosine",
               "reciprocal": "Reciprocal"}
LABEL_ORDER = list(LABEL_NAMES.keys())

# 載入 V1 結合後的資料
data1 = pd.read_csv(r'D:\Python\我的AI作品集\專案1_數學函數辨識\data\v1\function_dataset_merged.csv')
X1 = data1.drop(['label'], axis=1)
y1 = data1['label']

train_v1_rows = int(len(data1)*0.8)
X1_train = X1.iloc[:train_v1_rows].copy()
X1_test = X1.iloc[train_v1_rows:].copy()
y1_train = y1.iloc[:train_v1_rows].copy()
y1_test = y1.iloc[train_v1_rows:].copy()

# 載入 V1 最終模型
model_v1_package = joblib.load(r'D:\Python\我的AI作品集\專案1_數學函數辨識\models\v1\final_model.joblib')
v1_final_model = model_v1_package['model']
v1_feature_names = model_v1_package['feature_names']

train_v1_pred = v1_final_model.predict(X1_train)
train_v1_acc = accuracy_score(y1_train, train_v1_pred)
test_v1_pred = v1_final_model.predict(X1_test)
test_v1_acc = accuracy_score(y1_test, test_v1_pred)
v1_error = len(y1_test) - int(np.sum(test_v1_pred == y1_test))

# 測量重新訓練時間
v1_model_for_timing = clone(v1_final_model)

start_time = time.perf_counter()
v1_model_for_timing.fit(X1_train, y1_train)
v1_training_time = time.perf_counter() - start_time

print("----V1 / V2模型總整理----")
print("V1模型:")
print(f"訓練數據準確度: {train_v1_acc:.4f}")
print(f"測試數據準確度: {test_v1_acc:.4f}")
print(f"訓練時間: {v1_training_time:.4f} 秒")
print(f"錯誤率: {v1_error / len(y1_test):.4f}")
print(f"混淆矩陣:\n{confusion_matrix(y1_test, test_v1_pred, labels=LABEL_ORDER)}\n")
print(f"分類報告:\n{classification_report(y1_test, test_v1_pred, labels=LABEL_ORDER,
                                      target_names=[LABEL_NAMES[x] for x in LABEL_ORDER])}\n\n")

# 載入 V2 結合後的資料
data2 = pd.read_csv(r'D:\Python\我的AI作品集\專案1_數學函數辨識\data\v2\function_dataset_merged_v2.csv')
X2 = data2.drop(['label'], axis=1)
y2 = data2['label']

train_v2_rows = int(len(data2)*0.8)
X2_train = X2.iloc[:train_v2_rows].copy()
X2_test = X2.iloc[train_v2_rows:].copy()
y2_train = y2.iloc[:train_v2_rows].copy()
y2_test = y2.iloc[train_v2_rows:].copy()

# 載入 V2 最終模型
model_v2_package = joblib.load(r'D:\Python\我的AI作品集\專案1_數學函數辨識\models\v2\final_model_v2.joblib')
v2_final_model = model_v2_package['model']
v2_feature_names = model_v2_package['feature_names']

train_v2_pred = v2_final_model.predict(X2_train)
train_v2_acc = accuracy_score(y2_train, train_v2_pred)
test_v2_pred = v2_final_model.predict(X2_test)
test_v2_acc = accuracy_score(y2_test, test_v2_pred)
v2_error = len(y2_test) - int(np.sum(test_v2_pred == y2_test))

# 測量重新訓練時間
v2_model_for_timing = clone(v2_final_model)

start_time = time.perf_counter()
v2_model_for_timing.fit(X2_train, y2_train)
v2_training_time = time.perf_counter() - start_time

print("V2模型:")
print(f"訓練數據準確度: {train_v2_acc:.4f}")
print(f"測試數據準確度: {test_v2_acc:.4f}")
print(f"訓練時間: {v2_training_time:.4f} 秒")
print(f"錯誤率: {v2_error / len(y2_test):.4f}")
print(f"混淆矩陣:\n{confusion_matrix(y2_test, test_v2_pred, labels=LABEL_ORDER)}\n")
print(f"分類報告:\n{classification_report(y2_test, test_v2_pred, labels=LABEL_ORDER,
                                      target_names=[LABEL_NAMES[x] for x in LABEL_ORDER])}\n")
print('-'*100)
higher_model = f"V1 {v1_final_model}" if (test_v1_acc >= test_v2_acc) else f"V2 {v2_final_model}"
print(f"最佳模型: {higher_model}\n")

comparison_df = pd.DataFrame([{"version": "V1",
                               "model": type(v1_final_model).__name__,
                               "sample_points": VERSION_INFO["V1"]["sample_points"],
                               "raw_features": VERSION_INFO["V1"]["raw_features"],
                               "feature_count": VERSION_INFO["V1"]["feature_count"],
                               "train_accuracy": train_v1_acc,
                               "test_accuracy": test_v1_acc,
                               "error_rate": v1_error / len(y1_test),
                               "training_time_seconds": v1_training_time},
                              {"version": "V2",
                               "model": type(v2_final_model).__name__,
                               "sample_points": VERSION_INFO["V2"]["sample_points"],
                               "raw_features": VERSION_INFO["V2"]["raw_features"],
                               "feature_count": VERSION_INFO["V2"]["feature_count"],
                               "train_accuracy": train_v2_acc,
                               "test_accuracy": test_v2_acc,
                               "error_rate": v2_error / len(y2_test),
                               "training_time_seconds": v2_training_time}])
comparison_df.to_csv(
    r'D:\Python\我的AI作品集\專案1_數學函數辨識\data\comparison_results\v1_v2_final_comparison.csv',
    index=False)

accuracy_difference = test_v1_acc - test_v2_acc
comparison_json = {"V1": {"model": type(v1_final_model).__name__,
                          "sample_points": VERSION_INFO["V1"]["sample_points"],
                          "raw_features": VERSION_INFO["V1"]["raw_features"],
                          "feature_count": VERSION_INFO["V1"]["feature_count"],
                          "train_accuracy": train_v1_acc,
                          "test_accuracy": test_v1_acc,
                          "error_rate": v1_error / len(y1_test),
                          "training_time_seconds": v1_training_time,
                          "confusion_matrix": confusion_matrix(y1_test, 
                                                               test_v1_pred, 
                                                               labels=LABEL_ORDER).tolist()},
                   "V2": {"model": type(v2_final_model).__name__,
                          "sample_points": VERSION_INFO["V2"]["sample_points"],
                          "raw_features": VERSION_INFO["V2"]["raw_features"],
                          "feature_count": VERSION_INFO["V2"]["feature_count"],
                          "train_accuracy": train_v2_acc,
                          "test_accuracy": test_v2_acc,
                          "error_rate": v2_error / len(y2_test),
                          "training_time_seconds": v2_training_time,
                          "confusion_matrix": confusion_matrix(y2_test, 
                                                               test_v2_pred, 
                                                               labels=LABEL_ORDER).tolist()},
                   "comparison": {"accuracy_difference_v1_minus_v2": accuracy_difference,
                                  "higher_test_accuracy_version": higher_model}
                  }
with open(r'D:\Python\我的AI作品集\專案1_數學函數辨識\data\comparison_results\v1_v2_final_comparison.json',
    'w', encoding='utf-8') as f:
    json.dump(comparison_json, f, ensure_ascii=False, indent=4)
print("最終比較結果已儲存!")