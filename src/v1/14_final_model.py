# 14_final_model.py
import pandas as pd
from joblib import load
from sklearn.metrics import *
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Microsoft JhengHei"]

labels = ["linear", "quadratic", "cubic", "exponential", "logarithmic", "sine", "cosine", "reciprocal"]

data = pd.read_csv(r'D:\Python\我的AI作品集\專案1_數學函數辨識\data\v1\function_dataset_merged.csv')
X = data.drop(['label'], axis=1)
y = data['label']

# 分割數據
train_rows = int(len(data)*0.8)
X_train = X.iloc[:train_rows].copy()
X_test = X.iloc[train_rows:].copy()
y_train = y.iloc[:train_rows].copy()
y_test = y.iloc[train_rows:].copy()

# 載入最終模型
model_package = load(r'D:\Python\我的AI作品集\專案1_數學函數辨識\models\v1\final_model.joblib')
final_model = model_package['model']
feature_names = model_package['feature_names']

train_pred = final_model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
test_pred = final_model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)

print("----最終結果----")
print(f"訓練數據準確度: {train_acc:.4f}")
print(f"測試數據準確度: {test_acc:.4f}")
print(f"混淆矩陣:\n{confusion_matrix(y_test, test_pred)}\n")

plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix(y_test, test_pred), 
            cmap='coolwarm', 
            xticklabels=[i for i in labels], 
            yticklabels=[i for i in labels],
            annot=True)
plt.xticks(rotation=30)
plt.yticks(rotation=0)
plt.title('最終模型混淆矩陣熱力圖')
plt.legend()
plt.savefig(r'D:\Python\我的AI作品集\專案1_數學函數辨識\data\v1\final_confusion_matrix.png')
plt.show()

print(f"分類報告:\n{classification_report(y_test, test_pred)}\n")