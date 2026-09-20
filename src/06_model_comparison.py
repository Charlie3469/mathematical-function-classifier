# 06_model_comparison.py
import pandas as pd
import joblib
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

df = pd.read_csv(r'D:\Python\我的AI作品集\專案1_數學函數辨識\data\function_dataset_cleaned.csv')
X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)

# 建立模型庫
model_names = ['LogisticRegression', 'DecisionTree', 'RandomForest', 'KNN', 'SVM']
results = []
trained_models = {}

for name in model_names:
    model = load(rf'D:\Python\我的AI作品集\專案1_數學函數辨識\models\{name}_baseline.joblib')   # 載入模型
    trained_models[name] = model        # 儲存模型物件

    # 進行預測
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    results.append({'Model': name, 
                    'Accuracy': acc, 
                    'Precision': pre, 
                    'Recall': recall, 
                    'F1': f1})
results_df = pd.DataFrame(results)

# 儲存成網頁
results_df.to_html(r'D:\Python\我的AI作品集\專案1_數學函數辨識\data\Model_Comparison.html')

print("模型效能比較:")
print(results_df.to_string(index=False, formatters={'Accuracy': '{:.4f}'.format, 
                                                    'Precision': '{:.4f}'.format, 
                                                    'Recall': '{:.4f}'.format, 
                                                    'F1': '{:.4f}'.format}))

# 找出 F1 最高的模型
best_result = results_df.loc[results_df['F1'].idxmax()]
best_model_name = best_result['Model']
best_model = trained_models[best_model_name]
print(f"\n最佳模型:\n{best_result}")
joblib.dump(best_model, r'D:\Python\我的AI作品集\專案1_數學函數辨識\models\best_model.joblib')
print("最佳模型已儲存!")