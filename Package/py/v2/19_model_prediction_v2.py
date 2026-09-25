# 19_model_prediction_v2.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import *
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

df = pd.read_csv(f'{DATA_DIR}/v2/function_dataset_cleaned_v2.csv')
X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)

models = {"LogisticRegression": make_pipeline(StandardScaler(),
                                              LogisticRegression(max_iter=1000)),
          "DecisionTree": DecisionTreeClassifier(random_state=10),
          
          "RandomForest": RandomForestClassifier(random_state=10),
          
          "KNN": make_pipeline(StandardScaler(),
                               KNeighborsClassifier(n_neighbors=3)),
          "SVM": make_pipeline(StandardScaler(),
                               SVC())
         }

results = []
trained_models = {}

# 預測模型
for name, model in models.items():
    model = joblib.load(f'{MODEL_DIR}/v2/{name}_baseline_v2.joblib')
    trained_models[name] = model        # 儲存模型物件

    # 進行預測
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"正在預測 {name} 模型:")
    print(f"準確度: {acc:.4f}")
    print(f"混淆矩陣:\n{confusion_matrix(y_test, y_pred)}")
    print(f"分類報告:\n{classification_report(y_test, y_pred)}\n")
    results.append({'Model': name, 
                    'Accuracy': acc, 
                    'Precision': pre, 
                    'Recall': recall, 
                    'F1': f1})
results_df = pd.DataFrame(results)

# 儲存成網頁
results_df.to_html(f'{DATA_DIR}/v2/model_results_v2.html')

print("模型效能比較:")
print(results_df.to_string(index=False, formatters={'Accuracy': '{:.3f}'.format, 
                                                    'Precision': '{:.3f}'.format, 
                                                    'Recall': '{:.3f}'.format, 
                                                    'F1': '{:.3f}'.format}))

# 找出 F1 最高的模型
best_result = results_df.loc[results_df['F1'].idxmax()]
best_model_name = best_result['Model']
best_model = trained_models[best_model_name]
print(f"\n最佳模型:\n{best_result}")
joblib.dump(best_model, f'{MODEL_DIR}/v2/best_model_v2.joblib')
print("已儲存 V2 最佳模型!")