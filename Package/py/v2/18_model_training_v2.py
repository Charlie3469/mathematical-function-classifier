# 18_model_training_v2.py
import pandas as pd
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

df = pd.read_csv(f'{DATA_DIR}/v2/function_dataset_cleaned_v2.csv')

# 定義特徵與目標變數
X = df.drop(columns=['label'])
y = df['label']

# 將數據分割為訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)
print(f"訓練集類別:\n{y_train.value_counts()}\n")
print(f"測試集類別:\n{y_test.value_counts()}\n")

# 建立pipeline, 將數據標準化, 並蒐集模型
models = {"LogisticRegression": make_pipeline(StandardScaler(),
                                              LogisticRegression(max_iter=1000)),
          "DecisionTree": DecisionTreeClassifier(random_state=10),
          
          "RandomForest": RandomForestClassifier(random_state=10),
          
          "KNN": make_pipeline(StandardScaler(),
                               KNeighborsClassifier(n_neighbors=3)),
          "SVM": make_pipeline(StandardScaler(),
                               SVC())
         }

model_times = []
for name, model in models.items():
    start_time = time.perf_counter()        # 開始計時
    model.fit(X_train, y_train)
    end_time = time.perf_counter()          # 結束計時
    elapsed_time = end_time - start_time    # 計算經過時間
    print(f"{name} 模型訓練完成, 耗時: {elapsed_time:.3f}秒")
    model_times.append(elapsed_time)

    # 儲存到joblib
    joblib.dump(model, f'{MODEL_DIR}/v2/{name}_baseline_v2.joblib')

print(model_times)