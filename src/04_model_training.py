# 04_model_training.py
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

df = pd.read_csv(r'D:\Python\我的AI作品集\專案1_數學函數辨識\data\v1\function_dataset_cleaned.csv')

# 定義特徵與目標變數
X = df.drop(columns=['label'])
y = df['label']

# 將數據分割為訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)
print(f"訓練集類別:\n{y_train.value_counts()}\n")
print(f"測試集類別:\n{y_test.value_counts()}\n")

# 建立pipeline, 將數據標準化, 並蒐集模型
models = {
    "LogisticRegression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),
    "DecisionTree": DecisionTreeClassifier(random_state=10),

    "RandomForest": RandomForestClassifier(random_state=10),

    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=3)
    ),
    "SVM": make_pipeline(
        StandardScaler(),
        SVC()
    )
}

for name, model in models.items():
    start_time = time.perf_counter()        # 開始計時
    model.fit(X_train, y_train)
    end_time = time.perf_counter()          # 結束計時
    elapsed_time = end_time - start_time    # 計算經過時間
    print(f"{name} 模型訓練完成, 耗時: {elapsed_time:.3f}秒\n")

    # 儲存到joblib
    joblib.dump(model, rf'D:\Python\我的AI作品集\專案1_數學函數辨識\models\v1\{name}_baseline.joblib')
