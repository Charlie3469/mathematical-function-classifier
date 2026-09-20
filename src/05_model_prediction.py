# 05_model_prediction.py
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import *

df = pd.read_csv(r'D:\Python\我的AI作品集\專案1_數學函數辨識\data\function_dataset_cleaned.csv')
X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)

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

# 預測模型
for name, model in models.items():
    model = load(rf'D:\Python\我的AI作品集\專案1_數學函數辨識\models\{name}_baseline.joblib')
    y_pred = model.predict(X_test)
    print(f"{name} 模型計算準確度: {accuracy_score(y_test, y_pred):.4f}")
    print(f"{name} 模型的混淆矩陣:\n{confusion_matrix(y_test, y_pred)}")
    print(f"{name} 模型的分類報告:\n{classification_report(y_test, y_pred)}\n")
