# 專案: 數學函數辨識（Mathematical Function Classifier）
---
## 專案簡介:
這是一個結合**數學與資工方面**的知識與概念，並融合起來，建立出一個以機器學習分類為主的專案，目標是根據使用者輸入的一些取樣值，來判斷它屬於哪一種函數類型。專案目前包含 8 種函數：
- Linear（線性）
- Quadratic（二次式）
- Cubic（三次式）
- Exponential（指數）
- Logarithmic（對數）
- Sine（正弦）
- Cosine（餘弦）
- Reciprocal（反比例 / 倒數函數）


---
## 研究動機:
看到一組函數資料或圖形時，人通常可以根據其變化趨勢與形狀推測函數類型，例如：
- 線性函數具有近似固定的變化率
- 二次與三次函數具有不同的彎曲特徵
- 指數與對數函數具有不同的成長 / 衰減型態
- 三角函數具有週期性
- 反比例函數具有特殊的非線性變化
本專案希望將這種「從資料判斷函數」的數學思考方式，轉換成可以實際運作的機器學習分類系統。


---
## 研究問題:

1. 不同數學函數是否能透過取樣值進行分類？
2. 哪些機器學習模型適合這個函數分類問題？
3. 加入數學特徵後，模型表現是否有改善？
4. 將取樣點由 100 點降低到 10 點後，模型是否還可以維持合理的辨識能力？
5. 完成分類後，能進一步估計函數參數嗎？


---
## 研究資料:
### 原始資料生成: `function_dataset.csv` 
資料是以 Python + Numpy 自行生成的合成資料。每筆資料代表一個函數在固定 x 範圍內的取樣結果，並加入隨機 Gaussian noise，使資料不會過度理想化。
資料包含了：
- 8 種數學函數
- 函數取樣值作為主要模型輸入
- `sample_id` 作為樣本識別
- `label` 作為分類答案
- `noise_sigma` 紀錄產生資料時使用的雜訊程度
### 生成Metadata: `function_metadata.csv` 
- 紀錄函數生成時使用的隱藏參數
- 僅用於資料分析、錯誤分析與研究，不直接作為模型輸入，以避免將函數生成參數直接提供給模型而造成**資料洩漏**。
### V2生成的新資料: `function_dataset_v2.csv` 
從原本 V1 的 100 個樣本點縮減到 10 個樣本點，可以讓使用這輸入方便。
包含了:
- 8 種函數
- 每種 500 筆
- 每筆資料 10 個取樣值
- 資料加入 Gaussian noise


---
## 研究過程
本專案依照完整的機器學習流程進行，主要包含：
### 1.資料生成
使用 NumPy 隨機生成 8 種數學函數，每種約 500 筆，共約 4,000 筆資料，並加入 Gaussian noise。
### 2.資料探索（EDA）
觀察資料的：
- 筆數與欄位數
- 各類別分布
- 函數數值範圍
- 函數圖形與基本統計特徵
### 3.資料前處理
進行：
- 資料清理
- 刪除不必要的數據，以免資料洩漏
- 資料分割
### 4.模型建立與預測
使用 5 種分類模型建立 Baseline：
- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- SVM
接著利用混淆矩陣與分類報告來做評估與比較。
### 5.錯誤分析
進一步觀察模型在哪些函數類別容易判斷錯誤，並針對錯誤樣本結合 metadata 進行分析。
其中 Sine 與 Cosine 是較明顯的混淆來源，因此特別分析：
- Sine 被誤判成 Cosine
- Cosine 被誤判成 Sine
- Noise level
- 函數生成參數與分類錯誤的關係
### 6.特徵工程
除了原本的函數取樣值外，加入數學相關的手工特徵，例如：
- 迴歸斜率
- 總變化量
- 一階差分的平均、標準差
- 二階差分的平均、標準差
藉此讓模型除了觀察原始樣本之外，也能取得部分描述函數形狀的資訊。同時分析最佳模型的特徵重要性，觀察哪些取樣位置對分類較有影響。
### 7.模型調參
使用 `GridSearchCV` 對最佳模型進行參數搜尋及調整，例如:(以下是RandomForest的範例)
- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
並比較調參前後的模型表現。
### 8.集成學習
使用 `StackingClassifier` 將決策樹、隨機森林與支援向量機結合，測試是否能改善分類結果。
### 9.最終模型
經過上述的實驗後，將原始模型、調參後的模型與 Stacking Ensemble 進行比較，最後選擇測試集準確率最高的模型作為最終模型，並以`joblib`儲存，同時也保存模型與特徵名稱，供後續上傳 GitHub 部署使用。


---
## 實驗結果
### 模型比較

| 模型 | V1 Accuracy | V2 Accuracy |
|---|---|---:|
| Logistic Regression | 45.50% | 21.62% |
| Decision Tree | 71.00% | 60.25% |
| Random Forest | **82.13%** | **74.25%** |
| KNN | 81.00% | 71.37% |
| SVM | 76.00% | 59.00% |

其中 Random Forest 的測試集準確率皆為最高，因此 **Random Forest** 是本專案的主要訓練模型。

### 錯誤分析
在專案 V1 初期，**Sine 與 Cosine** 曾是較明顯的混淆來源，因此進一步分析函數生成參數與 phase 範圍(例如:
從[-pi/2, pi/2]調整成[-pi/4, pi/4])，並調整資料生成條件。調整後，兩者之間的混淆狀況有明顯降低。
但目前仍有其他部分類別容易混淆，顯示不同函數在特定參數與雜訊條件下，可能具有相似的局部形狀。

### 特徵工程
後續加入一些數學相關特徵，包括迴歸斜率、變化量、一階差分與二階差分等，將原本的取樣值擴充特徵。
實驗結果顯示：
V1：
- 原始取樣值：**82.13%**
- 加入數學特徵後：**81.63%**
測試準確率**下降約 0.50 個百分點**
V2：
- 原始取樣值：**74.25%**
- 加入數學特徵後：**82.50%**
測試準確率**上升約 8.25 個百分點**
因此可得知:在V1版本資料與模型設定下，數學特徵沒有改善整體分類效能，反而在V2版本有改善許多。
不過，Feature Importance 分析仍發現部分數學特徵具有一定的重要性，其中 Amplitude 的重要性最高，表示這些特徵可能包含模型可以利用的資訊，但其資訊與原始函數取樣值可能存在部分重複，因此加入特徵後不一定能直接提升測試準確率。

### 模型調參
使用 GridSearchCV 對 Random Forest 模型進行參數搜尋後，最佳參數為：
```text
n_estimators = 150
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
```
調參後測試集準確率為**81.63%**，與原本使用 106 個特徵的 Random Forest 模型相同，因此在目前 V1 搜尋範圍內，模型調參並沒有進一步提升測試集表現。

### 集成學習
再使用 `StackingClassifier`，結合：
- Decision Tree
- Random Forest
- SVM
建立 Stacking Ensemble。
結果 V1 Ensemble 測試集準確率：**81.37%**，也沒有優於單一 Random Forest 模型。

### 結論
因此目前的實驗結果：
> Random Forest（100 個原始取樣值）以 82.13% 測試集準確率，為目前 V1 實驗中表現較高的單模型。

另一方面，特徵工程、模型調參與 Stacking Ensemble 在目前設定下都沒有進一步提升測試集準確率。
這些實驗結果也成為後續版本進一步改善資料、特徵與模型的重要依據。

---
## 研究過程遇到的困難
### 1. 使用的是合成資料
目前資料由程式自行產生，和真實世界中的數學資料、感測器資料或手繪曲線仍有差異，因此模型在真實資料上的表現尚未經過驗證。
### 2. 資料集調整
在專案初期，曾使用過約 24,000 筆資料的較大版本。由於實際運算成本較高，電腦要跑的時間也相當久，所以後續又重新調整資料規模，因此目前 V1 版本使用的是 4,000 筆資料。這次調整也成為本專案實作過程中的重要經驗：
**資料規模不是越大越好，還必須考慮實際的運算資源與實驗成本。**
### 3. 特徵工程的改善幅度有限
加入目前設計的數學特徵後，模型表現反而小幅下降，表示單靠目前這幾個統計/差分特徵還不足以改善函數間的分類效果。
### 4. 使用者只能輸入一定數量的樣本點
最後在實作`app.py`的時候，發現使用者一定只能輸入100個樣本點，否則系統就不會自動判斷，這也是後續V2版本需要改善的問題，看看能不能讓使用者只要輸入幾個樣本點就能預測。
### 5. 輸出結果有時會出錯
在輸入"0, 0.5, 1, 1.5, ..., 49, 49.5"這個線性數據之後，得出來的結果竟然是Exponential(照理來說應該是Linear)。後來有查出原因是需要標準化數據才能預測正確。
### 6. 目前版本只顯示n次多項式的係數
因為目前技術還沒有到可以顯示出其他函數的係數，因此這也是V2版本需要改善的因素之一。

---
## 未來展望
雖然第一版已完成，但後續仍可以從以下方向延伸:
### 更好的資料
- 增加更多函數參數範圍
- 縮減樣本點
- 加入真實世界或人工繪製的曲線資料
### 更好的數學特徵
可以增加：
- Zero crossing
- Curvature / 二階導數相關特徵
- 對稱性
- 頻率與週期特徵
- FFT / Frequency-domain features
- 峰值與谷值數量
這些特徵可能有助於進一步區分其他形狀相近的函數。
### 更好的模型與驗證方式
後續可以比較更多模型、使用更穩定的交叉驗證方式，或嘗試相關深度學習等概念能直接處理序列型資料的方法。
### 部署與使用者介面
後續也可以進一步加入手寫函數或手繪曲線的輸入方式，讓使用者直接透過畫圖的方式提供函數資料，再由模型進行辨識。

---
## 專案流程

```text
資料生成
   ↓
資料探索（EDA）
   ↓
資料前處理
   ↓
訓練模型
   ↓
模型預測 & 比較
   ↓
錯誤分析
   ↓
Metadata / Noise Analysis
   ↓
參數分析
   ↓
特徵工程
   ↓
模型調參
   ↓
Stacking Ensemble
   ↓
最終模型儲存
   ↓
使用者介面的開發
   ↓
整理資料
   ↓
上傳到GitHub部署
```

---
## 專案結構
```text
專案1_數學函數辨識/
│
├─ data/
│  ├─ v1/
│  │  ├─ function_dataset.csv
│  │  ├─ function_dataset_cleaned.csv
│  │  ├─ function_dataset_merged.csv
│  │  ├─ function_metadata.csv
│  │  ├─ Model_Comparison.html
│  │  ├─ parameter_analysis.html
│  │  ├─ error_analysis.png
│  │  ├─ sine_cosine_parameter_analysis.html
│  │  ├─ sine_cosine_error_analysis.png
│  │  ├─ feature_importance.html
│  │  ├─ feature_importance_top20.png
│  │  └─ final_confusion_matrix.png
│  │
│  ├─ v2/
│  │  ├─ function_dataset_v2.csv
│  │  ├─ function_dataset_cleaned_v2.csv
│  │  ├─ function_dataset_merged_v2.csv
│  │  ├─ function_metadata_v2.csv
│  │  ├─ Model_Comparison_v2.html
│  │  ├─ parameter_analysis_v2.html
│  │  ├─ error_analysis_v2.png
│  │  ├─ sine_cosine_parameter_analysis_v2.html
│  │  ├─ sine_cosine_error_analysis_v2.png
│  │  ├─ feature_importance_v2.html
│  │  ├─ feature_importance_top10_v2.png
│  │  └─ final_confusion_matrix_v2.png
│  │
│  └─ comparison_results/
│     ├─ v1_v2_final_comparision.csv
│     ├─ v1_v2_final_comparision.html
│     └─ v1_v2_final_comparision.json
│
├─ models/
│  ├─ v1/
│  │  ├─ LogisticRegression_baseline.joblib
│  │  ├─ DecisionTree_baseline.joblib
│  │  ├─ RandomForest_baseline.joblib
│  │  ├─ KNN_baseline.joblib
│  │  ├─ SVM_baseline.joblib
│  │  ├─ best_model_tuned.joblib
│  │  ├─ best_model.joblib
│  │  └─ final_model.joblib
│  │
│  └─ v2/
│     ├─ LogisticRegression_baseline_v2.joblib
│     ├─ DecisionTree_baseline_v2.joblib
│     ├─ RandomForest_baseline_v2.joblib
│     ├─ KNN_baseline_v2.joblib
│     ├─ SVM_baseline_v2.joblib
│     ├─ best_model_tuned_v2.joblib
│     ├─ best_model_v2.joblib
│     └─ final_model_v2.joblib
│
├─ notebooks/
│  ├── 01_function_generator.ipynb
│  ├── 02_eda.ipynb
│  ├── 03_models.ipynb
│  ├── 04_error_analysis.ipynb
│  ├── 05_features.ipynb
│  ├── 06_ensembling.ipynb
│  ├── 07_final.ipynb
│  ├── 08_function_generator_v2.ipynb
│  ├── 09_eda_v2.ipynb
│  ├── 10_models_v2.ipynb
│  ├── 11_error_analysis_v2.ipynb
│  ├── 12_features_v2.ipynb
│  ├── 13_ensembling_v2.ipynb
│  ├── 14_final_v2.ipynb
│  └── 15_final_comparison.ipynb
│
├─ src/
│  ├─ v1/
│  │  ├─ 01_function_generator.py
│  │  ├─ 02_data_exploration.py
│  │  ├─ 03_data_preparation.py
│  │  ├─ 04_model_training.py
│  │  ├─ 05_model_prediction.py
│  │  ├─ 06_error_analysis.py
│  │  ├─ 07_parameter_analysis.py
│  │  ├─ 08_sine_cosine_analysis.py
│  │  ├─ 09_feature_engineering.py
│  │  ├─ 10_feature_experiment.py
│  │  ├─ 11_feature_importance.py
│  │  ├─ 12_model_tuning.py
│  │  ├─ 13_ensemble.py
│  │  └─ 14_final_model.py
│  │
│  ├─ v2/
│  │  ├─ 15_function_generator_v2.py
│  │  ├─ 16_data_exploration_v2.py
│  │  ├─ 17_data_preparation_v2.py
│  │  ├─ 18_model_training_v2.py
│  │  ├─ 19_model_prediction_v2.py
│  │  ├─ 20_error_analysis_v2.py
│  │  ├─ 21_parameter_analysis_v2.py
│  │  ├─ 22_sine_cosine_analysis_v2.py
│  │  ├─ 23_feature_engineering_v2.py
│  │  ├─ 24_feature_experiment_v2.py
│  │  ├─ 25_feature_importance_v2.py
│  │  ├─ 26_model_tuning_v2.py
│  │  ├─ 27_ensemble_v2.py
│  │  └─ 28_final_model_v2.py
│  │
│  └─ 29_final_comparison.py
│
├─ app.py
├─ app_v2.py
├─ requirements.txt
└─ README.md
```

---
## 專案版本狀態
- 2026/9/4有初步想法
- 2026/9/7開始動工
- 2026/9/20完成 V1 版本作品
- 預計 2026/9/27 完成 V2 版本作品
共花了約3個禮拜時間

後續 V3 版本將從新的資料生成與實驗流程開始，並重新進行模型訓練、模型比較、錯誤分析、特徵工程與模型調整。也會再針對目前的結果、模型表現、錯誤分析與整體專案內容進行整理與改善。

---
## 使用技術
- Python==3.12.10
- numpy==2.3.5
- pandas==2.3.3
- scikit-learn==1.9.0
- matplotlib==3.10.0
- seaborn
- joblib
- scipy

---
## 學習重點
這個專案主要練習的不只是模型 API，而是完整理解機器學習專案的工作流程：
**問題定義 → 生成資料 → 資料探索(EDA) → 資料前處理 → 模型訓練 → 預測 → 比較 → 錯誤分析 → 特徵工程 → 模型調參 → 集成學習 → 最後做成使用者可操作介面 → 上傳到GitHub部署**
透過這次專案，也實際遇到資料規模與電腦運算能力之間的取捨，讓整個流程更接近實際 AI 專案開發。
