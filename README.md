# 專案｜數學函數辨識（Mathematical Function Classifier）
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

本專案從人工設計的數學函數資料出發，建立了：

**問題定義 → 生成資料 → 資料探索(EDA) → 資料前處理 → 模型訓練 → 預測 → 比較 → 錯誤分析 → 特徵工程 → 模型調參 → 集成學習 → 最後做成使用者可操作介面 → 上傳到GitHub部署**

完整流程。
最終希望建立一個可以從數值資料推測函數類型，並進一步估計函數參數的系統。

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
本專案主要探討以下問題：
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
從原本 V1 的 100 個樣本點縮減到 10 個樣本點，可以讓使用者輸入方便。
包含了:
- 8 種函數
- 每種 500 筆
- 每筆資料 10 個取樣值
- 資料加入 Gaussian noise

---
## 研究過程
本專案依照完整的機器學習流程進行，主要包含：
### 1. 資料生成
使用 NumPy 隨機生成 8 種數學函數，每種約 500 筆，共約 4,000 筆資料，並加入 Gaussian noise。
### 2. 資料探索（EDA）
觀察資料的：
- 形狀
- 欄位數
- 有無遺失值
- 各類別分布
- 函數數值範圍
- 函數圖形與基本統計特徵
### 3. 資料前處理
進行：
- 資料清理
- 刪除不必要的數據，以免資料洩漏
- 資料分割
### 4. 模型建立與預測
使用 5 種分類模型建立 Baseline：
- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- SVM
接著利用混淆矩陣與分類報告來做評估與比較。
### 5. 錯誤分析
進一步觀察模型在哪些函數類別容易判斷錯誤，並針對錯誤樣本結合 metadata 進行分析。
在 V1 版本中， Sine 與 Cosine 是較明顯的混淆來源，整體錯誤率高達62%，因此特別分析：
- Sine 被誤判成 Cosine
- Cosine 被誤判成 Sine
- Noise level
- 函數生成參數與分類錯誤的關係
而在 V2 版本中， Logarithm 與 Reciprocal 是較明顯的混淆來源，整體錯誤率也來到30%
### 6. 特徵工程
除了原本的函數取樣值外，加入數學相關的手工特徵，例如：
- 迴歸斜率 `slope`
- 總變化量 `amplitude`
- 一階差分的平均、標準差 `first_diff_mean` `first_diff_std`
- 二階差分的平均、標準差 `second_diff_mean` `second_diff_std`
藉此讓模型除了觀察原始樣本之外，也能取得部分描述函數形狀的資訊。同時分析最佳模型的特徵重要性，觀察哪些取樣位置對分類較有影響。
### 7. 模型調參
使用 `GridSearchCV` 對最佳模型進行參數搜尋及調整，例如:
(以下是RandomForest的範例)
- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
並比較調參前後的模型表現。
### 8. Ensemble
使用 Stacking Ensemble 將決策樹、隨機森林與支援向量機結合，測試是否能改善分類結果。
### 9. 最終模型
經過上述的實驗後，將原始模型、調參後的模型與 Stacking Ensemble 進行比較，最後選擇測試集準確率最高的模型作為最終模型，並以`joblib`儲存，同時也保存模型與特徵名稱，供後續上傳 GitHub 部署使用。
### 10. Streamlit App
本專案另外建立 Streamlit 使用者介面。
使用者可以直接輸入函數取樣值，系統會自動：

```text
輸入取樣值
      ↓
資料前處理
      ↓
建立 Features
      ↓
訓練並預測模型
      ↓
函數分類
      ↓
參數估計
      ↓
圖形化顯示
```

---
## 實驗結果
### 模型比較

| 模型 | V1 Accuracy | V2 Accuracy |
|---|---:|---:|
| Logistic Regression | 18.38% | 19.13% |
| Decision Tree | 62.00% | 58.00% |
| Random Forest | **69.63%** | **70.88%** |
| KNN | 63.88% | 60.13% |
| SVM | 35.88% | 47.00% |

其中 Random Forest 的測試集準確率皆為最高，因此 **Random Forest** 是本專案的主要訓練模型。

### 錯誤分析
在專案 V1 初期，**Sine 與 Cosine** 曾是較明顯的混淆來源，因此在 V2 版本有進一步分析函數生成參數與 phase 範圍(例如:從[-pi/2, pi/2]調整成[-pi/4, pi/4])，並調整資料生成條件。調整後，兩者之間的混淆狀況有明顯降低。
但目前仍有其他部分類別容易混淆，顯示不同函數在特定參數與雜訊條件下，可能具有相似的局部形狀。

### 特徵工程
後續加入一些數學相關特徵，包括迴歸斜率、變化量、一階差分與二階差分等，將原本的取樣值擴充特徵。
實驗結果顯示：

| 版本 | V1 Accuracy | V2 Accuracy |
|---|---:|---:|
| 原始取樣值 | 69.63% | 70.87% |
| 加入數學特徵後 | **69.50%** | **78.25%** |
| 改善程度 | **-0.13%** | **+7.37%** |

因此，在 V1 版本資料與模型設定下，數學特徵沒有改善整體分類效能，反而在V2版本有改善許多。
不過，Feature Importance 分析仍發現部分數學特徵具有一定的重要性，其中 Amplitude 的重要性最高，為0.1697，表示這些特徵可能包含模型可以利用的資訊，但其資訊與原始函數取樣值可能存在部分重複，因此加入特徵後不一定能直接提升測試準確率。

### 模型調參
使用 GridSearchCV 對 Random Forest 模型進行參數搜尋後，最佳參數為：

| 參數 | V1 | V2 |
|---|---:|---:|
| n_estimators | 150 | 100 |
| max_depth | 20 | None |
| min_samples_split | 3 | 5 |
| min_samples_leaf | 1 | 1 |
| CV準確度 | 0.7116 | 0.7653 |

調參後實驗結果顯示：

| 版本 | V1 Accuracy | V2 Accuracy |
|---|---:|---:|
| 模型調參前 | 69.50% | 78.25% |
| 模型調參後 | **72.25%** | **78.75%** |
| 改善程度 | **+2.75%** | **+0.50%** |

可以發現，調參後兩個版本的模型都有改善。

### 集成學習
再使用 `StackingClassifier`，結合：
- Decision Tree
- Random Forest
- SVM
建立 Stacking Ensemble。
實驗結果顯示：

| 版本 | V1 Accuracy | V2 Accuracy |
|---|---:|---:|
| 模型調參前 | 69.50% | 78.25% |
| 模型調參後 | 66.87% | 68.87% |
| Ensemble | **70.63%** | **69.00%** |

因此在V1模型中， Ensemble 為該版本準確度最高的模型，而在V2模型中，Ensemble 並沒有優於調參前模型，反而是調參前的 Random Forest 模型是三者當中最高的。

### 結論
???

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
最後在實作 app V1 介面的時候，發現使用者一定只能輸入 100 個樣本點，否則系統就不會自動判斷，所以在後續V2版本中有改成只要輸入 10 個取樣點就能預測。
### 5. 輸出結果有時會出錯
在輸入"0, 0.5, 1, 1.5, ..., 49, 49.5"這個線性數據之後，得出來的結果竟然是Exponential(照理來說應該是Linear)。後來有查出原因是需要標準化數據才能預測正確。
### 6. V1 版本只顯示n次多項式的係數
在V1版本中，只有成功預測出多項式，而在V2版本中有利用
```python
scipy.optimize.curve_fit
```
來解決預測更多函數的問題。

---
## 未來展望
雖然 V2 已完成，但後續仍可以從以下方向延伸:
### 資料面
- 建立獨立的 Final Holdout Test Set，避免反覆使用測試集進行模型調整
- 測試更多不同程度的 Noise
- 建立更接近真實觀測資料的測試集
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
### 更好的模型
後續可以比較更多模型、使用更穩定的交叉驗證方式，或嘗試相關深度學習等概念能直接處理序列型資料的方法。
例如:
* XGBoost
* LightGBM
* 1D CNN / TCN
* 神經網路

### 使用者介面
未來希望將目前的「輸入數值」介面進一步擴充為圖形化辨識系統：
- 手繪函數圖形
- 由函數圖形直接辨識

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
├─ Package/
│  ├─ data/
│  │  ├─ v1/
│  │  │  ├─ function_dataset.csv
│  │  │  ├─ function_dataset_cleaned.csv
│  │  │  ├─ function_dataset_reset.csv
│  │  │  ├─ function_metadata.csv
│  │  │  ├─ model_results.html
│  │  │  ├─ parameter_results.html
│  │  │  ├─ advanced_parameter_results.html
│  │  │  ├─ advanced_error_analysis.png
│  │  │  ├─ feature_importance.html
│  │  │  ├─ feature_importance_results.png
│  │  │  └─ v1_results.png
│  │  │
│  │  ├─ v2/
│  │  │  ├─ function_dataset_v2.csv
│  │  │  ├─ function_dataset_cleaned_v2.csv
│  │  │  ├─ function_dataset_reset_v2.csv
│  │  │  ├─ function_metadata_v2.csv
│  │  │  ├─ model_results_v2.html
│  │  │  ├─ parameter_results_v2.html
│  │  │  ├─ advanced_parameter_results_v2.html
│  │  │  ├─ advanced_error_analysis_v2.png
│  │  │  ├─ feature_importance_v2.html
│  │  │  ├─ feature_importance_results_v2.png
│  │  │  └─ v2_results.png
│  │  │
│  │  ├─ final_results.csv
│  │  ├─ final_results.html
│  │  └─ final_comparision_results.json
│  │
│  ├─ models/
│  │  ├─ v1/
│  │  │  ├─ LogisticRegression_baseline.joblib
│  │  │  ├─ DecisionTree_baseline.joblib
│  │  │  ├─ RandomForest_baseline.joblib
│  │  │  ├─ KNN_baseline.joblib
│  │  │  ├─ SVM_baseline.joblib
│  │  │  ├─ best_model_tuned.joblib
│  │  │  ├─ best_model.joblib
│  │  │  └─ final_model.joblib
│  │  │
│  │  └─ v2/
│  │     ├─ LogisticRegression_baseline_v2.joblib
│  │     ├─ DecisionTree_baseline_v2.joblib
│  │     ├─ RandomForest_baseline_v2.joblib
│  │     ├─ KNN_baseline_v2.joblib
│  │     ├─ SVM_baseline_v2.joblib
│  │     ├─ best_model_tuned_v2.joblib
│  │     ├─ best_model_v2.joblib
│  │     └─ final_model_v2.joblib
│  │
│  ├─ py/
│  │  ├─ v1/
│  │  │  ├─ 01_function_generator.py
│  │  │  ├─ 02_data_exploration.py
│  │  │  ├─ 03_data_preparation.py
│  │  │  ├─ 04_model_training.py
│  │  │  ├─ 05_model_prediction.py
│  │  │  ├─ 06_error_analysis.py
│  │  │  ├─ 07_parameter_analysis.py
│  │  │  ├─ 08_advanced_analysis.py
│  │  │  ├─ 09_feature_engineering.py
│  │  │  ├─ 10_feature_experiment.py
│  │  │  ├─ 11_feature_importance.py
│  │  │  ├─ 12_model_tuning.py
│  │  │  ├─ 13_ensemble.py
│  │  │  └─ 14_final_model.py
│  │  │
│  │  ├─ v2/
│  │  │  ├─ 15_function_generator_v2.py
│  │  │  ├─ 16_data_exploration_v2.py
│  │  │  ├─ 17_data_preparation_v2.py
│  │  │  ├─ 18_model_training_v2.py
│  │  │  ├─ 19_model_prediction_v2.py
│  │  │  ├─ 20_error_analysis_v2.py
│  │  │  ├─ 21_parameter_analysis_v2.py
│  │  │  ├─ 22_advanced_analysis_v2.py
│  │  │  ├─ 23_feature_engineering_v2.py
│  │  │  ├─ 24_feature_experiment_v2.py
│  │  │  ├─ 25_feature_importance_v2.py
│  │  │  ├─ 26_model_tuning_v2.py
│  │  │  ├─ 27_ensemble_v2.py
│  │  │  └─ 28_final_model_v2.py
│  │  │
│  │  └─ 29_final_comparison.py
│  │
│  ├── .gitignore
│  ├── 01_eda.ipynb
│  ├── 02_models.ipynb
│  ├── 03_error_analysis.ipynb
│  ├── 04_advanced_analysis.ipynb
│  ├── 05_features.ipynb
│  ├── 06_final.ipynb
│  ├── 07_eda_v2.ipynb
│  ├── 08_models_v2.ipynb
│  ├── 09_error_analysis_v2.ipynb
│  ├── 10_advanced_analysis_v2.ipynb
│  ├── 11_features_v2.ipynb
│  ├── 12_final_v2.ipynb
│  └── 13_final_comparison.ipynb
│
├─ app.py
├─ app_v2.py
├─ README.md
└─ requirements.txt
```

---
## 專案版本狀態
- 2026/9/4 有初步想法
- 2026/9/7 開始動工
- 2026/9/20 完成 V1 版本作品
- 2026/9/25 完成 V2 版本作品

總共花了約3個禮拜時間

雖然有比 V1 版本更進步了一些，但是我覺得 V2 版本可以有更好的地方，像是輸入不定數量也能預測，不用像 V1 或 V2 一樣只能輸入特定數量才能預測，這也是未來 V3 版本可以改善的問題，並進一步進化成使用者畫圖就能輸出那個是甚麼函數的介面。

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
