# 使用者介面.py
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# 基本設定
N_POINTS = 100
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "v1" / "final_model.joblib"

LABEL_NAMES = {"linear": "Linear(線性)",
               "quadratic": "Quadratic(二次式)",
               "cubic": "Cubic(三次式)",
               "exponential": "Exponential(指數)",
               "logarithmic": "Logarithmic(對數)",
               "sine": "Sine(正弦)",
               "cosine": "Cosine(餘弦)",
               "reciprocal": "Reciprocal(反比例/倒數)"}

st.set_page_config(page_title="AI數學函數辨識",
                   page_icon="📐",
                   layout="centered")

# ----載入模型----
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型檔案:\n{MODEL_PATH}\n\n請確認 final_model.joblib 是否存在。")
    package = joblib.load(MODEL_PATH)

    # 目前 V1 儲存格式：
    # {'model': model, 
    #  'feature_names': [...]}
    if isinstance(package, dict) and "model" in package:
        model = package["model"]
        feature_names = package.get("feature_names")
    else:
        model = package
        feature_names = None

    return model, feature_names

# ----特徵工程----
def make_features(y_values, feature_names):
    y = np.asarray(y_values, dtype=float)

    # 標準化
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.std(y)
    if scale < 1e-8:
        scale = 1.0
    y = (y-np.mean(y)) / scale

    raw_features = {f"y_{i:03d}": y[i] for i in range(N_POINTS)}

    # 數學特徵
    x_index = np.arange(N_POINTS, dtype=float)
    slope = np.polyfit(x_index, y, 1)[0]
    first_diff = np.diff(y)
    second_diff = np.diff(y, n=2)
    math_features = {"slope": slope,
                     "amplitude": (np.max(y) - np.min(y)),
                     "first_diff_mean": np.mean(first_diff),
                     "first_diff_std": np.std(first_diff),
                     "second_diff_mean": np.mean(second_diff),
                     "second_diff_std": np.std(second_diff)}
    all_features = {**raw_features, **math_features}

    # 依訓練時的順序排列
    if feature_names:
        missing = [name for name in feature_names if name not in all_features]
        if missing:
            raise ValueError(f"缺少:{missing}")
        return pd.DataFrame([[all_features[name] for name in feature_names]], columns=feature_names)

    return pd.DataFrame([all_features])

# ----推估函數參數----
def estimate_parameters(prediction, y_values):
    y = np.asarray(y_values, dtype=float)
    x = np.linspace(-2, 2, len(y))
    prediction = str(prediction).lower()
    if prediction == "linear":                  # Linear: y = ax + b
        a, b = np.polyfit(x, y, 1)
        return {"formula": f"y = {a:.4f}x + {b:.4f}",
                "parameters": {"a": a, "b": b}}
    elif prediction == "quadratic":             # Quadratic: y = ax² + bx + c
        a, b, c = np.polyfit(x, y, 2)
        return {"formula": (f"y = {a:.4f}x² {b:+.4f}x {c:+.4f}"),
                "parameters": {"a": a, "b": b, "c": c}}
    elif prediction == "cubic":                 # Cubic: y = ax³ + bx² + cx + d
        a, b, c, d = np.polyfit(x, y, 3)
        return {"formula": (f"y = {a:.4f}x³ {b:+.4f}x² {c:+.4f}x {d:+.4f}"),
                "parameters": {"a": a, "b": b, "c": c, "d": d}}
    else:
        return None

# ----分析使用者輸入----
def parse_input(text):
    tokens = re.split(r"[\s,;]+", text.strip())     # 支援空白、換行、逗號、分號等分隔方式。
    tokens = [token for token in tokens if token]
    values = []
    for token in tokens:
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(f"無法解析：{token}，請再試一次。") from exc
    return values

# ----UI介面----
st.title("📈數學函數辨識系統")      # 標題
st.write("V1 模型需要 100 個取樣值。輸入一組函數取樣值，讓 AI 判斷它最可能屬於哪一種數學函數")  # 背景與說明
st.info("請輸入至少 100 個樣本點。")        # 對於使用者的指令

# 所有函數類型清單
with st.expander("可辨識的 8 種函數類型"):
    for name in LABEL_NAMES.values():
        st.write(f"- {name}")

# 輸入文字
input_text = st.text_area("例如： 1.2 1.4 1.6 1.8 ...，也可以使用逗號分隔。", height=200,
                          placeholder=("請輸入數值"))

# 按鈕
col1, col2 = st.columns(2)
with col1:
    predict_button = st.button("🔍 確認", use_container_width=True) # 確認並開始預測
with col2:
    clear_button = st.button("🗑️ 清除", use_container_width=True)   # 清理輸入的資料

if clear_button:
    st.rerun()

if predict_button:
    try:
        y_values = parse_input(input_text)
        if len(y_values) != N_POINTS:
            st.error(f"目前 V1 模型需要剛好 {N_POINTS} 個數據，您現在輸入了 {len(y_values)} 個。")
            st.stop()

        # 顯示使用者輸入的曲線
        st.subheader("### 輸入資料的曲線")
        chart_df = pd.DataFrame({"y": y_values})
        st.line_chart(chart_df, use_container_width=True)

        # 進行預測
        model, feature_names = load_model()
        X_input = make_features(y_values, feature_names)
        prediction = model.predict(X_input)[0]
        prediction_text = LABEL_NAMES.get(str(prediction).lower(), str(prediction))
        st.success(f"🎯 AI 預測結果：**{prediction_text}**")

        # 推估函數參數
        parameter_result = estimate_parameters(prediction, y_values)
        if parameter_result is not None:
            st.subheader("- 推估函數:")
            st.code(parameter_result["formula"], language="text")
            st.subheader("- 參數:")
            for name, value in parameter_result["parameters"].items():
                st.write(f"**{name} = {value:.4f}**")

    except FileNotFoundError as exc:
        st.error(str(exc))
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"預測時發生錯誤：{exc}，請再試一次。")

