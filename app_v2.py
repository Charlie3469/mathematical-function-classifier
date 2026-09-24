# app_v2.py
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from scipy.optimize import curve_fit

# 基本設定
N_POINTS = 10
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "v2" / "final_model_v2.joblib"

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
        raise FileNotFoundError(f"找不到模型檔案:\n{MODEL_PATH}\n\n請確認 final_model_v2.joblib 是否存在。")
    package = joblib.load(MODEL_PATH)

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
    x = np.linspace(-2, 2, N_POINTS)
    raw_features = {f"y_{i}": y[i] for i in range(N_POINTS)}

    # 數學特徵
    slope = np.polyfit(x, y, 1)[0]
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
        return {"formula": f"y = {a:.2f}x + {b:.2f}",
                "parameters": {"a": a, "b": b}}
    
    elif prediction == "quadratic":             # Quadratic: y = ax² + bx + c
        a, b, c = np.polyfit(x, y, 2)
        return {"formula": (f"y = {a:.2f}x² {b:+.2f}x {c:+.2f}"),
                "parameters": {"a": a, "b": b, "c": c}}
    
    elif prediction == "cubic":                 # Cubic: y = ax³ + bx² + cx + d
        a, b, c, d = np.polyfit(x, y, 3)
        return {"formula": (f"y = {a:.2f}x³ {b:+.2f}x² {c:+.2f}x {d:+.2f}"),
                "parameters": {"a": a, "b": b, "c": c, "d": d}}
    
    elif prediction == "exponential":           # Exponential: y = a * e^(bx) + cx + d
        def func(x, a, b, c, d):
            return a * np.exp(b*x) + c*x + d
        params, _ = curve_fit(func, x.astype(np.float64), y.astype(np.float64),
                              p0=[1.0, 0.5, 0.0, 0.0], method="trf", maxfev=10000)
        a, b, c, d = params
        return {"formula": (f"y = {a:.2f}e^({b:.2f}x) {c:+.2f}x {d:+.2f}"),
                "parameters": {"a": a, "b": b, "c": c, "d": d}}
    
    elif prediction == "logarithmic":           # Logarithmic: y = a * log(|x+b|) + c
        def func(x, a, b, c):
            return a * np.log(np.abs(x+b)) + c
        params, _ = curve_fit(func, x.astype(np.float64), y.astype(np.float64),
                              p0=[1.0, 0.5, 0.0], method="trf", maxfev=10000)
        a, b, c = params
        return {"formula": (f"y = {a:.2f}*log(|x {b:+.2f}|) {c:+.2f}"),
                "parameters": {"a": a, "b": b, "c": c}}
    
    elif prediction == "sine":                  # Sine: y = a * sin(bx+c)+d
        def func(x, a, b, c, d):
            return a * np.sin(b*x + c) + d
        params, _ = curve_fit(func, x, y, p0=[1.5, 2.0, 0.0, 0.0], maxfev=10000)
        a, b, c, d = params
        return {"formula": (f"y = {a:.2f}*sin({b:.2f}x {c:+.2f}) {d:+.2f}"),
                "parameters": {"a": a, "b": b, "c": c, "d": d}}
    
    elif prediction == "cosine":                # Cosine: y = a * cos(bx+c)+d
        def func(x, a, b, c, d):
            return a * np.cos(b*x + c) + d
        params, _ = curve_fit(func, x, y, p0=[1.5, 2.0, 0.0, 0.0], maxfev=10000)
        a, b, c, d = params
        return {"formula": (f"y = {a:.2f}*cos({b:.2f}x {c:+.2f}) {d:+.2f}"),
                "parameters": {"a": a, "b": b, "c": c, "d": d}}
    
    elif prediction == "reciprocal":            # Reciprocal: y = a/(x+b)+c
        def func(x, a, b, c):
            return a / (x + b) + c
        params, _ = curve_fit(func, x, y, p0=[1.0, 3.0, 0.0], maxfev=10000)
        a, b, c = params
        return {"formula": (f"y = {a:.2f}/(x {b:+.2f}) {c:+.2f}"),
                "parameters": {"a": a, "b": b, "c": c}}
    
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
st.title("📈數學函數辨識系統V2")      # 標題
st.write("V2 模型需要 10 個取樣值。輸入一組函數取樣值，讓 AI 判斷它最可能屬於哪一種數學函數")  # 背景與說明
st.info("請輸入 10 個樣本點。")        # 對於使用者的指令

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
    predict_button = st.button("🔍 確認", width="stretch")  # 確認並開始預測
with col2:
    clear_button = st.button("🗑️ 清除", width="stretch")    # 清理輸入的資料

if clear_button:
    st.rerun()

if predict_button:
    try:
        y_values = parse_input(input_text)
        x = np.linspace(-2, 2, N_POINTS)
        if len(y_values) != N_POINTS:
            st.error(f" V2 模型需要 {N_POINTS} 個數據，您輸入了 {len(y_values)} 個。")
            st.stop()

        # 顯示使用者輸入的曲線
        st.subheader("### 輸入資料的曲線")
        chart_df = pd.DataFrame({"x": x, "y": y_values})
        st.line_chart(chart_df, x="x", y="y", width="stretch")

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
                st.write(f"**{name} = {value:.2f}**")

    except FileNotFoundError as exc:
        st.error(str(exc))
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"預測時發生錯誤：{exc}，請再試一次。")

