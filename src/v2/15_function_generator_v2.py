# 15_function_generator_v2.py
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
x = np.linspace(-2, 2, 10)         # x 座標

# 函數產生器
def generate_function(label, x, rng):       
    param_1 = np.nan
    param_2 = np.nan
    param_3 = np.nan
    param_4 = np.nan
    if label == "linear":                   # Linear: y = ax + b
        a = rng.uniform(-2, 2)
        b = rng.uniform(-5, 5)
        param_1 = a
        param_2 = b
        y_true = a*x + b
    elif label == "quadratic":              # Quadratic: y = ax^2 + bx + c
        a = rng.uniform(-2, 2)
        if abs(a) < 0.1:
            a = 0.1 if a >= 0 else -0.1
        b = rng.uniform(-2, 2)
        c = rng.uniform(-3, 3)
        param_1 = a
        param_2 = b
        param_3 = c
        y_true = a*x**2 + b*x + c
    elif label == "cubic":                  # Cubic: y = ax^3 + bx^2 + cx + d
        a = rng.uniform(-2, 2)
        if abs(a) < 0.1:
            a = 0.1 if a >= 0 else -0.1
        b = rng.uniform(-2, 2)
        c = rng.uniform(-2, 2)
        d = rng.uniform(-3, 3)
        param_1 = a
        param_2 = b
        param_3 = c
        param_4 = d
        y_true = a*x**3 + b*x**2 + c*x + d
    elif label == "exponential":            # Exponential: y = a * exp(bx) + cx + d
        a = rng.uniform(-2.0, 2.0)
        if abs(a) < 0.3:
            a = 0.3 if a >= 0 else -0.3
        b = rng.uniform(-1.5, 1.5)
        if abs(b) < 0.1:
            b = 0.1 if b >= 0 else -0.1
        c = rng.uniform(-1, 1)
        d = rng.uniform(-2, 2)
        param_1 = a
        param_2 = b
        param_3 = c
        param_4 = d
        y_true = a * np.exp(b*x) + c*x + d
    elif label == "logarithmic":            # Logarithmic: y = a * log(|x+b|) + c
        a = rng.uniform(-2, 2)
        if abs(a) < 0.1:
            a = 0.1 if a >= 0 else -0.1
        b = rng.uniform(2.25, 10.0)
        c = rng.uniform(-2, 2)
        param_1 = a
        param_2 = b
        param_3 = c
        y_true = a * np.log(np.abs(x+b)) + c
    elif label == "sine":                   # Sine: y = a * sin(bx + c) + d
        a = rng.uniform(-2.0, 2.0)
        if abs(a) < 0.3:
            a = 0.3 if a >= 0 else -0.3
        b = rng.uniform(1.0, 4.0)
        c = rng.uniform(-np.pi/4, np.pi/4)
        d = rng.uniform(-2, 2)
        param_1 = a
        param_2 = b
        param_3 = c
        param_4 = d
        y_true = a * np.sin(b*x+c) + d
    elif label == "cosine":                 # Cosine: y = a * cos(bx + c) + d
        a = rng.uniform(-2.0, 2.0)
        if abs(a) < 0.3:
            a = 0.3 if a >= 0 else -0.3
        b = rng.uniform(1.0, 4.0)
        c = rng.uniform(-np.pi/4, np.pi/4)
        d = rng.uniform(-2, 2)
        param_1 = a
        param_2 = b
        param_3 = c
        param_4 = d
        y_true = a * np.cos(b*x+c) + d
    elif label == "reciprocal":             # Reciprocal: y = a / (x + b) + c
        a = rng.uniform(-3, 3)
        if abs(a) < 0.1:
            a = 0.1 if a >= 0 else -0.1
        b = rng.uniform(2.5, 4.0)
        c = rng.uniform(-2, 2)
        param_1 = a
        param_2 = b
        param_3 = c
        y_true = a/(x+b) + c
    else:
        raise ValueError(f"未知的label: {label}")

    # noise_sigma
    y_std = np.std(y_true)
    noise_ratio = rng.uniform(0.02, 0.10)
    noise_sigma = y_std * noise_ratio
    noise_sigma = max(noise_sigma, 0.001)

    # 加入 Gaussian Noise
    noise = rng.normal(loc=0, scale=noise_sigma, size=len(x))
    y = y_true + noise

    params = [param_1, param_2, param_3, param_4]
    return y, params, noise_sigma

labels = ["linear", "quadratic", "cubic", "exponential", "logarithmic", "sine", "cosine", "reciprocal"]
dataset_rows = []
metadata_rows = []
sample_id = 0
for label in labels:
    for _ in range(500):
        y, params, noise_sigma = generate_function(label, x, rng)
        dataset_row = {"sample_id": sample_id,
                       "label": label}
        for i in range(10):
            dataset_row[f"y_{i}"] = y[i]
        dataset_row["noise_sigma"] = noise_sigma
        dataset_rows.append(dataset_row)

        metadata_row = {"sample_id": sample_id,
                        "label": label,
                        "param_1": params[0],
                        "param_2": params[1],
                        "param_3": params[2],
                        "param_4": params[3],
                        "noise_sigma": noise_sigma}
        metadata_rows.append(metadata_row)

        sample_id += 1

# 建立DataFrame
dataset_df = pd.DataFrame(dataset_rows)
metadata_df = pd.DataFrame(metadata_rows)

# 儲存檔案
dataset_df.to_csv('data/v2/function_dataset_v2.csv', index=False)
metadata_df.to_csv('data/v2/function_metadata_v2.csv', index=False)
print("已建立V2數據資料！")