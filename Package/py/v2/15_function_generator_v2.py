# 15_function_generator_v2.py
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

x = np.linspace(-2, 2, 10)         # x 座標
rng = np.random.default_rng(42)
def generate_function(label, x, range):     # V2 函數產生器
    param_1 = np.nan
    param_2 = np.nan
    param_3 = np.nan
    param_4 = np.nan

    if label == "linear":                   # 1. Linear: y = ax + b
        a = range.uniform(-2, 2)
        b = range.uniform(-3, 3)
        param_1 = a
        param_2 = b
        y_true = a*x + b

    elif label == "quadratic":              # 2. Quadratic: y = ax^2 + bx + c
        a = range.uniform(-2, 2)
        if abs(a) < 0.1:
            a = 0.1 if a >= 0 else -0.1
        b = range.uniform(-2, 2)
        c = range.uniform(-3, 3)
        param_1 = a
        param_2 = b
        param_3 = c
        y_true = a*x**2 + b*x + c
        
    elif label == "cubic":                  # 3. Cubic: y = ax^3 + bx^2 + cx + d
        a = range.uniform(-2, 2)
        if abs(a) < 0.1:
            a = 0.1 if a >= 0 else -0.1
        b = range.uniform(-2, 2)
        c = range.uniform(-2, 2)
        d = range.uniform(-3, 3)
        param_1 = a
        param_2 = b
        param_3 = c
        param_4 = d
        y_true = a*x**3 + b*x**2 + c*x + d

    elif label == "exponential":            # 4. Exponential: y = a * exp(bx) + c
        a = range.uniform(-2.0, 2.0)
        if abs(a) < 0.1:
            a = 0.1 if a >= 0 else -0.1
        b = range.uniform(-3.0, 3.0)
        if abs(b) < 0.3:
            b = 0.3 if b >= 0 else -0.3
        c = range.uniform(-3, 3)
        param_1 = a
        param_2 = b
        param_3 = c
        y_true = a * np.exp(b*x) + c

    elif label == "logarithmic":            # 5. Logarithm: y = a * log(|x+b|) + c
        a = range.uniform(-2, 2)
        if abs(a) < 0.1:
            a = 0.1 if a >= 0 else -0.1
        b = range.uniform(2.1, 8.0)
        c = range.uniform(-3, 3)
        param_1 = a
        param_2 = b
        param_3 = c
        y_true = a * np.log(np.abs(x+b)) + c

    elif label == "sine":                   # 6. Sine: y = a * sin(bx + c) + d
        a = range.uniform(-3.0, 3.0)
        if abs(a) < 0.3:
            a = 0.3 if a >= 0 else -0.3
        b = range.uniform(0.1, 3.0)
        c = range.uniform(-np.pi/4, np.pi/4)
        d = range.uniform(-3, 3)
        param_1 = a
        param_2 = b
        param_3 = c
        param_4 = d
        y_true = a * np.sin(b*x+c) + d

    elif label == "cosine":                 # 7. Cosine: y = a * cos(bx + c) + d
        a = range.uniform(-3.0, 3.0)
        if abs(a) < 0.3:
            a = 0.3 if a >= 0 else -0.3
        b = range.uniform(0.1, 3.0)
        c = range.uniform(-np.pi/4, np.pi/4)
        d = range.uniform(-3, 3)
        param_1 = a
        param_2 = b
        param_3 = c
        param_4 = d
        y_true = a * np.cos(b*x+c) + d

    elif label == "reciprocal":             # 8. Reciprocal: y = a / (x + b) + c
        a = range.uniform(-3, 3)
        if abs(a) < 0.1:
            a = 0.1 if a >= 0 else -0.1
        b = range.uniform(2.5, 10.0)
        c = range.uniform(-2, 2)
        param_1 = a
        param_2 = b
        param_3 = c
        y_true = a/(x+b) + c
        
    else:
        raise ValueError(f"Unknown: {label}")

    # noise_sigma
    y_std = np.std(y_true)
    noise_ratio = range.uniform(0.02, 0.10)
    noise_sigma = y_std * noise_ratio
    noise_sigma = max(noise_sigma, 0.001)

    # 加入 Gaussian Noise
    noise = range.normal(loc=0, scale=noise_sigma, size=len(x))
    y = y_true + noise

    params = [param_1, param_2, param_3, param_4]
    return y, params, noise_sigma

labels = ["linear", "quadratic", "cubic", "exponential", 
          "logarithmic", "sine", "cosine", "reciprocal"]
dataset_rows = []
metadata_rows = []
sample_id = 1

for label in labels:
    for _ in range(500):
        y, params, noise_sigma = generate_function(label, x, rng)
        dataset_row = {"id": sample_id,
                       "label": label}
        
        for i in range(10):
            dataset_row[f"y_{i:02d}"] = y[i]
        dataset_row["noise"] = noise_sigma
        dataset_rows.append(dataset_row)
        metadata_row = {"id": sample_id,
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
dataset_df.to_csv(f'{PROJECT_DIR}/data/v2/function_dataset_v2.csv', index=False)
print("已建立 V2 數據資料\n")

metadata_df = pd.DataFrame(metadata_rows)
metadata_df.to_csv(f'{PROJECT_DIR}/data/v2/function_metadata_v2.csv', index=False)
print("已建立 V2 metadata資料")