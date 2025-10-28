# 📄 src/model_trainer.py

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    mean_squared_error,
)
import joblib


def train_models(data_path, output_dir):
    print("📥 Loading dataset...")
    df = pd.read_csv(data_path)

    # ---------------- Clean data ----------------
    if "target" not in df.columns or "close" not in df.columns:
        raise KeyError("❌ 'target' or 'close' column not found in dataset!")

    df.dropna(subset=["target", "close"], inplace=True)

    features = [
        "open", "high", "low", "volume",
        "MA5", "MA10", "MA20", "Volatility", "RSI",
        "lag_close_1", "lag_pct_1"
    ]

    # Ensure all features exist
    for col in features:
        if col not in df.columns:
            print(f"⚠️ Missing column '{col}', creating placeholder zeros.")
            df[col] = 0

    # Replace inf and NaN
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    # ---------------- Logistic Regression ----------------
    print("\n⚙️ Training Logistic Regression model (predicting UP/DOWN)...")
    X_log = df[features]
    y_log = df["target"].astype(int)

    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
        X_log, y_log, test_size=0.2, random_state=42, stratify=y_log
    )

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_log, y_train_log)
    y_pred_log = log_model.predict(X_test_log)

    # Metrics
    cm = confusion_matrix(y_test_log, y_pred_log)
    acc = accuracy_score(y_test_log, y_pred_log)
    prec = precision_score(y_test_log, y_pred_log, zero_division=0)
    rec = recall_score(y_test_log, y_pred_log, zero_division=0)

    print("\n📊 Confusion Matrix (Logistic Regression):")
    print(cm)
    print(f"✅ Accuracy:  {acc:.4f}")
    print(f"✅ Precision: {prec:.4f}")
    print(f"✅ Recall:    {rec:.4f}")

    # ---------------- Linear Regression ----------------
    print("\n⚙️ Training Linear Regression model (predicting Close Price)...")

    X_lin = df[features]
    y_lin = df["close"]

    X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
        X_lin, y_lin, test_size=0.2, random_state=42
    )

    lin_model = LinearRegression()
    lin_model.fit(X_train_lin, y_train_lin)
    y_pred_lin = lin_model.predict(X_test_lin)

    mse = mean_squared_error(y_test_lin, y_pred_lin)
    rmse = np.sqrt(mse)

    print(f"\n💰 Linear Regression MSE:  {mse:.6f}")
    print(f"💰 Linear Regression RMSE: {rmse:.6f}")

    # ---------------- Save models ----------------
    os.makedirs(output_dir, exist_ok=True)

    logistic_path = os.path.join(output_dir, "logistic_model.pkl")
    linear_path = os.path.join(output_dir, "linear_model.pkl")

    joblib.dump(log_model, logistic_path)
    joblib.dump(lin_model, linear_path)

    print("\n💾 Models saved successfully:")
    print(f"📁 Logistic Model: {logistic_path}")
    print(f"📁 Linear Model:   {linear_path}")

    print("\n✅ Training completed successfully!\n")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))

    data_dir = os.path.join(project_root, "data")
    model_dir = os.path.join(project_root, "models")

    sample_data_path = os.path.join(data_dir, "engineered_stock_data_sample.csv")
    full_data_path = os.path.join(data_dir, "engineered_stock_data.csv")

    if os.path.exists(sample_data_path):
        data_path = sample_data_path
        print("📁 Using lightweight sample dataset for faster training.")
    elif os.path.exists(full_data_path):
        data_path = full_data_path
        print("⚠️ Using full dataset (~50MB). This may take a while.")
    else:
        raise FileNotFoundError("❌ No dataset found in the 'data/' folder.")

    train_models(data_path, model_dir)
