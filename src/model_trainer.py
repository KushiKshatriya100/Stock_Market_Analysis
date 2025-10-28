# 📄 src/model_trainer.py

import pandas as pd
import os
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
    print("📥 Loading engineered dataset...")
    df = pd.read_csv(data_path)

    # ---------------- Clean data ----------------
    df.dropna(subset=["target", "close"], inplace=True)

    features = [
        "open", "high", "low", "volume",
        "MA5", "MA10", "MA20", "Volatility", "RSI",
        "lag_close_1", "lag_pct_1"
    ]

    df[features] = df[features].replace([float('inf'), -float('inf')], pd.NA).fillna(0)

    # ---------------- Logistic Regression ----------------
    print("\n⚙️ Training Logistic Regression model (for UP/DOWN movement)...")
    X_log = df[features]
    y_log = df["target"]

    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
        X_log, y_log, test_size=0.2, random_state=42, stratify=y_log
    )

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_log, y_train_log)
    y_pred_log = log_model.predict(X_test_log)

    # Metrics
    cm = confusion_matrix(y_test_log, y_pred_log)
    acc = accuracy_score(y_test_log, y_pred_log)
    prec = precision_score(y_test_log, y_pred_log)
    rec = recall_score(y_test_log, y_pred_log)

    print("\n📊 Confusion Matrix (Logistic Regression):")
    print(cm)
    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ Precision: {prec:.4f}")
    print(f"✅ Recall: {rec:.4f}")

    # ---------------- Linear Regression ----------------
    print("\n⚙️ Training Linear Regression model (for price prediction)...")

    X_lin = df[features]
    y_lin = df["close"]

    X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
        X_lin, y_lin, test_size=0.2, random_state=42
    )

    lin_model = LinearRegression()
    lin_model.fit(X_train_lin, y_train_lin)
    y_pred_lin = lin_model.predict(X_test_lin)

    mse = mean_squared_error(y_test_lin, y_pred_lin)
    print(f"\n💰 Linear Regression MSE: {mse:.6f}")

    # ---------------- Save models ----------------
    os.makedirs(output_dir, exist_ok=True)

    logistic_path = os.path.join(output_dir, "logistic_model.pkl")
    linear_path = os.path.join(output_dir, "linear_model.pkl")

    joblib.dump(log_model, logistic_path)
    joblib.dump(lin_model, linear_path)

    print(f"\n💾 Models saved successfully to: {output_dir}")
    print(f"📁 Logistic Model: {logistic_path}")
    print(f"📁 Linear Model:   {linear_path}")

    print("\n✅ Training completed successfully.\n")


if __name__ == "__main__":
    # Get project root (one level up from src/)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))

    data_dir = os.path.join(project_root, "data")
    model_dir = os.path.join(project_root, "models")

    # Prefer sample dataset if exists
    sample_data_path = os.path.join(data_dir, "engineered_stock_data_sample.csv")
    full_data_path = os.path.join(data_dir, "engineered_stock_data.csv")

    if os.path.exists(sample_data_path):
        data_path = sample_data_path
        print("📁 Using lightweight sample dataset for faster training.")
    elif os.path.exists(full_data_path):
        data_path = full_data_path
        print("⚠️ Using full dataset (large file). May take longer.")
    else:
        raise FileNotFoundError("❌ No dataset found in the 'data/' folder.")

    train_models(data_path, model_dir)
