# 📄 src/model_trainer.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, mean_squared_error
import joblib

def train_models(data_path, output_dir):
    print("📥 Loading engineered dataset...")
    df = pd.read_csv(data_path)

    # ---------------- Clean data ----------------
    # Drop rows with missing target or close
    df.dropna(subset=["target", "close"], inplace=True)

    # Define features for Logistic Regression (exclude 'close')
    log_features = [
        "open", "high", "low", "volume",
        "MA5", "MA10", "MA20", "Volatility", "RSI",
        "lag_close_1", "lag_pct_1"
    ]

    # Replace infinities with NaN, then fill NaN with 0
    df[log_features] = df[log_features].replace([float('inf'), -float('inf')], pd.NA)
    df[log_features] = df[log_features].fillna(0)

    # ---------------- Logistic Regression ----------------
    print("\n⚙️ Training Logistic Regression model...")
    X_log = df[log_features]
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
    print(f"\n✅ Accuracy: {acc:.4f}")
    print(f"✅ Precision: {prec:.4f}")
    print(f"✅ Recall: {rec:.4f}")

    # ---------------- Linear Regression ----------------
    print("\n⚙️ Training Linear Regression model for price prediction...")

    lin_features = [
        "open", "high", "low", "volume",
        "MA5", "MA10", "MA20", "Volatility", "RSI",
        "lag_close_1", "lag_pct_1"
    ]
    X_lin = df[lin_features]
    y_lin = df["close"]

    # Clean Lin features too
    X_lin = X_lin.replace([float('inf'), -float('inf')], pd.NA).fillna(0)

    X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
        X_lin, y_lin, test_size=0.2, random_state=42
    )

    lin_model = LinearRegression()
    lin_model.fit(X_train_lin, y_train_lin)
    y_pred_lin = lin_model.predict(X_test_lin)

    mse = mean_squared_error(y_test_lin, y_pred_lin)
    print(f"\n💰 Linear Regression MSE: {mse:.6f}")  # realistic MSE

    # ---------------- Save models ----------------
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(log_model, os.path.join(output_dir, "logistic_model.pkl"))
    joblib.dump(lin_model, os.path.join(output_dir, "linear_model.pkl"))
    print(f"\n💾 Models saved to: {output_dir}")

    print("\n✅ Training completed successfully. Script terminated.\n")


if __name__ == "__main__":
    data_path = r"D:\Stock_Market_Analysis\data\engineered_stock_data.csv"
    output_dir = r"D:\Stock_Market_Analysis\models"
    train_models(data_path, output_dir)
