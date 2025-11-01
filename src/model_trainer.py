# 📄 src/model_trainer.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib

def get_latest_data_file():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    live_files = [f for f in os.listdir(data_dir) if f.endswith("_live.csv")]

    if live_files:
        live_files = sorted(
            live_files,
            key=lambda f: os.path.getmtime(os.path.join(data_dir, f)),
            reverse=True
        )
        latest_file = live_files[0]
        print(f"📈 Using latest live dataset: {latest_file}")
        return os.path.join(data_dir, latest_file)
    else:
        print("⚠️ No live data found. Using default engineered dataset.")
        return os.path.join(data_dir, "engineered_stock_data.csv")


def find_close_column(df):
    """Try to identify the column that represents 'close' price."""
    for col in df.columns:
        if 'close' in col.lower():
            return col
    return None


def train_models():
    data_path = get_latest_data_file()
    df = pd.read_csv(data_path)
    print(f"✅ Data loaded from: {data_path}")
    print(f"🔢 Shape: {df.shape}")

    # Drop non-numeric columns like Ticker, Date, etc.
    df = df.select_dtypes(include=["number", "float64", "int64"])

    # Find close column dynamically
    close_col = find_close_column(df)
    if not close_col:
        raise ValueError("❌ Could not find any 'Close'-related column in dataset!")

    # ⚙️ If 'target' missing or has NaN, create it
    if "target" not in df.columns or df["target"].isna().any():
        print(f"⚠️ 'target' missing. Using '{close_col}' to generate it...")
        df["target"] = (df[close_col].shift(-1) > df[close_col]).astype(int)
        df.dropna(subset=["target"], inplace=True)

    # Separate features & target
    X = df.drop(columns=["target"])
    y = df["target"]

    # Handle missing values
    X = X.replace([float("inf"), -float("inf")], 0).fillna(0)
    y = y.fillna(0)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # ----- Linear Regression -----
    print("\n🚀 Training Linear Regression...")
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_lin)
    print(f"📊 Linear Regression MSE: {mse:.4f}")

    lin_path = os.path.join(os.path.dirname(__file__), "..", "models", "linear_model.pkl")
    joblib.dump(lin_reg, lin_path)
    print(f"💾 Saved linear model to: {lin_path}")

    # ----- Logistic Regression -----
    print("\n🚀 Training Logistic Regression...")
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    acc = accuracy_score(y_test, y_pred_log)
    print(f"📈 Logistic Regression Accuracy: {acc:.4f}")

    log_path = os.path.join(os.path.dirname(__file__), "..", "models", "logistic_model.pkl")
    joblib.dump(log_reg, log_path)
    print(f"💾 Saved logistic model to: {log_path}")

    print("\n✅ Model training complete!")


if __name__ == "__main__":
    train_models()
