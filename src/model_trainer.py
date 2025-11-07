# 📄 src/model_trainer.py — Balanced Random Forest Trainer
"""
Improved Model Trainer — Balanced Random Forest
✅ Balances target classes to avoid always predicting 'DOWN'.
✅ Handles both live and engineered datasets.
✅ Supports small CSV for Streamlit cloud deployment.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import joblib


# ---------- Helpers ----------
def get_latest_data_file():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    # Priority: Live data > Test data > Engineered full data
    live_files = [f for f in os.listdir(data_dir) if f.endswith("_live.csv")]
    test_files = [f for f in os.listdir(data_dir) if f.endswith("_test.csv")]

    small_file = os.path.join(data_dir, "engineered_stock_data_sample.csv")
    full_file = os.path.join(data_dir, "engineered_stock_data.csv")

    # Streamlit cloud friendly
    if os.getenv("STREAMLIT_SERVER") is not None and os.path.exists(small_file):
        print(f"📦 Using small sample CSV: {small_file}")
        return small_file

    if live_files:
        live_files = sorted(
            live_files,
            key=lambda f: os.path.getmtime(os.path.join(data_dir, f)),
            reverse=True
        )
        latest_file = live_files[0]
        print(f"📈 Using latest live dataset: {latest_file}")
        return os.path.join(data_dir, latest_file)

    if test_files:
        latest_file = sorted(
            test_files,
            key=lambda f: os.path.getmtime(os.path.join(data_dir, f)),
            reverse=True
        )[0]
        print(f"🧪 Using latest test dataset: {latest_file}")
        return os.path.join(data_dir, latest_file)

    # Default fallback
    if os.path.exists(full_file):
        print(f"⚠️ Using full engineered dataset: {full_file}")
        return full_file

    raise FileNotFoundError("❌ No dataset found to train the model!")


def find_close_column(df):
    for col in df.columns:
        if 'close' in col.lower():
            return col
    return None


# ---------- Main Training ----------
def train_random_forest():
    data_path = get_latest_data_file()
    df = pd.read_csv(data_path)
    print(f"✅ Data loaded from: {data_path}")
    print(f"🔢 Shape: {df.shape}")

    # Ensure numeric data only
    df = df.select_dtypes(include=["number", "float64", "int64"])

    # Find close column
    close_col = find_close_column(df)
    if not close_col:
        raise ValueError("❌ Could not find any 'close'-related column in dataset!")

    # Generate target (UP=1, DOWN=0)
    if "target" not in df.columns or df["target"].isna().any():
        print(f"⚙️ Generating target column using '{close_col}'...")
        df["target"] = (df[close_col].shift(-1) > df[close_col]).astype(int)
        df.dropna(subset=["target"], inplace=True)

    # Balance dataset
    print("⚖️ Balancing dataset to avoid bias...")
    df_majority = df[df["target"] == 0]
    df_minority = df[df["target"] == 1]

    if len(df_minority) == 0:
        raise ValueError("❌ No upward movements found — cannot train balanced model.")

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )

    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    print(df_balanced["target"].value_counts())

    # Split features/target
    X = df_balanced.drop(columns=["target"])
    y = df_balanced["target"]

    X = X.replace([float("inf"), -float("inf")], 0).fillna(0)
    y = y.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print("\n🚀 Training Balanced Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Evaluate
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"📊 Random Forest Accuracy: {acc:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "random_forest_model.pkl")
    joblib.dump(rf_model, model_path)
    print(f"💾 Model saved successfully to: {model_path}")

    print("\n✅ Model training complete and balanced!")


if __name__ == "__main__":
    train_random_forest()
