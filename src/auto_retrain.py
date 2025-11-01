import os
import time
import joblib
import hashlib
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from plyer import notification

# ---------- Helper: Get MD5 hash of file ----------
def get_file_hash(file_path):
    """Return md5 hash of the file contents"""
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

# ---------- Function to get latest live data file ----------
def get_latest_data_file():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    live_files = [f for f in os.listdir(data_dir) if f.endswith("_live.csv")]
    if not live_files:
        return None
    live_files = sorted(
        live_files,
        key=lambda f: os.path.getmtime(os.path.join(data_dir, f)),
        reverse=True
    )
    return os.path.join(data_dir, live_files[0])

# ---------- Core Model Training Function ----------
def train_models(data_path):
    df = pd.read_csv(data_path)
    df = df.select_dtypes(include=["number"])

    # Handle missing or invalid target
    if "target" not in df.columns or df["target"].isna().any():
        print("⚠️ 'target' missing. Using 'lag_close_1' to generate it...")
        if "lag_close_1" in df.columns and "Close" in df.columns:
            df["target"] = (df["Close"] > df["lag_close_1"]).astype(int)
        else:
            df["target"] = 0

    X = df.drop(columns=["target"])
    y = df["target"].fillna(0)
    X = X.replace([float("inf"), -float("inf")], 0).fillna(0)

    # Avoid single-class errors
    unique_classes = y.unique()
    skip_log_reg = False
    if len(unique_classes) < 2:
        print(f"⚠️ Only one class ({unique_classes[0]}) found in 'target'. Skipping logistic regression.")
        skip_log_reg = True

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # Train Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    mse = mean_squared_error(y_test, lin_reg.predict(X_test))

    # Save models
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(lin_reg, os.path.join(models_dir, "linear_model.pkl"))

    acc = 0.0
    if not skip_log_reg:
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(X_train, y_train)
        acc = accuracy_score(y_test, log_reg.predict(X_test))
        joblib.dump(log_reg, os.path.join(models_dir, "logistic_model.pkl"))

    # Summary
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"✅ Retrained at {timestamp}")
    print(f"📊 Linear Regression MSE: {mse:.4f}")
    print(f"📈 Logistic Regression Accuracy: {acc:.4f}")

    # Logging
    with open(os.path.join(models_dir, "retrain_log.txt"), "a") as log_file:
        log_file.write(f"[{timestamp}] MSE={mse:.4f}, ACC={acc:.4f}\n")

    # Desktop Notification
    try:
        notification.notify(
            title="✅ Stock Model Retrained",
            message=f"Retrained successfully at {timestamp}\nMSE={mse:.4f}, ACC={acc:.4f}",
            timeout=5
        )
    except Exception:
        pass


# ---------- Smart Watcher Logic ----------
if __name__ == "__main__":
    print("🔁 Smart Auto-Retrain Watcher Started...")

    total_minutes = 1            # ⏱ run for 1 minute only
    interval_seconds = 30        # check every 30 seconds
    end_time = time.time() + (total_minutes * 60)

    latest_file = get_latest_data_file()
    last_hash = None

    while time.time() < end_time:
        if latest_file and os.path.exists(latest_file):
            current_hash = get_file_hash(latest_file)

            # Compare hashes to detect actual data change
            if last_hash != current_hash:
                print(f"📈 Data changed in {os.path.basename(latest_file)} — retraining...")
                train_models(latest_file)
                last_hash = current_hash
            else:
                print(f"⏳ No new data content change... checking again in {interval_seconds}s")
        else:
            print("⚠️ No live dataset found.")
            latest_file = get_latest_data_file()

        time.sleep(interval_seconds)

    print("🕒 Auto-retraining stopped (1-minute runtime limit reached).")
