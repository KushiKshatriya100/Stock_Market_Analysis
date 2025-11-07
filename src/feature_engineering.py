# 📄 src/feature_engineering.py — Compute Technical Features Safely
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# ---------- RSI Computation ----------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

# ---------- Feature Engineering Function ----------
def feature_engineering(input_path, output_path):
    print("🔧 Loading combined dataset...")
    df = pd.read_csv(input_path, parse_dates=["date"])
    df.columns = [c.lower().strip() for c in df.columns]  # lowercase columns
    df.sort_values(by=["symbol", "date"], inplace=True)

    # Compute pct_change if not exists
    if "pct_change" not in df.columns:
        df["pct_change"] = df["close"].pct_change()

    feature_frames = []

    for symbol, group in df.groupby("symbol"):
        group = group.copy()

        # Safely compute moving averages and rolling stats
        group["ma5"] = group["close"].rolling(window=5, min_periods=1).mean()
        group["ma10"] = group["close"].rolling(window=10, min_periods=1).mean()
        group["ma20"] = group["close"].rolling(window=20, min_periods=1).mean()
        group["volatility"] = group["close"].pct_change().rolling(window=10, min_periods=1).std()

        # RSI
        group["rsi"] = compute_rsi(group["close"])

        # Lag features
        group["lag_close_1"] = group["close"].shift(1)
        group["lag_pct_1"] = group["pct_change"].shift(1)

        group.dropna(inplace=True)
        feature_frames.append(group)

    final_df = pd.concat(feature_frames, ignore_index=True)

    # Normalize selected numeric columns
    scaler = MinMaxScaler()
    cols_to_scale = [
        "open", "high", "low", "close", "volume",
        "ma5", "ma10", "ma20", "volatility", "rsi"
    ]
    # Scale only if column exists
    cols_to_scale = [c for c in cols_to_scale if c in final_df.columns]
    final_df[cols_to_scale] = scaler.fit_transform(final_df[cols_to_scale])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"✅ Feature engineering complete! Saved to: {output_path}")

    return final_df

# ---------- Main ----------
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "..", "data", "combined_data.csv")  # Updated to match data_loader.py
    output_path = os.path.join(base_dir, "..", "data", "engineered_stock_data.csv")
    feature_engineering(input_path, output_path)
