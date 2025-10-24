import pandas as pd
import numpy as np
import os

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def feature_engineering(input_path, output_path):
    print("🔧 Loading combined dataset...")
    df = pd.read_csv(input_path, parse_dates=["date"])
    df.sort_values(by=["symbol", "date"], inplace=True)

    # Group by symbol for independent feature generation
    feature_frames = []
    for symbol, group in df.groupby("symbol"):
        group = group.copy()

        # Moving Averages
        group["MA5"] = group["close"].rolling(window=5).mean()
        group["MA10"] = group["close"].rolling(window=10).mean()
        group["MA20"] = group["close"].rolling(window=20).mean()

        # Volatility (Rolling Std Dev of Close)
        group["Volatility"] = group["close"].rolling(window=10).std()

        # RSI
        group["RSI"] = compute_rsi(group["close"])

        # Lag features
        group["lag_close_1"] = group["close"].shift(1)
        group["lag_pct_1"] = group["pct_change"].shift(1)

        # Drop NaNs from rolling calculations
        group.dropna(inplace=True)

        feature_frames.append(group)

    final_df = pd.concat(feature_frames, ignore_index=True)

    # Normalize selected numeric columns
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    cols_to_scale = ["open", "high", "low", "close", "volume", "MA5", "MA10", "MA20", "Volatility", "RSI"]
    final_df[cols_to_scale] = scaler.fit_transform(final_df[cols_to_scale])

    # Save final dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"✅ Feature engineering complete! Saved to: {output_path}")

    return final_df


if __name__ == "__main__":
    input_path = r"D:\Stock_Market_Analysis\data\combined_stock_data.csv"
    output_path = r"D:\Stock_Market_Analysis\data\engineered_stock_data.csv"
    feature_engineering(input_path, output_path)
