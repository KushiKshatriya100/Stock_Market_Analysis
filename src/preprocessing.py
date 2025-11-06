import pandas as pd
import os

def preprocess_stock_data(file_path):
    """Clean and preprocess a single stock CSV file."""
    try:
        df = pd.read_csv(file_path)

        # 🧩 Normalize column names
        df.columns = [col.strip().lower() for col in df.columns]

        # ✅ Rename known columns safely
        rename_map = {
            'date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',  # 'total_trade_quantity' mapped to 'volume'
            'total_trade_quantity': 'volume',
            'totaltradedquantity': 'volume'
        }

        df.rename(columns=rename_map, inplace=True)

        # ✅ Filter only relevant columns
        keep_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        df = df[[col for col in keep_cols if col in df.columns]]

        if len(df.columns) < 5:
            raise ValueError(f"Missing key columns in {file_path}")

        # ✅ Convert numeric columns safely
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # ✅ Drop NaNs and sort
        df.dropna(inplace=True)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        df.sort_values(by='date', inplace=True)

        # ✅ Add derived features
        df['pct_change'] = df['close'].pct_change()
        df['target'] = (df['pct_change'] > 0).astype(int)

        # ✅ Reset index
        df.reset_index(drop=True, inplace=True)

        return df

    except Exception as e:
        print(f"⚠️ Skipped {os.path.basename(file_path)} due to error: {e}")
        return None


def combine_all_stocks(data_dir):
    """Combine all stock CSVs into one clean dataset."""
    combined_data = []

    # 🗂 Scan all CSVs except already processed ones
    all_files = [
        f for f in os.listdir(data_dir)
        if f.endswith(".csv") and not f.endswith("_live.csv") and "combined" not in f
    ]

    if not all_files:
        raise FileNotFoundError(f"No raw stock CSV files found in {data_dir}")

    for file in all_files:
        path = os.path.join(data_dir, file)
        df = preprocess_stock_data(path)
        if df is not None and not df.empty:
            df['symbol'] = file.replace('.csv', '').replace('NSE-', '').upper()
            combined_data.append(df)

    if not combined_data:
        raise ValueError("❌ No valid stock data available to combine.")

    # ✅ Combine all stock data
    final_df = pd.concat(combined_data, ignore_index=True)

    print(f"✅ Combined {len(all_files)} files successfully.")
    print(f"📊 Final Shape: {final_df.shape}")

    return final_df


if __name__ == "__main__":
    # ✅ Use dynamic base path so script works anywhere
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")

    # 🔄 Combine all preprocessed data
    combined_df = combine_all_stocks(data_dir)

    # 💾 Save final combined dataset
    output_path = os.path.join(data_dir, "combined_stock_data.csv")
    combined_df.to_csv(output_path, index=False)

    print(f"💾 Combined dataset saved successfully → {output_path}")
