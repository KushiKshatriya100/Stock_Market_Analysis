# 📄 src/data_loader.py — Combine Multiple Stock CSVs (Safe for small sample CSV)
import os
import pandas as pd

def load_all_csv(data_dir):
    """
    Loads all stock CSV files from the given directory and combines them.
    If no CSVs found, fallback to engineered_stock_data_sample.csv.
    Adds a 'symbol' column (derived from filename).
    Saves the combined dataset as 'combined_data.csv'.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"⚠️ Data directory created: {data_dir}")

    # List all CSV files
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # Fallback: use small sample CSV
    sample_csv = "engineered_stock_data_sample.csv"
    if not all_files and os.path.exists(os.path.join(data_dir, sample_csv)):
        all_files = [sample_csv]

    if not all_files:
        raise FileNotFoundError("❌ No CSV files found in the data directory or sample CSV missing!")

    df_list = []
    for file in all_files:
        file_path = os.path.join(data_dir, file)
        try:
            temp = pd.read_csv(file_path)
            temp.columns = [c.lower().strip() for c in temp.columns]  # lowercase columns for consistency
            temp["symbol"] = file.replace("nse-", "").replace(".csv", "")
            df_list.append(temp)
            print(f"✅ Loaded {file} ({len(temp)} records)")
        except Exception as e:
            print(f"⚠️ Skipped {file}: {e}")

    combined = pd.concat(df_list, ignore_index=True)
    combined.dropna(inplace=True)

    output_path = os.path.join(data_dir, "combined_data.csv")
    combined.to_csv(output_path, index=False)

    print(f"\n📁 Combined dataset saved → {output_path}")
    print(f"📊 Total records combined: {len(combined)}")

    return combined


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    data = load_all_csv(data_dir)
    print(data.head())
