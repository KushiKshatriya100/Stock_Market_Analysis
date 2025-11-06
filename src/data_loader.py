# 📄 src/data_loader.py — Combine Multiple Stock CSVs
import os
import pandas as pd

def load_all_csv(data_dir):
    """
    Loads all stock CSV files from the given directory and combines them.
    Adds a 'Symbol' column (derived from filename).
    Saves the combined dataset as 'combined_data.csv'.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Data directory not found: {data_dir}")

    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not all_files:
        raise FileNotFoundError("⚠️ No CSV files found in the data directory.")

    df_list = []
    for file in all_files:
        file_path = os.path.join(data_dir, file)
        try:
            temp = pd.read_csv(file_path)
            temp["Symbol"] = file.replace("NSE-", "").replace(".csv", "")
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
    data = load_all_csv("../data")
    print(data.head())
