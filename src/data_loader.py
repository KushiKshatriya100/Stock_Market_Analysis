import os
import pandas as pd

def load_all_csv(data_dir):
    """Loads all Nifty50 CSV files and combines them."""
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    df_list = []

    for file in all_files:
        temp = pd.read_csv(os.path.join(data_dir, file))
        temp['Symbol'] = file.replace('NSE-', '').replace('.csv', '')
        df_list.append(temp)

    combined = pd.concat(df_list, ignore_index=True)
    combined.dropna(inplace=True)
    combined.to_csv(os.path.join(data_dir, 'combined_data.csv'), index=False)
    return combined

if __name__ == "__main__":
    data = load_all_csv("../data")
    print(data.head())
