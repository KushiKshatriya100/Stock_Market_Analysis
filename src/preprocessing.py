import pandas as pd
import os

def preprocess_stock_data(file_path):
    try:
        df = pd.read_csv(file_path)

        # Ensure consistent lowercase column names
        df.columns = [col.strip().lower() for col in df.columns]

        # Rename columns if needed
        rename_map = {
            'date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'  # 'total_trade_quantity' replaced with 'volume'
        }

        available_cols = [col for col in rename_map if col in df.columns]
        df = df[available_cols].rename(columns=rename_map)

        # Drop any missing values
        df.dropna(inplace=True)

        # Sort by date (ascending)
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(by='date', inplace=True)

        # Add percentage change
        df['pct_change'] = df['close'].pct_change()

        # Add simple label for model: 1 = price went up, 0 = down
        df['target'] = (df['pct_change'] > 0).astype(int)

        return df

    except Exception as e:
        print(f"⚠️ Skipped {os.path.basename(file_path)} due to error: {e}")
        return None


def combine_all_stocks(data_dir):
    combined_data = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            df = preprocess_stock_data(os.path.join(data_dir, file))
            if df is not None:
                df['symbol'] = file.replace('.csv', '')
                combined_data.append(df)

    if not combined_data:
        raise ValueError("No valid stock data files found to concatenate.")

    final_df = pd.concat(combined_data, ignore_index=True)
    return final_df


if __name__ == "__main__":
    data_dir = r"D:\Stock_Market_Analysis\data"
    combined_df = combine_all_stocks(data_dir)

    # Save combined dataset
    output_path = os.path.join(data_dir, "combined_stock_data.csv")
    combined_df.to_csv(output_path, index=False)

    print(f"✅ Combined dataset saved successfully to {output_path}")