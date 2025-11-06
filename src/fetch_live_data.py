import os
import time
import pandas as pd
from datetime import datetime
import random

# ✅ Directory setup
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(data_dir, exist_ok=True)

stock_name = "RELIANCE_NS"
file_path = os.path.join(data_dir, f"{stock_name}_live.csv")

print("📡 Live Data Fetcher Started...")
print(f"💾 Writing live data to: {file_path}")

# ✅ Initialize file if not exists
if not os.path.exists(file_path):
    df_init = pd.DataFrame(columns=["Datetime", "Close"])
    df_init.to_csv(file_path, index=False)

try:
    while True:
        # Simulate live stock price every 60 seconds (1 min interval)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        close_price = round(random.uniform(2500, 2800), 2)

        # Create new row
        df_new = pd.DataFrame({
            "Datetime": [now],
            "Close": [close_price]
        })

        # Append new data
        df_new.to_csv(file_path, mode='a', index=False, header=False)
        print(f"✅ New data added: {now} | Close: {close_price}")

        # Sleep for 60 seconds before fetching again
        time.sleep(60)

except KeyboardInterrupt:
    print("\n🛑 Live data fetcher stopped manually.")
except Exception as e:
    print(f"❌ Error: {e}")
