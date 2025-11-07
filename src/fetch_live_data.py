# 📄 src/fetch_live_data.py — Simulated Live Stock Data Fetcher
import os
import time
import pandas as pd
from datetime import datetime
import random

# ---------- Configuration ----------
STOCK_NAME = "RELIANCE_NS"
INTERVAL_SECONDS = 60  # Fetch every 60 seconds
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
FILE_PATH = os.path.join(DATA_DIR, f"{STOCK_NAME}_live.csv")

print("📡 Live Data Fetcher Started...")
print(f"💾 Writing live data to: {FILE_PATH}")

# ---------- Initialize file if not exists ----------
if not os.path.exists(FILE_PATH):
    df_init = pd.DataFrame(columns=["datetime", "close"])
    df_init.to_csv(FILE_PATH, index=False)

# ---------- Main Loop ----------
try:
    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        close_price = round(random.uniform(2500, 2800), 2)  # Simulate live price

        df_new = pd.DataFrame({
            "datetime": [now],
            "close": [close_price]
        })

        # Append new row safely
        df_new.to_csv(FILE_PATH, mode='a', index=False, header=False)
        print(f"✅ New data added: {now} | Close: {close_price}")

        time.sleep(INTERVAL_SECONDS)

except KeyboardInterrupt:
    print("\n🛑 Live data fetcher stopped manually.")
except Exception as e:
    print(f"❌ Error: {e}")
