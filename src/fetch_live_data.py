import os
import time
import pandas as pd
from datetime import datetime
import random

# Directory for live data
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(data_dir, exist_ok=True)

stock_name = "RELIANCE_NS"

print("📡 Live Data Fetcher Started...")

while True:
    # Simulate new stock price every 30 seconds
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    close_price = round(random.uniform(2500, 2800), 2)
    
    # Create a dataframe
    df = pd.DataFrame({
        "Datetime": [now],
        "Close": [close_price]
    })
    
    file_path = os.path.join(data_dir, f"{stock_name}_live.csv")
    
    # Append or create
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', index=False, header=False)
    else:
        df.to_csv(file_path, index=False)
    
    print(f"💾 Updated {stock_name}_live.csv at {now}")
    time.sleep(30)
