# 📄 src/app.py

import streamlit as st
import pandas as pd
import joblib
import os

# ---------------- Load Models ----------------
log_model_path = os.path.join("models", "logistic_model.pkl")
lin_model_path = os.path.join("models", "linear_model.pkl")

try:
    log_model = joblib.load(log_model_path)
    lin_model = joblib.load(lin_model_path)
except FileNotFoundError:
    st.error("❌ Model files not found. Please ensure they exist in the 'models/' directory.")
    st.stop()

# ---------------- Load Data ----------------
sample_data_path = os.path.join("data", "engineered_stock_data_sample.csv")
full_data_path = os.path.join("data", "engineered_stock_data.csv")

if os.path.exists(sample_data_path):
    data_path = sample_data_path
    st.info("📁 Using lightweight sample dataset for performance.")
elif os.path.exists(full_data_path):
    data_path = full_data_path
    st.warning("⚠️ Using full dataset (large file).")
else:
    st.error("❌ No dataset found. Please ensure the CSV exists in the 'data/' folder.")
    st.stop()

df = pd.read_csv(data_path)

st.title("📈 Stock Market Analysis Dashboard")

# ---------------- Recent Stock Data ----------------
st.subheader("Recent Stock Data (Last 50 Days)")
recent_data = df.tail(50)
st.dataframe(
    recent_data[
        ["symbol", "close", "MA5", "MA10", "Volatility", "RSI", "pct_change"]
    ]
)

# ---------------- User Input ----------------
st.sidebar.header("Input Stock Data for Prediction")
open_price = st.sidebar.number_input("Open Price")
high_price = st.sidebar.number_input("High Price")
low_price = st.sidebar.number_input("Low Price")
volume = st.sidebar.number_input("Volume")
MA5 = st.sidebar.number_input("MA5")
MA10 = st.sidebar.number_input("MA10")
MA20 = st.sidebar.number_input("MA20")
Volatility = st.sidebar.number_input("Volatility")
RSI = st.sidebar.number_input("RSI")
lag_close_1 = st.sidebar.number_input("Previous Close")
lag_pct_1 = st.sidebar.number_input("Previous % Change")

input_df = pd.DataFrame(
    [
        [
            open_price,
            high_price,
            low_price,
            volume,
            MA5,
            MA10,
            MA20,
            Volatility,
            RSI,
            lag_close_1,
            lag_pct_1,
        ]
    ],
    columns=[
        "open",
        "high",
        "low",
        "volume",
        "MA5",
        "MA10",
        "MA20",
        "Volatility",
        "RSI",
        "lag_close_1",
        "lag_pct_1",
    ],
)

# ---------------- Prediction ----------------
if st.button("Predict"):
    try:
        # Logistic Regression → Next Day Movement
        movement = log_model.predict(input_df)[0]
        movement_label = "📈 UP" if movement == 1 else "📉 DOWN"
        st.subheader("Stock Movement Prediction:")
        st.write(movement_label)

        # Linear Regression → Predicted Close Price
        predicted_price = lin_model.predict(input_df)[0]
        st.subheader("Predicted Closing Price:")
        st.write(round(predicted_price, 2))
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
