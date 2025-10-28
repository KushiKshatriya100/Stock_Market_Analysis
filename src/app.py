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
    st.error("❌ No dataset found. Please ensure a CSV exists in the 'data/' folder.")
    st.stop()

df = pd.read_csv(data_path)

st.title("📈 Stock Market Analysis Dashboard")

# ---------------- Display Recent Stock Data ----------------
st.subheader("Recent Stock Data (Last 50 Days)")
recent_data = df.tail(50)
cols_to_show = [c for c in ["symbol", "close", "MA5", "MA10", "Volatility", "RSI", "pct_change"] if c in recent_data.columns]
st.dataframe(recent_data[cols_to_show])

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("Input Stock Data for Prediction")

fields = {
    "open": "Open Price",
    "high": "High Price",
    "low": "Low Price",
    "volume": "Volume",
    "MA5": "MA5",
    "MA10": "MA10",
    "MA20": "MA20",
    "Volatility": "Volatility",
    "RSI": "RSI",
    "lag_close_1": "Previous Close",
    "lag_pct_1": "Previous % Change",
}

inputs = {k: st.sidebar.number_input(v, value=0.0) for k, v in fields.items()}
input_df = pd.DataFrame([inputs])

# ---------------- Prediction ----------------
if st.button("Predict"):
    try:
        movement = log_model.predict(input_df)[0]
        movement_label = "📈 UP" if movement == 1 else "📉 DOWN"
        st.subheader("Stock Movement Prediction:")
        st.success(movement_label)

        predicted_price = lin_model.predict(input_df)[0]
        st.subheader("Predicted Closing Price:")
        st.info(f"💰 {round(predicted_price, 2)}")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
