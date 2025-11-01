# 📄 src/app.py — Streamlit Stock Market Live Dashboard (Fixed + Final Enhanced)
import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
import numpy as np

# ----------------------------------------------------------
# 🌐 Streamlit Page Config
# ----------------------------------------------------------
st.set_page_config(page_title="📊 Stock Market Live Dashboard", layout="wide")

# ----------------------------------------------------------
# 📦 Load Models
# ----------------------------------------------------------
model_dir = "models"
log_model_path = os.path.join(model_dir, "logistic_model.pkl")
lin_model_path = os.path.join(model_dir, "linear_model.pkl")

try:
    log_model = joblib.load(log_model_path)
    lin_model = joblib.load(lin_model_path)
except Exception as e:
    st.error(f"❌ Model files missing or corrupted: {e}")
    st.stop()

# ----------------------------------------------------------
# 📁 Load Dataset (live → sample → full)
# ----------------------------------------------------------
data_dir = "data"
data_path = None
live_data_path = None

for file in os.listdir(data_dir):
    if file.endswith("_live.csv"):
        live_data_path = os.path.join(data_dir, file)
        st.sidebar.success(f"📡 Live dataset detected: {file}")
        break

sample_path = os.path.join(data_dir, "engineered_stock_data_sample.csv")
full_path = os.path.join(data_dir, "engineered_stock_data.csv")

if live_data_path:
    data_path = live_data_path
elif os.path.exists(sample_path):
    data_path = sample_path
    st.sidebar.info("📁 Using sample dataset for preview.")
elif os.path.exists(full_path):
    data_path = full_path
    st.sidebar.warning("⚠️ Using full dataset (large).")
else:
    st.error("❌ No dataset found in 'data/' folder.")
    st.stop()

# ----------------------------------------------------------
# 🧠 Read & Clean Data
# ----------------------------------------------------------
df = pd.read_csv(data_path)
df.columns = [col.lower().strip() for col in df.columns]

# --- Smart Feature Engineering (preserve non-zero data) ---
def safe_create_column(df, col, func):
    if col not in df.columns or df[col].isnull().all() or (df[col] == 0).all():
        df[col] = func()
    return df

# Compute technical indicators only if needed
safe_create_column(df, "ma5", lambda: df["close"].rolling(window=5).mean())
safe_create_column(df, "ma10", lambda: df["close"].rolling(window=10).mean())
safe_create_column(df, "ma20", lambda: df["close"].rolling(window=20).mean())
safe_create_column(df, "volatility", lambda: df["close"].pct_change().rolling(window=10).std())

# RSI
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

safe_create_column(df, "rsi", lambda: compute_rsi(df["close"]))
safe_create_column(df, "pct_change", lambda: df["close"].pct_change())

# Fill only small gaps
df = df.ffill().bfill()
recent_data = df.tail(50).reset_index(drop=True)

# ----------------------------------------------------------
# 🏷️ App Header
# ----------------------------------------------------------
st.title("📈 Stock Market Live Dashboard")
st.markdown("---")

# ----------------------------------------------------------
# 📋 Show Latest Data
# ----------------------------------------------------------
st.subheader("🕒 Recent Stock Data (Last 50 Records)")
show_cols = [c for c in ["date", "close", "ma5", "ma10", "ma20", "volatility", "rsi", "pct_change"] if c in df.columns]
st.dataframe(recent_data[show_cols].round(3), use_container_width=True)

# ----------------------------------------------------------
# 📊 Chart 1 — Close Prices
# ----------------------------------------------------------
if "close" in df.columns:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data["close"],
        mode="lines+markers",
        line=dict(color="#2196F3", width=2),
        name="Close Price"
    ))
    fig.update_layout(
        title="📉 Last 50 Close Prices",
        xaxis_title="Record Index",
        yaxis_title="Price",
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------
# 🧮 Sidebar — Prediction Inputs
# ----------------------------------------------------------
st.sidebar.header("🧮 Input Stock Data for Prediction")

input_fields = {
    "open": "Open Price",
    "high": "High Price",
    "low": "Low Price",
    "volume": "Volume",
    "ma5": "MA5",
    "ma10": "MA10",
    "ma20": "MA20",
    "volatility": "Volatility",
    "rsi": "RSI",
    "lag_close_1": "Previous Close",
    "lag_pct_1": "Previous % Change",
}

inputs = {k: st.sidebar.number_input(v, value=0.0) for k, v in input_fields.items()}
input_df = pd.DataFrame([inputs])

# ----------------------------------------------------------
# 🎯 Prediction Section
# ----------------------------------------------------------
st.subheader("🎯 Stock Movement & Price Prediction")

if st.button("🚀 Predict Now"):
    try:
        expected_features = list(lin_model.feature_names_in_) if hasattr(lin_model, "feature_names_in_") else list(input_df.columns)
        for feat in expected_features:
            if feat not in input_df.columns:
                input_df[feat] = 0.0
        input_df = input_df[expected_features]

        # Predict movement (UP/DOWN)
        movement = log_model.predict(input_df)[0]
        movement_label = "📈 UP" if movement == 1 else "📉 DOWN"
        movement_color = "green" if movement == 1 else "red"

        # Predict close price
        predicted_price = lin_model.predict(input_df)[0]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<h4 style='color:{movement_color};'>Stock Movement: {movement_label}</h4>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h4>💰 Predicted Close Price: <span style='color:#4CAF50;'>{predicted_price:.2f}</span></h4>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

st.markdown("---")

# ----------------------------------------------------------
# 📈 Chart 2 — Predicted vs Actual
# ----------------------------------------------------------
st.subheader("📊 Predicted vs Actual Closing Prices")

try:
    expected_features = getattr(lin_model, "feature_names_in_", [])
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0.0

    df_recent = df.dropna().tail(50)
    X_recent = df_recent[expected_features]
    y_actual = df_recent["close"]
    y_pred = lin_model.predict(X_recent)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        y=y_actual, mode="lines+markers", name="Actual Close", line=dict(color="blue", width=2)
    ))
    fig2.add_trace(go.Scatter(
        y=y_pred, mode="lines+markers", name="Predicted Close", line=dict(color="green", width=2)
    ))
    fig2.update_layout(
        title="📈 Predicted vs Actual (Last 50 Records)",
        xaxis_title="Record Index",
        yaxis_title="Close Price",
        template="plotly_white"
    )
    st.plotly_chart(fig2, use_container_width=True)

    mse = mean_squared_error(y_actual, y_pred)
    st.success(f"📉 Mean Squared Error (MSE): {mse:.4f}")

except Exception as e:
    st.error(f"❌ Error generating comparison chart: {e}")

# ----------------------------------------------------------
# 🦶 Footer
# ----------------------------------------------------------
st.markdown("---")
st.caption("✅ Live Dashboard Ready | Preserves Real Data | Auto Feature Alignment | Powered by Streamlit")
