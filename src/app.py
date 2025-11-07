# 📄 src/app.py — Streamlit Stock Market Live Dashboard (Random Forest Integrated + Mode Switch)
import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go

# ----------------------------------------------------------
# 🌐 Streamlit Page Config
# ----------------------------------------------------------
st.set_page_config(page_title="📊 Stock Market Live Dashboard", layout="wide")

# ----------------------------------------------------------
# 📦 Load Random Forest Model
# ----------------------------------------------------------
model_dir = "models"
rf_model_path = os.path.join(model_dir, "random_forest_model.pkl")

try:
    rf_model = joblib.load(rf_model_path)
    st.sidebar.success("✅ Random Forest Model Loaded Successfully")
except Exception as e:
    st.error(f"❌ Random Forest model not found: {e}")
    st.stop()

# ----------------------------------------------------------
# 🧭 Dataset Mode Selection
# ----------------------------------------------------------
st.sidebar.header("📂 Dataset Mode")
mode = st.sidebar.radio(
    "Select dataset to view:",
    ["Live Data", "Test Data", "Full Historical"],
    index=0
)

data_dir = "data"
data_path = None

# Detect files automatically
live_file = next((f for f in os.listdir(data_dir) if f.endswith("_live.csv")), None)
test_file = next((f for f in os.listdir(data_dir) if f.endswith("_test.csv")), None)
full_file = "engineered_stock_data.csv"

# Streamlit Cloud friendly: use small CSV if on cloud
if os.getenv("STREAMLIT_SERVER") is not None:
    small_file = "combined_stock_data_small.csv"
    if os.path.exists(os.path.join(data_dir, small_file)):
        full_file = small_file

if mode == "Live Data" and live_file:
    data_path = os.path.join(data_dir, live_file)
    st.sidebar.success(f"📡 Using Live dataset: {live_file}")
elif mode == "Test Data" and test_file:
    data_path = os.path.join(data_dir, test_file)
    st.sidebar.info(f"🧪 Using Test dataset: {test_file}")
elif os.path.exists(os.path.join(data_dir, full_file)):
    data_path = os.path.join(data_dir, full_file)
    st.sidebar.warning(f"⚠️ Using full historical dataset: {full_file}")
else:
    st.error("❌ No valid dataset found in 'data/' folder.")
    st.stop()

# ----------------------------------------------------------
# 🧠 Load Data
# ----------------------------------------------------------
df = pd.read_csv(data_path)
# lowercase and strip column names
df.columns = [col.lower().strip() for col in df.columns]
df = df.ffill().bfill()

# ----------------------------------------------------------
# 🔧 Auto-Rename Columns to Expected Names
# ----------------------------------------------------------
expected_columns = ["close", "open", "high", "low", "volume"]
for col in expected_columns:
    candidates = [c for c in df.columns if col in c.lower()]
    if candidates:
        df.rename(columns={candidates[0]: col}, inplace=True)
    else:
        st.error(f"❌ The '{col}' column is missing in the dataset! Please make sure your CSV has it.")
        st.stop()

# ----------------------------------------------------------
# 📈 Compute Technical Indicators (if not present)
# ----------------------------------------------------------
if "ma5" not in df.columns:
    df["ma5"] = df["close"].rolling(window=5).mean()
if "ma10" not in df.columns:
    df["ma10"] = df["close"].rolling(window=10).mean()
if "ma20" not in df.columns:
    df["ma20"] = df["close"].rolling(window=20).mean()
if "volatility" not in df.columns:
    df["volatility"] = df["close"].pct_change().rolling(window=10).std()

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

if "rsi" not in df.columns:
    df["rsi"] = compute_rsi(df["close"])

recent_data = df.tail(50).reset_index(drop=True)

# ----------------------------------------------------------
# 🏷️ Header
# ----------------------------------------------------------
st.title("📈 Stock Market Live Dashboard (Random Forest)")
st.caption(f"🗂 Current Mode: **{mode}** — showing {len(df)} records")
st.markdown("---")

# ----------------------------------------------------------
# 🕒 Show Data Table
# ----------------------------------------------------------
st.subheader("🕒 Recent Stock Data (Last 50 Records)")
show_cols = [c for c in ["date", "close", "ma5", "ma10", "ma20", "volatility", "rsi"] if c in df.columns]
st.dataframe(recent_data[show_cols].round(3), use_container_width=True)

# ----------------------------------------------------------
# 📊 Price Chart
# ----------------------------------------------------------
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
# 🧮 Sidebar Inputs for Manual Prediction
# ----------------------------------------------------------
st.sidebar.header("🧮 Input Stock Data for Prediction")

input_fields = {
    "open": "Open Price",
    "high": "High Price",
    "low": "Low Price",
    "close": "Current Close",
    "volume": "Volume",
    "ma5": "MA5",
    "ma10": "MA10",
    "ma20": "MA20",
    "volatility": "Volatility",
    "rsi": "RSI",
    "lag_close_1": "Previous Close",
}

inputs = {k: st.sidebar.number_input(v, value=0.0) for k, v in input_fields.items()}
input_df = pd.DataFrame([inputs])

# ----------------------------------------------------------
# 🎯 Prediction Section
# ----------------------------------------------------------
st.subheader("🎯 Stock Movement Prediction (Random Forest)")

if st.button("🚀 Predict Now"):
    try:
        expected_features = (
            list(rf_model.feature_names_in_) 
            if hasattr(rf_model, "feature_names_in_") 
            else list(input_df.columns)
        )
        for feat in expected_features:
            if feat not in input_df.columns:
                input_df[feat] = 0.0
        input_df = input_df[expected_features]

        movement = rf_model.predict(input_df)[0]
        movement_label = "📈 UP" if movement == 1 else "📉 DOWN"
        movement_color = "green" if movement == 1 else "red"

        st.markdown(f"<h4 style='color:{movement_color};'>Predicted Movement: {movement_label}</h4>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

st.markdown("---")
st.caption("✅ Powered by Random Forest | Live + Test + Historical Mode | Streamlit Dashboard")
