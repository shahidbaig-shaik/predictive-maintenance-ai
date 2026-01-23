import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Industrial AI: Predictive Maintenance", layout="wide")
st.title("✈️ Aero-Guard: Intelligent Engine Monitoring")
st.markdown("**System Status:** `ONLINE` | **Model:** `LSTM-v1 (PyTorch)` | **Data Source:** `NASA CMAPSS IoT`")

# --- MODEL ARCHITECTURE (Must match training) ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :] # Take the last time step
        out = self.dropout2(out)
        out = self.fc(out)
        return out

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = LSTMModel(input_size=24, hidden_size1=100, hidden_size2=50, output_size=1, dropout_rate=0.2)
    try:
        model.load_state_dict(torch.load('nasa_engine_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    st.error("Model file not found! Please run train_model.py first.")
    st.stop()

# --- SIDEBAR: SIMULATION CONTROLS ---
st.sidebar.header("🛠️ Simulation Controls")
engine_id = st.sidebar.selectbox("Select Engine Unit", range(1, 101))
cycle_slider = st.sidebar.slider("Current Flight Cycle", 50, 300, 150)

# --- MAIN DASHBOARD LOGIC ---
def generate_live_data(cycle):
    # Generating 50 time-steps of random data (normalized 0-1)
    noise = np.random.normal(0, 0.1, (1, 50, 24)) 
    base_signal = np.linspace(0.5, 0.8, 50).reshape(1, 50, 1) 
    data = np.zeros((1, 50, 24))
    data[:, :, :] = 0.5 + noise 
    
    degradation_factor = cycle / 300.0
    data = data + (degradation_factor * 0.2) 
    return data.astype(np.float32)

live_data = generate_live_data(cycle_slider)

# PREDICT
with torch.no_grad():
    input_tensor = torch.tensor(live_data)
    prediction = model(input_tensor)
    rul_prediction = int(prediction.item())

# --- VISUALIZATION ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Current Cycle", value=f"{cycle_slider}")

with col2:
    color = "normal"
    if rul_prediction < 50: color = "off"
    elif rul_prediction < 100: color = "off"
    
    st.metric(label="Predicted Remaining Life (RUL)", value=f"{rul_prediction} Cycles", delta=f"{rul_prediction - 50} vs Threshold")

with col3:
    status = "HEALTHY"
    if rul_prediction < 50: status = "CRITICAL FAILURE IMMINENT"
    elif rul_prediction < 100: status = "MAINTENANCE REQUIRED"
    
    st.info(f"Status: **{status}**")

# --- INTERPRETABILITY ---
st.subheader("📊 Sensor Fusion Analysis")
st.caption("Visualizing the last 50 cycles of vibration and temperature sensors.")

chart_data = pd.DataFrame(live_data[0][:, [2, 3, 4]], columns=["Fan Speed", "Core Temp", "Pressure"])
st.line_chart(chart_data)

st.divider()
st.markdown("### 🧠 Technical Architecture")
st.code("""
[SENSORS] --> [DATA INGESTION] --> [SLIDING WINDOW (50)] --> [LSTM LAYER 1] --> [LSTM LAYER 2] --> [RUL PREDICTION]
""", language="text")
