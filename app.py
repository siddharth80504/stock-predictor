import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import requests
from bs4 import BeautifulSoup
import base64
import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

st.set_page_config(page_title="SS Stock App", layout="wide", initial_sidebar_state="collapsed")

if "page" not in st.session_state:
    st.session_state.page = None

# --- Background ---
def add_bg(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(f"""
            <style>
                .stApp {{
                    background-image: url("data:image/png;base64,{encoded}");
                    background-size: cover;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                }}
            </style>
        """, unsafe_allow_html=True)

add_bg("ss.png")

stock_list = ['AAPL', 'GOOG', 'MSFT', 'TSLA', 'AMZN', 'NFLX', 'META']

if st.session_state.page:
    back_col = st.columns([10, 1])[1]
    with back_col:
        if st.button("Back", help="Go to Home"):
            st.session_state.page = None

if st.session_state.page is None:
    st.markdown("""
        <h1 style='text-align: center; color: gold;'>Stock Dashboard</h1>
        <p style='text-align: center;'>Select a feature below</p>
        <p style='text-align: center; font-size: 18px; margin-top: 30px; color: white;'>
            "The stock market is a device for transferring money from the impatient to the patient."
        </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Predict Stock (Transformer)", use_container_width=True):
            st.session_state.page = "predict"
        if st.button("View Stock Chart", use_container_width=True):
            st.session_state.page = "chart"
    with col2:
        if st.button("Today's Stock News", use_container_width=True):
            st.session_state.page = "news"

# --- TRANSFORMER MODEL (CPU-optimized) ---
class StockTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=32, nhead=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(d_model, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.input_linear(x)
        x = self.transformer(x)
        x = self.fc1(x[:, -1, :])  # last timestep
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --- PREDICT PAGE ---
if st.session_state.page == "predict":
    st.header("Predict Stock (Transformer)")
    ticker = st.selectbox("Choose stock", stock_list)
    forecast_days = st.selectbox("Days to Forecast", [30, 60, 90])

    df = yf.download(ticker, start="2015-01-01")[["Close"]].dropna()
    data = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(30, len(data_scaled)):
        X.append(data_scaled[i - 30:i])
        y.append(data_scaled[i])
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32)

    model = StockTransformer()
    model_file = f"{ticker}_transformer_model.pt"
    info_file = f"{ticker}_last_trained.txt"
    last_date = df.index[-1].strftime("%Y-%m-%d")
    retrain = True

    if os.path.exists(model_file) and os.path.exists(info_file):
        with open(info_file, "r") as f:
            saved_date = f.read().strip()
        if saved_date == last_date:
            # Load tolerantly in case old checkpoints had different shapes
            state = torch.load(model_file, map_location="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            model.eval()
            retrain = False
            st.info("✅ Transformer model loaded from disk.")

    if retrain:
        st.warning("🔄 Training Transformer model (CPU-optimized)...")
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # ✅ Mini-batch training for speed on CPU
        batch_size = 64
        epochs = 10
        ds = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in dl:
                optimizer.zero_grad()
                output = model(xb)
                loss = loss_fn(output.squeeze(), yb.squeeze())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
        torch.save(model.state_dict(), model_file)
        with open(info_file, "w") as f:
            f.write(last_date)
        st.success("🚀 Transformer model trained and saved.")

    # Forecast
    model.eval()
    input_seq = torch.tensor(data_scaled[-30:], dtype=torch.float32).unsqueeze(0)
    predictions = []

    last_real = float(data[-1][0])
    min_limit = 0.85 * last_real
    max_limit = 1.15 * last_real

    with torch.no_grad():
        for _ in range(forecast_days):
            pred = model(input_seq).item()
            if np.isnan(pred) or pred < 0:
                break
            real = scaler.inverse_transform([[pred]])[0][0]
            real = max(min(real, max_limit), min_limit)
            predictions.append(real)
            scaled = scaler.transform([[real]])[0][0]
            next_input = torch.tensor([[[scaled]]], dtype=torch.float32)
            input_seq = torch.cat((input_seq[:, 1:, :], next_input), dim=1)

    if len(predictions) < 2:
        st.error("Forecast failed: Not enough prediction data.")
    else:
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=len(predictions))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index[-60:], df["Close"].values[-60:], label="Historical")
        ax.plot(future_dates, predictions, label="Forecast (Transformer)", linestyle="--", color="gold")
        ax.set_title(f"{ticker} {forecast_days}-Day Forecast (Transformer)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.metric("Last Known Price", f"${float(df['Close'].values[-1]):.2f}")
        st.metric(f"{forecast_days}-Day Forecast", f"${predictions[-1]:.2f}")

# --- STOCK CHART PAGE ---
elif st.session_state.page == "chart":
    st.header("View Stock Chart")
    selected = st.selectbox("Choose stock", stock_list)
    df = yf.download(selected, start="2015-01-01")
    st.line_chart(df["Close"])
    st.write(df.tail())

# --- NEWS PAGE ---
elif st.session_state.page == "news":
    st.header("Latest Stock Market News")
    def fetch_news():
        url = "https://finance.yahoo.com/"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        links = soup.find_all("a", href=True)
        headlines = []
        for link in links:
            if "/news/" in link["href"] and len(link.get_text(strip=True)) > 30:
                headlines.append(link.get_text(strip=True))
        return headlines[:5]

    for i, news in enumerate(fetch_news(), 1):
        st.write(f"**{i}.** {news}")
