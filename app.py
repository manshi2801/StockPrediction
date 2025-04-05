import streamlit as st
import pandas as pd
import numpy as np
import cv2
import pytesseract
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import pyttsx3
import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set Streamlit theme and emoji header
st.set_page_config(page_title="Stock Buddy 💖", page_icon="📈", layout="centered")
st.markdown("""
    <style>
        .main {
            background-color: #FFF0F5;
            color: #4B0082;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #BA55D3;
        }
        .stButton>button {
            background-color: #FFB6C1;
            color: white;
            border-radius: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("💖 Your Cute Stock Investment Buddy 🐻✨")

import threading

def speak(text):
    def run():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run).start()

# Upload File (CSV or Image)
st.markdown("### 📤 Upload a file, sweetie!")
uploaded_file = st.file_uploader("Choose a Stock History CSV or Chart Image (PNG, JPG)", type=["csv", "png", "jpg"])

# Helper to save/load session
SAVE_PATH = "session_data.json"
def save_session(data):
    with open(SAVE_PATH, 'w') as f:
        json.dump(data, f)
def load_session():
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, 'r') as f:
            return json.load(f)
    return {}

# Process CSV

def process_csv(df_stock):
    required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_cols.issubset(df_stock.columns):
        st.error(f"CSV must contain columns: {required_cols}")
        return

    st.success("✨ Yay! File uploaded successfully!")

    df_stock['SMA_5'] = df_stock['Close'].rolling(window=5).mean()
    df_stock['SMA_20'] = df_stock['Close'].rolling(window=20).mean()

    scaler = StandardScaler()
    df_stock_scaled = scaler.fit_transform(df_stock[['Open', 'High', 'Low', 'Close', 'Volume']].dropna())

    X_stock = df_stock_scaled[:, :-1]
    y_stock = df_stock_scaled[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X_stock, y_stock, test_size=0.2, random_state=42)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    predictions = model.predict(X_test)

    predictions_unscaled = scaler.inverse_transform(
        np.hstack((X_test.reshape(X_test.shape[0], -1), predictions))
    )[:, -1]

    y_test_unscaled = scaler.inverse_transform(
        np.hstack((X_test.reshape(X_test.shape[0], -1), y_test.reshape(-1, 1)))
    )[:, -1]

    st.subheader("🎀 Predicted vs Actual Prices")
    plt.figure(figsize=(12, 5))
    plt.plot(y_test_unscaled, label="Actual Prices", color='orchid')
    plt.plot(predictions_unscaled, label="Predicted Prices", linestyle="dashed", color='mediumvioletred')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("💹 Price Prediction Magical Graph")
    st.pyplot(plt)

    # Accuracy Metrics
    mae = mean_absolute_error(y_test_unscaled, predictions_unscaled)
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions_unscaled))

    st.subheader("📏 Prediction Accuracy Metrics")
    st.markdown(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.markdown(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")

    last_actual_price = y_test_unscaled[-1]
    last_predicted_price = predictions_unscaled[-1]
    price_change = (last_predicted_price - last_actual_price) / last_actual_price

    confidence = abs(price_change) * 100
    st.subheader("💫 Confidence Level")
    st.progress(min(confidence / 100, 1.0))
    st.write(f"Confidence: **{confidence:.2f}%**")

    st.subheader("🧠 Investment Advice")
    threshold = 0.02
    if price_change > threshold:
        st.success(f"💖 This stock is a sweet deal! 📈 ({price_change:.2%} increase)")
        speak("This stock is a sweet deal! Great job!")
    else:
        st.error(f"💔 Risk alert! Might want to hug your money tight. 📉 ({price_change:.2%} change)")
        speak("Be careful! This might be a risky choice.")

    # Save prediction
    save_session({"last_price": float(last_predicted_price), "confidence": float(confidence)})

# Process Image

def process_image(image):
    st.image(image, caption="🖼️ Your Lovely Stock Chart", use_container_width=True)
    st.subheader("🔍 Extracting Sweet Data from Image...")

    img_cv = np.array(image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    text = pytesseract.image_to_string(gray)
    st.text("📝 Extracted Text:")
    st.write(text)

    edges = cv2.Canny(gray, 50, 150)
    st.image(edges, caption="✨ Detected Edges", use_container_width=True)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
    stock_trend = "📈 Uptrend detected! Sparkles ahead!" if lines is not None and len(lines) > 10 else "📉 Hmm... Downtrend! Tread carefully, cupcake."

    st.subheader("🧠 Investment Advice")
    if "Uptrend" in stock_trend:
        st.success(stock_trend)
        speak("Sparkles ahead! This looks promising.")
    else:
        st.error(stock_trend)
        speak("Be cautious sweetheart, this might be risky.")

    save_session({"trend": stock_trend})

# Load previous session
if st.button("🕘 Load Previous Session"):
    session = load_session()
    if session:
        st.success("🧸 Loaded your last session!")
        st.write(session)
    else:
        st.warning("No previous session found, honey!")

# File Upload Handler
if uploaded_file is not None:
    file_type = uploaded_file.type
    if file_type == "text/csv":
        df_stock = pd.read_csv(uploaded_file)
        process_csv(df_stock)
    elif file_type in ["image/png", "image/jpeg"]:
        image = Image.open(uploaded_file)
        process_image(image)
