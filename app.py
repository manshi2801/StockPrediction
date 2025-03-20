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

# App Title
st.title("📈 Stock Investment Advisor")

# Upload File (CSV or Image)
uploaded_file = st.file_uploader("Upload Stock History CSV or Stock Chart (PNG, JPG)", type=["csv", "png", "jpg"])

def process_csv(df_stock):
    """Processes CSV file data and makes stock price predictions."""
    required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_cols.issubset(df_stock.columns):
        st.error(f"CSV must contain columns: {required_cols}")
        return

    st.success("File uploaded successfully! Processing data...")

    # Compute Moving Averages
    df_stock['SMA_5'] = df_stock['Close'].rolling(window=5).mean()
    df_stock['SMA_20'] = df_stock['Close'].rolling(window=20).mean()

    # Preprocess Data
    scaler = StandardScaler()
    df_stock_scaled = scaler.fit_transform(df_stock[['Open', 'High', 'Low', 'Close', 'Volume']].dropna())

    # Prepare Features and Labels
    X_stock = df_stock_scaled[:, :-1]  # All except the last column
    y_stock = df_stock_scaled[:, -1]   # Last column (Close price)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_stock, y_stock, test_size=0.2, random_state=42)

    # Reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define LSTM Model
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train Model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # Make Predictions
    predictions = model.predict(X_test)

    # Convert back to original scale
    predictions_unscaled = scaler.inverse_transform(
        np.hstack((X_test.reshape(X_test.shape[0], -1), predictions))
    )[:, -1]

    y_test_unscaled = scaler.inverse_transform(
        np.hstack((X_test.reshape(X_test.shape[0], -1), y_test.reshape(-1, 1)))
    )[:, -1]

    # Display Graph
    st.subheader("📊 Predicted vs Actual Prices")
    plt.figure(figsize=(12, 5))
    plt.plot(y_test_unscaled, label="Actual Prices")
    plt.plot(predictions_unscaled, label="Predicted Prices", linestyle="dashed")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("Stock Price Prediction")
    st.pyplot(plt)

    # Investment Decision
    last_actual_price = y_test_unscaled[-1]
    last_predicted_price = predictions_unscaled[-1]
    price_change = (last_predicted_price - last_actual_price) / last_actual_price

    st.subheader("📌 Investment Decision")
    threshold = 0.02  # 2% increase required for a "good" investment
    if price_change > threshold:
        st.success(f"✅ This stock is a good investment! 📈 ({price_change:.2%} increase)")
    else:
        st.error(f"❌ This stock is risky! Consider other options. 📉 ({price_change:.2%} change)")

def process_image(image):
    """Extracts stock data from image charts using OCR and trend detection."""
    st.image(image, caption="Uploaded Stock Chart", use_column_width=True)
    st.subheader("🔍 Extracting Stock Data from Image...")

    # Convert image to grayscale for better OCR results
    img_cv = np.array(image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    # Use OCR to extract text (stock prices) from the image
    text = pytesseract.image_to_string(gray)
    st.text("Extracted Text from Image:")
    st.write(text)

    # Detect edges to identify stock trends
    edges = cv2.Canny(gray, 50, 150)
    st.image(edges, caption="Stock Trend Detection", use_column_width=True)

    # Analyze trend using Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
    stock_trend = "📈 Uptrend detected! Good investment." if lines is not None and len(lines) > 10 else "📉 Downtrend detected! Be cautious."

    # Display Investment Decision
    st.subheader("📌 Investment Decision")
    st.success(stock_trend) if "Uptrend" in stock_trend else st.error(stock_trend)

# Handle file upload
if uploaded_file is not None:
    file_type = uploaded_file.type

    if file_type == "text/csv":
        df_stock = pd.read_csv(uploaded_file)
        process_csv(df_stock)

    elif file_type in ["image/png", "image/jpeg"]:
        image = Image.open(uploaded_file)
        process_image(image)
