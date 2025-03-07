import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

# App Title
st.title("📈 Stock Investment Advisor")

# Upload CSV File
uploaded_file = st.file_uploader("Upload Stock History CSV", type=["csv"])

if uploaded_file is not None:
    df_stock = pd.read_csv(uploaded_file)

    # Check if required columns exist
    required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_cols.issubset(df_stock.columns):
        st.error(f"CSV must contain columns: {required_cols}")
    else:
        st.success("File uploaded successfully! Processing data...")

        # Preprocess Data
        def preprocess_stock_data(df_stock):
            scaler = StandardScaler()
            df_stock_scaled = scaler.fit_transform(df_stock[['Open', 'High', 'Low', 'Close', 'Volume']])
            return df_stock_scaled, scaler

        df_stock_scaled, scaler = preprocess_stock_data(df_stock)

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

        # Save Model & Scaler
        model.save("stock_prediction_model.h5")
        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

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

        st.subheader("📌 Investment Decision")
        if last_predicted_price > last_actual_price:
            st.success("✅ This stock is a good investment! 📈")
        else:
            st.error("❌ This stock is risky! Consider other options. 📉")
