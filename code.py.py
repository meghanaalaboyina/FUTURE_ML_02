# STOCK PRICE PREDICTION USING LSTM

# Install and import necessary libraries
!pip install yfinance --quiet
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# Load Apple's stock data and select closing prices
df = yf.download('AAPL', start='2015-01-01', end='2024-12-31')[['Close']].dropna()

# Visualize closing prices
plt.figure(figsize=(14,5))
plt.plot(df['Close'], label='Close Price')
plt.title('AAPL Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Scale data to (0,1) range
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Prepare training data: 60-day window
window = 60
X, y = [], []
for i in range(window, len(scaled_data)):
    X.append(scaled_data[i-window:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(window,1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=20, batch_size=32)

# Predict on training data for evaluation
train_pred = model.predict(X)
train_pred = scaler.inverse_transform(train_pred)
real_prices = scaler.inverse_transform(y.reshape(-1,1))

# Calculate error metrics
rmse = math.sqrt(mean_squared_error(real_prices, train_pred))
mae = mean_absolute_error(real_prices, train_pred)
print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# Plot actual vs predicted prices
plt.figure(figsize=(14,6))
plt.plot(real_prices, label='Actual Price')
plt.plot(train_pred, label='Predicted Price')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
