# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1️⃣ Download Historical Stock Data
stock_symbol = "AAPL"  # Change this to any stock ticker (e.g., "GOOGL", "TSLA")
start_date = "2015-01-01"
end_date = "2024-01-01"

df = yf.download(stock_symbol, start=start_date, end=end_date)
df = df[['Close']]  # Use only the 'Close' price

# 2️⃣ Data Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# Create sequences for LSTM model
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Define time steps
time_steps = 60

# Prepare training and testing data
train_size = int(len(df_scaled) * 0.8)
train_data, test_data = df_scaled[:train_size], df_scaled[train_size - time_steps:]

X_train, y_train = create_sequences(train_data, time_steps)
X_test, y_test = create_sequences(test_data, time_steps)

# Reshape inputs for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 3️⃣ Build the LSTM Model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)  # Predict the closing price
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# 4️⃣ Train the Model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 5️⃣ Make Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Convert back to original scale

# Convert actual test values back to original scale
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 6️⃣ Visualize the Results
plt.figure(figsize=(14, 6))
plt.plot(y_test_actual, color='blue', label="Actual Stock Price")
plt.plot(predictions, color='red', label="Predicted Stock Price")
plt.title(f"{stock_symbol} Stock Price Prediction using LSTM")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
