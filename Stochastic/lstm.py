import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Function to simulate geometric Brownian motion
def simulate_geometric_brownian_motion(mu, sigma, S0, dt, num_steps):
    t = np.linspace(0, dt * num_steps, num_steps + 1)
    W = np.random.normal(0, np.sqrt(dt), size=num_steps)
    S = S0 * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W))
    return t, S

# Function to generate financial time series data
def generate_financial_data(mu, sigma, S0, dt, num_steps):
    _, prices = simulate_geometric_brownian_motion(mu, sigma, S0, dt, num_steps)
    returns = np.diff(np.log(prices))
    return prices, returns

# Function to preprocess data for LSTM
def preprocess_lstm_data(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

# Parameters
mu = 0.02
sigma = 0.2
S0 = 100
dt = 1/252
num_steps = 252
look_back = 10
epochs = 50
batch_size = 32

# Generate financial time series data
prices, returns = generate_financial_data(mu, sigma, S0, dt, num_steps)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
returns_scaled = scaler.fit_transform(returns.reshape(-1, 1)).reshape(-1)

# Preprocess data for LSTM
X, y = preprocess_lstm_data(returns_scaled, look_back)

# Reshape input for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(X.shape[1], 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=epochs, batch_size=batch_size)

# Make predictions
predicted_returns_scaled = model.predict(X)
predicted_returns = scaler.inverse_transform(predicted_returns_scaled)

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(prices, label='Actual Prices')
plt.title('Actual Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(returns, label='Actual Returns', color='blue')
plt.plot(np.arange(look_back, len(predicted_returns) + look_back), predicted_returns, label='Predicted Returns', color='red')
plt.title('Actual and Predicted Returns')
plt.xlabel('Time')
plt.ylabel('Return')
plt.legend()

plt.tight_layout()
plt.show()
