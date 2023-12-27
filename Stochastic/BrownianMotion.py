import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to generate simulated Brownian motion data
def generate_brownian_motion(T, dt, mu, sigma, seed=None):
    np.random.seed(seed)
    num_steps = int(T / dt)
    t = np.linspace(0, T, num_steps + 1)
    W = np.zeros(num_steps + 1)

    for i in range(1, num_steps + 1):
        dW = np.random.normal(0, np.sqrt(dt))
        W[i] = W[i - 1] + dW

    return t, W

# Simulate Brownian motion data
T = 1  # Time horizon in years
dt = 1/252  # Daily time step
mu = 0.05  # Drift
sigma = 0.2  # Volatility
t, W = generate_brownian_motion(T, dt, mu, sigma, seed=42)

# Plot simulated Brownian motion
plt.figure(figsize=(10, 6))
plt.plot(t, W, label='Simulated Brownian Motion')
plt.title('Simulated Brownian Motion')
plt.xlabel('Time (Years)')
plt.ylabel('Value')
plt.legend()
plt.show()

# Calculate parameters
returns = np.diff(W) / W[:-1]
volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility

# Print calculated parameters
print(f"Volatility: {volatility:.4f}")

# Function to calculate Value at Risk (VaR)
def calculate_var(returns, confidence_level=0.95):
    return -np.percentile(returns, 100 * (1 - confidence_level))

# Calculate VaR at 95% confidence level
var_95 = calculate_var(returns)
print(f"Value at Risk (VaR) at 95% confidence level: {var_95:.4f}")

# Function to calculate Sharpe Ratio
def calculate_sharpe_ratio(returns, risk_free_rate=0):
    mean_return = np.mean(returns) * 252  # Annualized mean return
    excess_return = mean_return - risk_free_rate
    volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
    return excess_return / volatility

# Calculate Sharpe Ratio
sharpe_ratio = calculate_sharpe_ratio(returns)
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

# Calculate Standard Deviation
std_deviation = np.std(returns) * np.sqrt(252)
print(f"Standard Deviation: {std_deviation:.4f}")

# Generate another Brownian motion for correlation and covariance
_, W2 = generate_brownian_motion(T, dt, mu, sigma, seed=10)

# Calculate Covariance and Correlation
covariance = np.cov(returns, W2[:-1])[0, 1]
correlation = np.corrcoef(returns, W2[:-1])[0, 1]

print(f"Covariance: {covariance:.4f}")
print(f"Correlation: {correlation:.4f}")
