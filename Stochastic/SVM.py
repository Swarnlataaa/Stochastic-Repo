import numpy as np
import matplotlib.pyplot as plt

def heston_model(S0, r, kappa, theta, sigma, rho, T, dt, num_paths):
    """
    Simulate the Heston stochastic volatility model.

    Parameters:
    - S0: Initial stock price
    - r: Risk-free interest rate
    - kappa: Mean-reversion speed of volatility
    - theta: Long-term mean of volatility
    - sigma: Volatility of volatility
    - rho: Correlation between the stock price and its volatility
    - T: Time to maturity
    - dt: Time step size
    - num_paths: Number of simulation paths

    Returns:
    - time_points: Array of time points
    - stock_prices: Simulated stock prices
    - volatilities: Simulated volatilities
    """
    num_steps = int(T / dt)
    time_points = np.linspace(0, T, num_steps + 1)

    stock_prices = np.zeros((num_paths, num_steps + 1))
    volatilities = np.zeros((num_paths, num_steps + 1))

    for i in range(num_paths):
        stock_prices[i, 0] = S0
        volatilities[i, 0] = theta

        for t in range(1, num_steps + 1):
            dW1 = np.random.normal(0, np.sqrt(dt))
            dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt))

            stock_prices[i, t] = stock_prices[i, t - 1] * np.exp((r - 0.5 * volatilities[i, t - 1]**2) * dt +
                                                                  volatilities[i, t - 1] * np.sqrt(dt) * dW1)
            volatilities[i, t] = volatilities[i, t - 1] + kappa * (theta - volatilities[i, t - 1]) * dt + \
                                 sigma * np.sqrt(volatilities[i, t - 1]) * np.sqrt(dt) * dW2

    return time_points, stock_prices, volatilities

# Example usage for Heston model
S0 = 100.0
r = 0.02
kappa = 2.0
theta = 0.05
sigma = 0.2
rho = -0.7
T = 1.0
dt = 1/252.0  # Daily time step
num_paths = 5

time_points, stock_prices, volatilities = heston_model(S0, r, kappa, theta, sigma, rho, T, dt, num_paths)

# Plot the results
plt.figure(figsize=(12, 6))
for i in range(num_paths):
    plt.plot(time_points, stock_prices[i], label=f'Path {i + 1}')

plt.title('Heston Model Simulation - Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
for i in range(num_paths):
    plt.plot(time_points, volatilities[i], label=f'Path {i + 1}')

plt.title('Heston Model Simulation - Volatilities')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.legend()
plt.show()
