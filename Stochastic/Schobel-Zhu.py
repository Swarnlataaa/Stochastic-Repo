import numpy as np
import matplotlib.pyplot as plt

def schobel_zhu_model(S0, mu, kappa, theta, xi, rho, gamma, phi, psi, T, dt, num_paths):
    """
    Simulate the Schobel-Zhu stochastic volatility model.

    Parameters:
    - S0: Initial stock price
    - mu: Drift for the stock price
    - kappa: Mean-reversion speed of variance
    - theta: Long-term mean of variance
    - xi: Volatility of variance
    - rho: Correlation between stock price and variance
    - gamma: Mean-reversion speed of volatility of volatility
    - phi: Long-term mean of volatility of volatility
    - psi: Volatility of volatility of variance
    - T: Time to maturity
    - dt: Time step size
    - num_paths: Number of simulation paths

    Returns:
    - time_points: Array of time points
    - stock_prices: Simulated stock prices
    - variances: Simulated variances
    - volatilities_of_volatility: Simulated volatilities of volatility
    """
    num_steps = int(T / dt)
    time_points = np.linspace(0, T, num_steps + 1)

    stock_prices = np.zeros((num_paths, num_steps + 1))
    variances = np.zeros((num_paths, num_steps + 1))
    volatilities_of_volatility = np.zeros((num_paths, num_steps + 1))

    for i in range(num_paths):
        for t in range(1, num_steps + 1):
            dW1 = np.random.normal(0, np.sqrt(dt))
            dW2 = np.random.normal(0, np.sqrt(dt))
            dW3 = np.random.normal(0, np.sqrt(dt))

            stock_prices[i, t] = stock_prices[i, t - 1] + mu * stock_prices[i, t - 1] * dt + \
                                  np.sqrt(variances[i, t - 1]) * stock_prices[i, t - 1] * dW1

            variances[i, t] = variances[i, t - 1] + kappa * (theta - variances[i, t - 1]) * dt + \
                              xi * np.sqrt(variances[i, t - 1]) * dW2

            volatilities_of_volatility[i, t] = volatilities_of_volatility[i, t - 1] + \
                                               gamma * (phi - volatilities_of_volatility[i, t - 1]) * dt + \
                                               psi * np.sqrt(volatilities_of_volatility[i, t - 1]) * dW3

    return time_points, stock_prices, variances, volatilities_of_volatility

# Example usage for the Schobel-Zhu model
S0 = 100.0
mu = 0.02
kappa = 2.0
theta = 0.05
xi = 0.2
rho = -0.7
gamma = 0.1
phi = 0.04
psi = 0.15
T = 1.0
dt = 1/252.0  # Daily time step
num_paths = 5

time_points, stock_prices, variances, volatilities_of_volatility = \
    schobel_zhu_model(S0, mu, kappa, theta, xi, rho, gamma, phi, psi, T, dt, num_paths)

# Plot the results
plt.figure(figsize=(12, 6))
for i in range(num_paths):
    plt.plot(time_points, stock_prices[i], label=f'Path {i + 1}')

plt.title('Schobel-Zhu Model Simulation - Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
for i in range(num_paths):
    plt.plot(time_points, variances[i], label=f'Path {i + 1}')

plt.title('Schobel-Zhu Model Simulation - Variances')
plt.xlabel('Time')
plt.ylabel('Variance')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
for i in range(num_paths):
    plt.plot(time_points, volatilities_of_volatility[i], label=f'Path {i + 1}')

plt.title('Schobel-Zhu Model Simulation - Volatilities of Volatility')
plt.xlabel('Time')
plt.ylabel('Volatility of Volatility')
plt.legend()
plt.show()
