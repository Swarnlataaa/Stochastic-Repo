import numpy as np
import matplotlib.pyplot as plt

def bates_model(S0, mu, kappa, theta, xi, rho, lam, delta, T, dt, num_paths):
    """
    Simulate the Bates stochastic volatility model.

    Parameters:
    - S0: Initial stock price
    - mu: Drift for the stock price
    - kappa: Mean-reversion speed of variance
    - theta: Long-term mean of variance
    - xi: Volatility of variance
    - rho: Correlation between stock price and variance
    - lam: Jump intensity
    - delta: Jump volatility
    - T: Time to maturity
    - dt: Time step size
    - num_paths: Number of simulation paths

    Returns:
    - time_points: Array of time points
    - stock_prices: Simulated stock prices
    - variances: Simulated variances
    """
    num_steps = int(T / dt)
    time_points = np.linspace(0, T, num_steps + 1)

    stock_prices = np.zeros((num_paths, num_steps + 1))
    variances = np.zeros((num_paths, num_steps + 1))

    for i in range(num_paths):
        for t in range(1, num_steps + 1):
            dW1 = np.random.normal(0, np.sqrt(dt))
            dW2 = np.random.normal(0, np.sqrt(dt))
            dZ = np.random.normal(0, np.sqrt(dt))

            N_t = np.random.poisson(lam * dt)
            J_t = np.sum(np.random.normal(0, delta, N_t))

            stock_prices[i, t] = stock_prices[i, t - 1] + mu * stock_prices[i, t - 1] * dt + \
                                  np.sqrt(variances[i, t - 1]) * stock_prices[i, t - 1] * dW1 + \
                                  J_t * stock_prices[i, t - 1]

            variances[i, t] = variances[i, t - 1] + kappa * (theta - variances[i, t - 1]) * dt + \
                              xi * np.sqrt(variances[i, t - 1]) * dW2 + \
                              lam * N_t

    return time_points, stock_prices, variances

# Example usage for the Bates model
S0 = 100.0
mu = 0.02
kappa = 2.0
theta = 0.05
xi = 0.2
rho = -0.7
lam = 0.1
delta = 0.15
T = 1.0
dt = 1/252.0  # Daily time step
num_paths = 5

time_points, stock_prices, variances = \
    bates_model(S0, mu, kappa, theta, xi, rho, lam, delta, T, dt, num_paths)

# Plot the results
plt.figure(figsize=(12, 6))
for i in range(num_paths):
    plt.plot(time_points, stock_prices[i], label=f'Path {i + 1}')

plt.title('Bates Model Simulation - Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
for i in range(num_paths):
    plt.plot(time_points, variances[i], label=f'Path {i + 1}')

plt.title('Bates Model Simulation - Variances')
plt.xlabel('Time')
plt.ylabel('Variance')
plt.legend()
plt.show()
