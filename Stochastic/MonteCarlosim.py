import numpy as np
import matplotlib.pyplot as plt

def geometric_brownian_motion(S0, mu, sigma, T, dt, num_paths):
    """
    Simulate geometric Brownian motion paths.

    Parameters:
    - S0: Initial stock price
    - mu: Drift (average rate of return)
    - sigma: Volatility (standard deviation of returns)
    - T: Time to maturity
    - dt: Time step size
    - num_paths: Number of simulation paths

    Returns:
    - time_points: Array of time points
    - stock_prices: Simulated stock prices
    """
    num_steps = int(T / dt)
    time_points = np.linspace(0, T, num_steps + 1)

    stock_prices = np.zeros((num_paths, num_steps + 1))
    stock_prices[:, 0] = S0

    for i in range(num_paths):
        for t in range(1, num_steps + 1):
            dW = np.random.normal(0, np.sqrt(dt))
            stock_prices[i, t] = stock_prices[i, t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * dW)

    return time_points, stock_prices

def european_call_option_price(S0, K, r, sigma, T, num_paths):
    """
    Price a European call option using Monte Carlo simulation.

    Parameters:
    - S0: Initial stock price
    - K: Strike price
    - r: Risk-free interest rate
    - sigma: Volatility (standard deviation of returns)
    - T: Time to maturity
    - num_paths: Number of simulation paths

    Returns:
    - option_price: Estimated option price
    """
    dt = T / 252.0  # Daily time step for simplicity
    mu = r - 0.5 * sigma**2
    time_points, stock_prices = geometric_brownian_motion(S0, mu, sigma, T, dt, num_paths)

    option_payoffs = np.maximum(stock_prices[:, -1] - K, 0)
    option_price = np.exp(-r * T) * np.mean(option_payoffs)

    return option_price

# Example usage
S0 = 100.0  # Initial stock price
K = 105.0   # Strike price
r = 0.05    # Risk-free interest rate
sigma = 0.2 # Volatility
T = 1.0     # Time to maturity
num_paths = 10000

call_option_price = european_call_option_price(S0, K, r, sigma, T, num_paths)
print(f"Estimated European Call Option Price: {call_option_price}")

# Visualization (optional)
time_points, stock_prices = geometric_brownian_motion(S0, r - 0.5 * sigma**2, sigma, T, T/252.0, num_paths)
plt.plot(time_points, stock_prices.T, color='lightgray', linewidth=0.5)
plt.title('Monte Carlo Simulation of Geometric Brownian Motion')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.show()
