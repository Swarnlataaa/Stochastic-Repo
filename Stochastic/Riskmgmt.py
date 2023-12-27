import numpy as np
import matplotlib.pyplot as plt

# Function to simulate geometric Brownian motion for asset price
def simulate_price(S0, mu, sigma, dt, num_steps):
    prices = np.zeros(num_steps + 1)
    prices[0] = S0

    for i in range(1, num_steps + 1):
        dW = np.random.normal(0, np.sqrt(dt))
        prices[i] = prices[i - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * dW)

    return prices

# Function to simulate portfolio value over time
def simulate_portfolio(prices, weights):
    return np.sum(prices * weights, axis=1)

# Function to calculate portfolio metrics
def calculate_portfolio_metrics(portfolio_values):
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    mean_return = np.mean(returns)
    volatility = np.std(returns)
    sharpe_ratio = mean_return / volatility
    max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values)
    
    return {
        'Mean Return': mean_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

# Main simulation function
def simulate_portfolio_risk(S0, mu, sigma, dt, num_steps, weights):
    prices = simulate_price(S0, mu, sigma, dt, num_steps)
    portfolio_values = simulate_portfolio(prices, weights)
    metrics = calculate_portfolio_metrics(portfolio_values)

    return portfolio_values, metrics

# Example usage
np.random.seed(42)

# Parameters
S0 = 100.0  # Initial stock price
mu = 0.05   # Expected return
sigma = 0.2  # Volatility
dt = 1/252.0  # Daily time step
num_steps = 252  # Number of trading days in a year
num_simulations = 1000  # Number of portfolio simulations

# Define portfolio weights (example: 60% stock, 40% bonds)
weights = np.array([0.6, 0.4])

# Run simulations
portfolio_values, metrics = simulate_portfolio_risk(S0, mu, sigma, dt, num_steps, weights)

# Visualization
time_points = np.arange(0, num_steps + 1) * dt
plt.plot(time_points, portfolio_values.T, color='lightgray', linewidth=0.5)
plt.xlabel('Time')
plt.ylabel('Portfolio Value')
plt.title('Monte Carlo Simulation of Portfolio Value')
plt.show()

# Display portfolio risk metrics
print("Portfolio Risk Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")
