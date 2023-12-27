import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma):
    """
    Calculate Black-Scholes option pricing formula for a European call option.

    Parameters:
    - S: Current stock price
    - K: Option strike price
    - T: Time to expiration (in years)
    - r: Risk-free interest rate
    - sigma: Volatility of the underlying stock

    Returns:
    - Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def monte_carlo_option_pricing(S, K, T, r, sigma, num_simulations=10000):
    """
    Use Monte Carlo simulation to price a European call option.

    Parameters:
    - S: Current stock price
    - K: Option strike price
    - T: Time to expiration (in years)
    - r: Risk-free interest rate
    - sigma: Volatility of the underlying stock
    - num_simulations: Number of Monte Carlo simulations

    Returns:
    - Monte Carlo estimated option price
    """
    dt = T / 252  # Assuming 252 trading days in a year
    num_steps = int(T / dt)

    simulations = np.zeros((num_simulations, num_steps + 1))
    simulations[:, 0] = S

    for i in range(1, num_steps + 1):
        Z = np.random.standard_normal(num_simulations)
        simulations[:, i] = simulations[:, i - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    option_payoffs = np.maximum(simulations[:, -1] - K, 0)
    option_price = np.exp(-r * T) * np.mean(option_payoffs)

    return option_price

# Example usage:
S0 = 100  # Current stock price
K = 100   # Option strike price
T = 1     # Time to expiration (in years)
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility of the underlying stock

# Calculate option price using Black-Scholes formula
bs_price = black_scholes(S0, K, T, r, sigma)
print(f"Black-Scholes Option Price: {bs_price:.4f}")

# Calculate option price using Monte Carlo simulation
mc_price = monte_carlo_option_pricing(S0, K, T, r, sigma)
print(f"Monte Carlo Option Price: {mc_price:.4f}")
