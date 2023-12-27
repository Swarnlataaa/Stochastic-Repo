import numpy as np
import matplotlib.pyplot as plt

# Parameters
r0 = 0.02  # Initial short rate
kappa = 0.1  # Mean-reversion speed
theta = 0.02  # Long-term mean of the short rate
sigma = 0.02  # Volatility
T = 1  # Time to maturity
num_paths = 5  # Number of simulation paths
num_steps = 252  # Number of time steps
dt = T / num_steps  # Time step

# Function to simulate the CIR process
def simulate_cir(r0, kappa, theta, sigma, T, num_paths, num_steps, dt):
    rates = np.zeros((num_paths, num_steps + 1))
    rates[:, 0] = r0

    for i in range(1, num_steps + 1):
        dW = np.random.normal(0, np.sqrt(dt), size=num_paths)
        dr = kappa * (theta - rates[:, i - 1]) * dt + sigma * np.sqrt(rates[:, i - 1]) * dW
        rates[:, i] = np.maximum(rates[:, i - 1] + dr, 0)  # Ensure non-negative rates

    return rates

# Function to calculate bond prices from simulated short rates
def calculate_bond_prices(rates, T, num_steps, dt):
    time_points = np.arange(num_steps + 1) * dt
    bond_prices = np.exp(-np.trapz(rates, dx=dt, axis=1))
    return np.exp(-rates[:, -1] * (T - time_points[-1])) * bond_prices

# Simulate interest rate paths
rate_paths = simulate_cir(r0, kappa, theta, sigma, T, num_paths, num_steps, dt)

# Calculate bond prices
bond_prices = calculate_bond_prices(rate_paths, T, num_steps, dt)

# Plot the results
plt.figure(figsize=(10, 6))

# Plot interest rate paths
plt.subplot(2, 1, 1)
for i in range(num_paths):
    plt.plot(np.arange(num_steps + 1) * dt, rate_paths[i, :], label=f'Path {i + 1}')
plt.title('Simulated Interest Rate Paths (CIR Model)')
plt.xlabel('Time')
plt.ylabel('Short Rate')
plt.legend()

# Plot bond prices
plt.subplot(2, 1, 2)
for i in range(num_paths):
    plt.plot(np.arange(num_steps + 1) * dt, bond_prices[i, :], label=f'Path {i + 1}')
plt.title('Simulated Bond Prices (CIR Model)')
plt.xlabel('Time')
plt.ylabel('Bond Price')
plt.legend()

plt.tight_layout()
plt.show()
