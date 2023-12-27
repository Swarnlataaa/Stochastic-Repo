import numpy as np
import matplotlib.pyplot as plt

def simulate_garch11(omega, alpha, beta, T, num_paths):
    """
    Simulate a GARCH(1,1) model.

    Parameters:
    - omega: Constant term
    - alpha: Coefficient for the lagged squared returns
    - beta: Coefficient for the lagged conditional variance
    - T: Time to maturity
    - num_paths: Number of simulation paths

    Returns:
    - time_points: Array of time points
    - returns: Simulated returns
    - conditional_variances: Simulated conditional variances
    """
    num_steps = int(T * 252)  # Assuming daily data, 252 trading days in a year
    time_points = np.linspace(0, T, num_steps + 1)

    returns = np.zeros((num_paths, num_steps + 1))
    conditional_variances = np.zeros((num_paths, num_steps + 1))

    for i in range(num_paths):
        for t in range(1, num_steps + 1):
            z_t = np.random.normal(0, 1)
            conditional_variances[i, t] = omega + alpha * returns[i, t - 1]**2 + beta * conditional_variances[i, t - 1]
            returns[i, t] = np.sqrt(conditional_variances[i, t]) * z_t

    return time_points, returns, conditional_variances

# Example usage for GARCH(1,1)
omega = 0.0001
alpha = 0.1
beta = 0.8
T = 1.0
num_paths = 5

time_points, returns, conditional_variances = simulate_garch11(omega, alpha, beta, T, num_paths)

# Plot the results
plt.figure(figsize=(12, 6))
for i in range(num_paths):
    plt.plot(time_points, returns[i], label=f'Path {i + 1}')

plt.title('GARCH(1,1) Simulation - Returns')
plt.xlabel('Time')
plt.ylabel('Returns')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
for i in range(num_paths):
    plt.plot(time_points, conditional_variances[i], label=f'Path {i + 1}')

plt.title('GARCH(1,1) Simulation - Conditional Variances')
plt.xlabel('Time')
plt.ylabel('Conditional Variances')
plt.legend()
plt.show()
