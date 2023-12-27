import numpy as np
import matplotlib.pyplot as plt

def simulate_garch(p, q, omega, alpha, beta, T, num_paths):
    """
    Simulate a GARCH(p, q) model.

    Parameters:
    - p: Order of autoregressive conditional variances
    - q: Order of moving average conditional variances
    - omega: Constant term
    - alpha: List of coefficients for the lagged squared returns (length p)
    - beta: List of coefficients for the lagged conditional variances (length q)
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

            # Calculate the conditional variance
            conditional_variance_t = omega
            for j in range(p):
                if t - j > 0:
                    conditional_variance_t += alpha[j] * returns[i, t - j]**2

            for j in range(q):
                if t - j > 0:
                    conditional_variance_t += beta[j] * conditional_variances[i, t - j]

            conditional_variances[i, t] = conditional_variance_t
            returns[i, t] = np.sqrt(conditional_variance_t) * z_t

    return time_points, returns, conditional_variances

# Example usage for GARCH(1,1)
p = 1
q = 1
omega = 0.0001
alpha = [0.1]  # Coefficients for the squared returns
beta = [0.8]   # Coefficients for the conditional variances
T = 1.0
num_paths = 5

time_points, returns, conditional_variances = simulate_garch(p, q, omega, alpha, beta, T, num_paths)

# Plot the results
plt.figure(figsize=(12, 6))
for i in range(num_paths):
    plt.plot(time_points, returns[i], label=f'Path {i + 1}')

plt.title(f'GARCH({p},{q}) Simulation - Returns')
plt.xlabel('Time')
plt.ylabel('Returns')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
for i in range(num_paths):
    plt.plot(time_points, conditional_variances[i], label=f'Path {i + 1}')

plt.title(f'GARCH({p},{q}) Simulation - Conditional Variances')
plt.xlabel('Time')
plt.ylabel('Conditional Variances')
plt.legend()
plt.show()
