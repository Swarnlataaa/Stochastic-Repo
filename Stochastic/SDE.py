import numpy as np
import matplotlib.pyplot as plt

def euler_maruyama_sde(drift, diffusion, initial_value, time_points, dt):
    """
    Numerical solver for stochastic differential equations (SDEs) using the Euler-Maruyama method.

    Parameters:
    - drift: Function representing the deterministic part of the SDE (e.g., mu(t, X(t)))
    - diffusion: Function representing the stochastic part of the SDE (e.g., sigma(t, X(t)))
    - initial_value: Initial value of the process X(0)
    - time_points: Array of time points at which to compute the solution
    - dt: Time step size

    Returns:
    - Array of approximate solution values at each time point
    """
    num_steps = len(time_points) - 1
    X = np.zeros(num_steps + 1)
    X[0] = initial_value

    for i in range(num_steps):
        dW = np.random.normal(0, np.sqrt(dt))  # Wiener process increment
        X[i + 1] = X[i] + drift(time_points[i], X[i]) * dt + diffusion(time_points[i], X[i]) * dW

    return X

# Example usage:
def drift_function(t, x):
    return 0.1 * x  # Example drift function (mu(t, x) = 0.1x)

def diffusion_function(t, x):
    return 0.2 * x  # Example diffusion function (sigma(t, x) = 0.2x)

# Set up time points and parameters
t_start = 0
t_end = 1
num_time_points = 100
time_points = np.linspace(t_start, t_end, num_time_points)
dt = (t_end - t_start) / num_time_points

# Solve the SDE using the Euler-Maruyama method
initial_value = 1.0
solution = euler_maruyama_sde(drift_function, diffusion_function, initial_value, time_points, dt)

# Plot the results
plt.plot(time_points, solution, label='Numerical Solution')
plt.xlabel('Time')
plt.ylabel('SDE Solution')
plt.title('Euler-Maruyama Method for SDE')
plt.legend()
plt.show()
