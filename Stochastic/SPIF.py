import numpy as np
import matplotlib.pyplot as plt

def stochastic_harmonic_oscillator_trajectory(dt, num_steps, gamma, omega, initial_state):
    """
    Simulate the trajectory of a quantum harmonic oscillator with noise.

    Parameters:
    - dt: Time step
    - num_steps: Number of steps
    - gamma: Damping coefficient
    - omega: Angular frequency
    - initial_state: Initial state [x, p]

    Returns:
    - time_points: Array of time points
    - positions: Array of positions
    """
    time_points = np.arange(0, num_steps * dt, dt)
    positions = np.zeros(num_steps)
    
    x, p = initial_state

    for i in range(num_steps):
        # Stochastic differential equations
        dW = np.random.normal(0, np.sqrt(dt))
        dx = p * dt
        dp = (-gamma * p - omega**2 * x) * dt + np.sqrt(2 * gamma) * dW

        x += dx
        p += dp

        positions[i] = x

    return time_points, positions

# Example usage
dt = 0.01
num_steps = 1000
gamma = 0.1  # Damping coefficient
omega = 1.0  # Angular frequency
initial_state = [1.0, 0.0]  # Initial position and momentum

time_points, positions = stochastic_harmonic_oscillator_trajectory(dt, num_steps, gamma, omega, initial_state)

# Visualization
plt.plot(time_points, positions)
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Stochastic Harmonic Oscillator Trajectory')
plt.show()
