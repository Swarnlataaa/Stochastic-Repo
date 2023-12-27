import numpy as np
import matplotlib.pyplot as plt

def generate_brownian_motion(num_steps, dt):
    """
    Generate Brownian motion paths.

    Parameters:
    - num_steps: Number of time steps
    - dt: Time step size

    Returns:
    - Array of Brownian motion paths
    """
    t = np.arange(0, num_steps + 1) * dt
    dW = np.sqrt(dt) * np.random.normal(size=num_steps)
    W = np.cumsum(dW)
    return t, W

def plot_brownian_motion(paths):
    """
    Plot Brownian motion paths.

    Parameters:
    - paths: List of Brownian motion paths to be plotted
    """
    for path in paths:
        plt.plot(path[0], path[1])

    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Brownian Motion Paths')
    plt.grid(True)
    plt.show()

# Example usage:
num_paths = 5
num_steps = 1000
dt = 0.01

brownian_paths = [generate_brownian_motion(num_steps, dt) for _ in range(num_paths)]
plot_brownian_motion(brownian_paths)
