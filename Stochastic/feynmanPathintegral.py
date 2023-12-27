import numpy as np
import matplotlib.pyplot as plt

def feynman_path_integral(x_final, x_initial, num_steps):
    """
    Discrete Feynman Path Integral for a free particle in one dimension.

    Parameters:
    - x_final: Final position
    - x_initial: Initial position
    - num_steps: Number of discrete steps

    Returns:
    - Probability amplitude for the particle to travel from x_initial to x_final
    """
    delta_x = (x_final - x_initial) / num_steps
    paths = np.arange(x_initial, x_final, delta_x)

    path_integral = np.exp(1j * np.sum(paths))  # Simplified complex exponential term

    return path_integral

# Example usage
x_initial = 0.0
x_final = 1.0
num_steps = 100

result = feynman_path_integral(x_final, x_initial, num_steps)

print(f"Probability amplitude: {result}")

# Visualization (optional)
plt.plot(np.linspace(x_initial, x_final, num_steps), np.ones(num_steps), 'o-', label='Path')
plt.xlabel('Position')
plt.ylabel('Amplitude')
plt.title('Feynman Path Integral')
plt.legend()
plt.show()
