import numpy as np
import matplotlib.pyplot as plt

def classical_action(x, x_dot, potential_function, dt):
    """
    Classical action for a particle.

    Parameters:
    - x: Displacement
    - x_dot: Velocity
    - potential_function: Potential energy function
    - dt: Time step

    Returns:
    - Action value
    """
    kinetic_term = 0.5 * x_dot**2
    potential_term = potential_function(x)
    action = (kinetic_term - potential_term) * dt
    return action

def path_integral_quantization(x_initial, x_final, potential_function, num_steps):
    """
    Path integral quantization for a one-dimensional quantum system.

    Parameters:
    - x_initial: Initial position
    - x_final: Final position
    - potential_function: Potential energy function
    - num_steps: Number of discretization steps

    Returns:
    - Transition amplitude
    """
    dt = (x_final - x_initial) / num_steps
    x_values = np.linspace(x_initial, x_final, num_steps + 1)

    amplitude = 0.0

    for i in range(num_steps):
        x = x_values[i]
        x_next = x_values[i + 1]
        x_midpoint = 0.5 * (x + x_next)
        x_dot = (x_next - x) / dt

        action = classical_action(x_midpoint, x_dot, potential_function, dt)
        amplitude += np.exp(1j * action)

    return amplitude

# Example usage
def harmonic_oscillator_potential(x):
    """Harmonic oscillator potential function."""
    return 0.5 * x**2

x_initial = -2.0
x_final = 2.0
num_steps = 1000

transition_amplitude = path_integral_quantization(x_initial, x_final, harmonic_oscillator_potential, num_steps)

print(f"Transition amplitude: {transition_amplitude}")

# Visualization (optional)
x_values = np.linspace(x_initial, x_final, num_steps + 1)
probability_density = np.abs(transition_amplitude)**2
plt.plot(x_values, probability_density)
plt.xlabel('Position')
plt.ylabel('Probability Density')
plt.title('Path Integral Quantization for Harmonic Oscillator')
plt.show()
