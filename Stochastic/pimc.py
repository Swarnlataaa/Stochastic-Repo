import numpy as np
import matplotlib.pyplot as plt

def quantum_harmonic_oscillator_action(x, beta, m, omega):
    """
    Action for a quantum harmonic oscillator at finite temperature.

    Parameters:
    - x: Displacement
    - beta: Inverse temperature (1/kT)
    - m: Particle mass
    - omega: Angular frequency

    Returns:
    - Action value
    """
    kinetic_term = 0.5 * m * x**2
    potential_term = 0.5 * m * omega**2 * x**2
    action = (kinetic_term + potential_term) * beta
    return action

def metropolis_algorithm(x, beta, m, omega, num_steps):
    """
    Metropolis algorithm for Path Integral Monte Carlo simulation.

    Parameters:
    - x: Initial position
    - beta: Inverse temperature (1/kT)
    - m: Particle mass
    - omega: Angular frequency
    - num_steps: Number of Monte Carlo steps

    Returns:
    - List of sampled positions
    """
    positions = [x]

    for _ in range(num_steps):
        x_new = x + np.random.normal(0, np.sqrt(2 * np.pi * beta / m))
        delta_action = quantum_harmonic_oscillator_action(x_new, beta, m, omega) - \
                        quantum_harmonic_oscillator_action(x, beta, m, omega)

        if np.random.rand() < np.exp(-delta_action):
            x = x_new

        positions.append(x)

    return positions

# Example usage
beta = 1.0  # Inverse temperature
m = 1.0    # Particle mass
omega = 1.0  # Angular frequency
num_steps = 1000

initial_position = 0.0
sampled_positions = metropolis_algorithm(initial_position, beta, m, omega, num_steps)

# Visualization
plt.plot(sampled_positions)
plt.xlabel('Monte Carlo Steps')
plt.ylabel('Particle Position')
plt.title('Path Integral Monte Carlo for Quantum Harmonic Oscillator')
plt.show()
