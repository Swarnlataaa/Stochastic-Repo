import numpy as np
import matplotlib.pyplot as plt

def harmonic_oscillator_action(x, x_dot, omega):
    """
    Action for a harmonic oscillator.

    Parameters:
    - x: Displacement
    - x_dot: Velocity
    - omega: Angular frequency

    Returns:
    - Action value
    """
    kinetic_term = 0.5 * x_dot**2
    potential_term = 0.5 * omega**2 * x**2
    action = kinetic_term - potential_term
    return action

def functional_integral(omega, num_points, dt):
    """
    Discretized functional integral for a harmonic oscillator.

    Parameters:
    - omega: Angular frequency
    - num_points: Number of discretization points
    - dt: Time step

    Returns:
    - Functional integral result
    """
    x_values = np.linspace(-5, 5, num_points)
    x_dot_values = np.linspace(-5, 5, num_points)

    integral_result = 0.0

    for x in x_values:
        for x_dot in x_dot_values:
            action = harmonic_oscillator_action(x, x_dot, omega)
            weight = np.exp(-action * dt)
            integral_result += weight

    integral_result *= (dt / np.pi)**0.5
    return integral_result

# Example usage
omega = 1.0
num_points = 100
dt = 0.1

result = functional_integral(omega, num_points, dt)

print(f"Functional integral result: {result}")

# Visualization (optional)
omega_values = np.linspace(0.1, 2.0, 20)
results = [functional_integral(omega_val, num_points, dt) for omega_val in omega_values]

plt.plot(omega_values, results, 'o-')
plt.xlabel('Angular Frequency (omega)')
plt.ylabel('Functional Integral Result')
plt.title('Functional Integral for a Harmonic Oscillator')
plt.show()
