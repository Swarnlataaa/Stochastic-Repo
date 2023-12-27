import numpy as np
import matplotlib.pyplot as plt

def action(phi, phi_prime, dx, dt, mass):
    """
    Action for a scalar field in 1+1 dimensions.

    Parameters:
    - phi: Field configuration at time t
    - phi_prime: Field configuration at time t + dt
    - dx: Spatial lattice spacing
    - dt: Temporal lattice spacing
    - mass: Mass of the field

    Returns:
    - Action value
    """
    kinetic_term = 0.5 * np.sum((phi_prime - phi)**2) / (dx**2 * dt**2)
    potential_term = 0.5 * mass**2 * np.sum(phi_prime**2) * dx**2 * dt
    return kinetic_term - potential_term

def metropolis_algorithm(phi, dx, dt, mass, num_steps):
    """
    Metropolis algorithm for a lattice QFT simulation.

    Parameters:
    - phi: Initial field configuration
    - dx: Spatial lattice spacing
    - dt: Temporal lattice spacing
    - mass: Mass of the field
    - num_steps: Number of Metropolis steps

    Returns:
    - List of field configurations over time
    """
    field_configs = [phi]

    for _ in range(num_steps):
        phi_prime = phi + np.random.normal(0, np.sqrt(dt), size=phi.shape)
        delta_action = action(phi, phi_prime, dx, dt, mass) - action(phi_prime, phi, dx, dt, mass)

        if np.random.rand() < np.exp(-delta_action):
            phi = phi_prime

        field_configs.append(phi)

    return field_configs

# Example usage
L = 10.0       # Spatial extent
T = 1.0        # Temporal extent
Nx = 50        # Number of spatial lattice points
Nt = 1000      # Number of temporal lattice points
dx = L / Nx    # Spatial lattice spacing
dt = T / Nt    # Temporal lattice spacing
mass = 0.1     # Mass of the field
num_steps = 500

# Initial field configuration (random)
phi_initial = np.random.normal(0, 1, size=(Nx,))

# Run Metropolis algorithm for lattice QFT simulation
field_configs = metropolis_algorithm(phi_initial, dx, dt, mass, num_steps)

# Visualization
plt.imshow(np.array(field_configs).T, extent=(0, T, 0, L), aspect='auto', cmap='viridis')
plt.colorbar(label='Field Value')
plt.xlabel('Time')
plt.ylabel('Spatial Position')
plt.title('Scalar Field Evolution in 1+1 Dimensions')
plt.show()
