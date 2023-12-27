import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_particles = 5
num_steps = 1000
dt = 0.01
diffusion_coefficient = 0.1

# Function to simulate Brownian motion
def simulate_brownian_motion(num_particles, num_steps, dt, diffusion_coefficient):
    particles = np.zeros((num_particles, num_steps + 1))
    
    for i in range(1, num_steps + 1):
        dW = np.random.normal(0, np.sqrt(dt), size=num_particles)
        particles[:, i] = particles[:, i - 1] + np.sqrt(2 * diffusion_coefficient * dt) * dW
    
    return particles

# Simulate Brownian motion
particles = simulate_brownian_motion(num_particles, num_steps, dt, diffusion_coefficient)

# Plot the results
plt.figure(figsize=(10, 6))
for i in range(num_particles):
    plt.plot(np.arange(num_steps + 1) * dt, particles[i, :], label=f'Particle {i + 1}')

plt.title('Brownian Motion of Particles in a Fluid')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.show()
