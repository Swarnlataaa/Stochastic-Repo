import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define parameters
T = 1  # Time horizon
dt = 0.01  # Time step
num_steps = int(T / dt)  # Number of time steps
gamma = 0.05  # Discount factor

# System dynamics
def dynamics(state, control):
    return state + control + np.random.normal(0, 1)

# Cost function
def cost(state, control):
    return state**2 + control**2

# Value function and HJB equation
def value_function(state, time, value_function_params):
    control_space = np.linspace(-5, 5, 100)
    objective = lambda u: cost(state, u) + gamma * np.interp(dynamics(state, u), value_function_params['state_values'], value_function_params['values'])
    result = minimize(objective, 0)
    optimal_control = result.x
    return cost(state, optimal_control) + gamma * np.interp(dynamics(state, optimal_control), value_function_params['state_values'], value_function_params['values'])

# Discretize state space
state_values = np.linspace(-10, 10, 100)

# Initialize value function
values = np.zeros((num_steps + 1, len(state_values)))

# Backward recursion to solve HJB equation
for t in range(num_steps, 0, -1):
    for i, state in enumerate(state_values):
        values[t-1, i] = value_function(state, (t-1) * dt, {'state_values': state_values, 'values': values[t, :]})

# Plot optimal control policy at time t=0
optimal_controls = np.zeros_like(state_values)
for i, state in enumerate(state_values):
    objective = lambda u: cost(state, u) + gamma * np.interp(dynamics(state, u), state_values, values[0, :])
    result = minimize(objective, 0)
    optimal_controls[i] = result.x

# Plot results
plt.figure(figsize=(12, 6))

# Plot value function
plt.subplot(1, 2, 1)
plt.plot(state_values, values[0, :], label='t=0')
plt.plot(state_values, values[num_steps//2, :], label=f't={T/2}')
plt.plot(state_values, values[-1, :], label=f't={T}')
plt.title('Value Function')
plt.xlabel('State')
plt.ylabel('Value')
plt.legend()

# Plot optimal control policy
plt.subplot(1, 2, 2)
plt.plot(state_values, optimal_controls)
plt.title('Optimal Control Policy')
plt.xlabel('State')
plt.ylabel('Control')

plt.show()
