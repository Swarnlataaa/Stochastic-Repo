import numpy as np
import matplotlib.pyplot as plt

def path_integral_control(x_initial, x_target, num_steps, dt, Q, R):
    """
    Path Integral Control to solve an optimal control problem.

    Parameters:
    - x_initial: Initial state
    - x_target: Target state
    - num_steps: Number of time steps
    - dt: Time step
    - Q: State cost matrix
    - R: Control cost matrix

    Returns:
    - u_optimal: Optimal control sequence
    """
    num_states = len(x_initial)
    num_controls = 1  # For simplicity, consider a single control variable

    # Initialize control sequence
    u_optimal = np.random.randn(num_steps, num_controls)

    # Discretized dynamics model (in this example, a simple first-order system)
    A = np.eye(num_states)  # Identity matrix
    B = dt * np.eye(num_controls)  # For simplicity, assume a direct mapping from control to state change

    # Iterative path integral control
    for _ in range(100):  # Number of iterations
        x = x_initial
        for t in range(num_steps):
            x += np.dot(A, x) + np.dot(B, u_optimal[t])

        # Compute cost
        cost = np.sum((x - x_target)**2)

        # Compute control gradients
        du = np.zeros_like(u_optimal)
        for t in range(num_steps):
            du[t] = -2 * np.dot(np.dot(B.T, Q), x - x_target)

        # Update control sequence using gradient descent
        u_optimal -= 0.1 * du

    return u_optimal

# Example usage
np.random.seed(42)

# System dynamics
A = np.array([[1]])  # Simple first-order system
B = np.array([[1]])  # Direct mapping from control to state change

# Initial and target states
x_initial = np.array([0])
x_target = np.array([10])

# Cost matrices
Q = np.eye(len(x_initial))  # State cost matrix
R = np.eye(1)  # Control cost matrix

# Parameters
num_steps = 50
dt = 0.1

# Solve optimal control problem using Path Integral Control
u_optimal = path_integral_control(x_initial, x_target, num_steps, dt, Q, R)

# Visualization
time_points = np.arange(0, num_steps * dt, dt)
plt.plot(time_points, u_optimal, label='Optimal Control')
plt.xlabel('Time')
plt.ylabel('Control Input')
plt.legend()
plt.title('Path Integral Control for Optimal Control')
plt.show()
