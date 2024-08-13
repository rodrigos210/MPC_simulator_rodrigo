import numpy as np

# Constants
mass = 1
f_thruster = 1
I = 1
d = 1

# Dynamics Function
def dynamics(states, controls):
    x, x_dot, y, y_dot, theta, theta_dot = states

    u = controls

    x_ddot = (np.cos(theta)/mass) * (u[0] - u[1] + u[2] - u[3]) - (np.sin(theta)/mass) * (u[4] - u[5] + u[6] - u[7])
    
    y_ddot = (np.sin(theta)/mass) * (u[0] - u[1] + u[2] - u[3]) + (np.cos(theta)/mass) * (u[4] - u[5] + u[6] - u[7])

    theta_ddot = d/I * (u[0] - u[1] - u[2] + u[3] + u[4] - u[5] - u[6] + u[7])

    dynamics = np.array([
        x_dot,
        x_ddot,
        y_dot,
        y_ddot,
        theta_dot,
        theta_ddot
    ])

    return dynamics

# RK4 integration method
def rk4_step(states, controls, dt):
    k1 = dynamics(states, controls)
    k2 = dynamics(states + 0.5 * dt * k1, controls)
    k3 = dynamics(states + 0.5 * dt * k2, controls)
    k4 = dynamics(states + dt * k3, controls)
    x_next = states + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next
