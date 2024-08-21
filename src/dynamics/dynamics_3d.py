import numpy as np
from ahrs.filters import AngularRate 

# Constants
mass = 1
f_thuster = 1
Ixx, Iyy, Izz = 1, 1, 1
I = np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
dx = 1
dy = 1

# Dynamics Function
def dynamics(states, controls):
    x, y, z, x_dot, y_dot, z_dot, q0, q1, q2, q3, omega1, omega2, omega3 = states
    u = controls

    Rot = np.array([[1-2*q2**2-2*q3**2, 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2], 
                    [2*q1*q2+2*q0*q3, 1-2*q1**2-2*q3**2, 2*q2*q3-2*q0*q1], 
                    [2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 1-2*q1**2-2*q2**2]])
    
    B_pos = np.array([[1, -1, 1, -1, 0, 0, 0, 0], 
                      [0, 0, 0, 0, 1, -1, 1, -1], 
                      [0, 0, 0, 0, 0, 0, 0, 0]])

    pos_ddot = (1/mass) * (Rot @ B_pos @ controls)
    x_ddot = pos_ddot[0]
    y_ddot = pos_ddot[1]
    z_ddot = pos_ddot[2]

    Omega = np.array([[0, -omega1, -omega2, -omega3],
                      [omega1, 0, omega3, -omega2],
                      [omega2, -omega3, 0, omega1],
                      [omega3, omega2, -omega1, 0]])
    
    quat = np.array([q0, q1, q2, q3])
    quat_dot = 0.5 * (Omega @ quat)
    q0_dot, q1_dot, q2_dot, q3_dot = quat_dot

    T_matrix = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [dx, -dx, -dx, dx, dy, -dy, -dy, dy]])
    
    omegas = np.array([omega1, omega2, omega3])

    omega_tilde = np.array([[0, -omega3, omega2],[omega3, 0, -omega1], [-omega2, omega1, 0]])

    omega_dot = np.linalg.inv(I) @ (T_matrix @ controls - omega_tilde @ I @ omegas)

    omega1_dot = omega_dot[0]
    omega2_dot = omega_dot[1]
    omega3_dot = omega_dot[2]

    dynamics = np.array([x_dot, y_dot, z_dot, 
                         x_ddot, y_ddot, z_ddot, 
                         q0_dot, q1_dot, q2_dot, q3_dot, 
                         omega1_dot, omega2_dot, omega3_dot])

    return dynamics

# RK4 integration method
def rk4_step(states, controls, dt):
    k1 = dynamics(states, controls)
    k2 = dynamics(states + 0.5 * dt * k1, controls)
    k3 = dynamics(states + 0.5 * dt * k2, controls)
    k4 = dynamics(states + dt * k3, controls)
    x_next = states + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # omega1, omega2, omega3 = states[10:13]
    # Omega = np.array([[0, -omega1, -omega2, -omega3],
    #                   [omega1, 0, omega3, -omega2],
    #                   [omega2, -omega3, 0, omega1],
    #                   [omega3, omega2, -omega1, 0]])
    
    # angular_rate = AngularRate()
    #x_next[6:10] = angular_rate.update(states[6:10], states[10:13])

    # x_next = states + dt * k1
    x_next[6:10] = x_next[6:10] / np.linalg.norm(x_next[6:10])
    return x_next
