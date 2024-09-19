import casadi as ca
import numpy as np

def quaternion_update_ca(quat, omegas, dt):
        
# Closed-Form quaternion update (https://ahrs.readthedocs.io/en/latest/filters/angular.html)
# Expanded Skew-Symmetric Omega Matrix
    Omega = ca.vertcat(
        ca.horzcat(0, -omegas[0], -omegas[1], -omegas[2]),
        ca.horzcat(omegas[0], 0, omegas[2], -omegas[1]),
        ca.horzcat(omegas[1], -omegas[2], 0, omegas[0]),
        ca.horzcat(omegas[2], omegas[1], -omegas[0], 0)
    )

    next_quat = ca.mtimes(ca.cos(0.5 * ca.norm_2(omegas) * dt) * ca.SX.eye(4) + (1/ca.norm_2(omegas)) * ca.sin(0.5 * ca.norm_2(omegas) * dt) * Omega, quat)
    next_quat = next_quat/ca.norm_2(next_quat)
    return next_quat

def quaternion_update_np(quat, omegas, dt):
    omega1, omega2, omega3 = omegas
        
# Closed-Form quaternion update (https://ahrs.readthedocs.io/en/latest/filters/angular.html)
# Expanded Skew-Symmetric Omega Matrix
    Omega = np.array([[0, -omega1, -omega2, -omega3],
                      [omega1, 0, omega3, -omega2],
                      [omega2, -omega3, 0, omega1],
                      [omega3, omega2, -omega1, 0]])
    
    next_quat = (np.cos(0.5 * np.linalg.norm(omegas) * dt) * np.eye(4) + (1/np.linalg.norm(omegas)) * np.sin(0.5 * np.linalg.norm(omegas) * dt) * Omega) @ quat
    next_quat = next_quat/np.linalg.norm(next_quat)
    return next_quat