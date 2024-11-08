import numpy as np

def quaternion_to_rotation_matrix_numpy(q):
    q0, q1, q2, q3 = q

    # Construct the rotation matrix
    rotation_matrix = np.array([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
        [2*q1*q2 + 2*q0*q3, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 1 - 2*q1**2 - 2*q2**2]
    ])

    return rotation_matrix

import casadi as ca

def quaternion_to_rotation_matrix_casadi(q0, q1, q2, q3):

    # Create rotation matrix
    rotation_matrix_ca = ca.vertcat(
        ca.horzcat(1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2),
        ca.horzcat(2*q1*q2 + 2*q0*q3, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1),
        ca.horzcat(2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 1 - 2*q1**2 - 2*q2**2)
    )

    return rotation_matrix_ca

def pos_prime_rot_casadi(q0, q1, q2, q3, x, y, z):

    rotation_matrix_ca = ca.vertcat(
        ca.horzcat(1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2),
        ca.horzcat(2*q1*q2 + 2*q0*q3, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1),
        ca.horzcat(2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 1 - 2*q1**2 - 2*q2**2)
    )

    X_prime_rotated = ca.mtimes(rotation_matrix_ca, ca.vertcat(x, y, z))
    return X_prime_rotated

def pos_prime_rot_numpy(q, states):
    q0, q1, q2, q3 = q
    x, y, z = states

    rotation_matrix = np.array([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
        [2*q1*q2 + 2*q0*q3, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 1 - 2*q1**2 - 2*q2**2]
    ])

    X_prime_rotated = np.dot(rotation_matrix, states)
    return X_prime_rotated
