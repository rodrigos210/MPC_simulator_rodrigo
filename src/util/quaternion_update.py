import casadi as ca

def next_quaternion(omegas, dt, Omega, quat):
    assert Omega.shape == (4, 4), "Omega must be a 4x4 CasADi symbolic matrix"
    assert quat.shape == (4, 1), "Quaternion must be a 4x1 CasADi symbolic matrix"
    
    omega_norm = ca.norm_2(omegas)
    

    cos_term = ca.cos(0.5 * omega_norm * dt)
    sin_term = ca.sin(0.5 * omega_norm * dt)
    
    # Handle the division by zero case for omega_norm
    # norm_factor = ca.if_else(omega_norm != 0, omega_norm / 10, 0)
    #rotation_matrix = ca.if_else(omega_norm == 0,ca.MX.eye(4), cos_term * ca.MX.eye(4) + omega_norm * sin_term * Omega)
    rotation_matrix = ca.if_else(omega_norm != 0, cos_term * ca.MX.eye(4) + omega_norm * sin_term * Omega, ca.MX.eye(4))

    next_quat = ca.mtimes(rotation_matrix, quat)
    # quat_update = ca.mtimes(rotation_matrix, quat)
    # next_quat = ca.if_else(omega_norm == 0, quat, quat_update)

    return next_quat
