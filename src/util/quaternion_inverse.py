import casadi as ca

def quaternion_inverse(q):
    assert q.shape == (4, 1), "Quaternion must be a 4x1 CasADi symbolic matrix"
    w, x, y, z = q[0], q[1], q[2], q[3]
    norm_squared = w**2 + x**2 + y**2 + z**2
    q_inv = ca.vertcat(w, -x, -y, -z) / norm_squared
    
    return q_inv

