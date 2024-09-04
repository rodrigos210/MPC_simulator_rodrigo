import casadi as ca

def ca_quaternion_inverse(q0, q1, q2, q3):
    q1 = -q1
    q2 = -q2
    q3 = -q3

    quat_num = ca.vertcat(q0,q1,q2,q3)
    quat_den = ca.norm_2(quat_num)

    return quat_num/quat_den