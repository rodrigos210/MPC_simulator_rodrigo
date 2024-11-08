import numpy as np
import casadi as ca

def np_quaternion_to_euler(q):
    
    q = np.array(q)
    roll = np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    pitch = np.arcsin(2 * (q[0] * q[2] - q[3] * q[1]))
    yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    
    return [yaw, pitch, roll]

def ca_quaternion_to_euler(q):
    q = ca.SX(q)
    roll = ca.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    pitch = ca.asin(2 * (q[0] * q[2] - q[3] * q[1]))
    yaw = ca.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    
    return ca.vertcat(yaw, pitch, roll)
