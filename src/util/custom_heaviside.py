import casadi as ca

def custom_heaviside(x):
    return ca.if_else(x > 0, 1, 0)

