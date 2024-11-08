## Custom Heaviside Function
# to avoid casadi heaviside function H(0) = 1/2 which is not ideal for this situation
import casadi as ca

def custom_heaviside(x):
    return ca.if_else(x > 0, 0, 1.00)

