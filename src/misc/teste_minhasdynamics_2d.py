from casadi import *

mass = 1
d=1
dt=1 
# States Variables Initialization
x = MX.sym('x')
x_dot = MX.sym('x_dot')
y = MX.sym('y')
y_dot = MX.sym('y_dot')
theta = MX.sym('theta')
theta_dot = MX.sym('theta_dot')

# State Vector Concatenation
states = vertcat(x, x_dot, y, y_dot, theta, theta_dot)
n = states.size1()

# Control Variables Initialization
u = []
for i in range(8):
    u.append(MX.sym(f'u{i}'))


controls = vertcat(*u)
m = controls.size2()

# Model Equations
x_ddot = (cos(theta)/mass) * (u[0] - u[1] + u[2] - u[3]) - (sin(theta)/mass) * (u[4] - u[5] + u[6] - u[7])


y_ddot = (sin(theta)/mass) * (u[0] - u[1] + u[2] - u[3]) + (cos(theta)/mass) * (u[4] - u[5] + u[6] - u[7])


theta_ddot = d * (u[0] - u[1] - u[2] + u[3] + u[4] - u[5] - u[6] + u[7])


dynamics = vertcat(
x_dot,
x_ddot,
y_dot,
y_ddot,
theta_dot,
theta_ddot
)

# Initial and Target State 
x_ref = MX.sym('x_ref', 6)
x0 = MX.sym('x0', 6)

# Continuous Dynamics Function
f = Function('f', [states, controls], [dynamics])

# Discretization 

# Euler Forward
next_state = states + dt * f(states, controls)
F = Function('F', [states, controls], [next_state])



## Range-Kutta 4 WRONGGGGG
# k1 = f(states, controls)
# k2 = f(states + 0.5 * dt * k1, controls)
# k3 = f(states + 0.5 * dt * k2, controls)
# k4 = f(states + dt * k3, controls)
# next_state = states + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
# F = Function('F', [states, controls], [next_state])


res1 = F([0, 0, 0, 0, 0, 0], [0.5, 0, 0.5, 0, 1, 0, 0, 0])
res2 = F([1, 0, 2, 0.5, 0, 0.2], [0, 0.2, 0, 0.5, 0.3, 0.2, 0, 0])

print(res1)
print(res2)