import casadi as ca

f_thruster = 1
M = 1
rd = 1
J = 1
dt = 1

# Define state variables
r_x = ca.SX.sym('r_x')
r_x_dot = ca.SX.sym('r_x_dot')
r_y = ca.SX.sym('r_y')
r_y_dot = ca.SX.sym('r_y_dot')
theta = ca.SX.sym('theta')
theta_dot = ca.SX.sym('theta_dot')

# Define control variables
u_list = []
for i in range(8):
    u_list.append(ca.SX.sym(f'u{i}'))

# Combine control variables into a single matrix
u = ca.vertcat(*u_list)

# Combine state variables into a single vector
x = ca.vertcat(r_x, r_x_dot, r_y, r_y_dot, theta, theta_dot)


# Define the rotation matrix as a function of theta
rotmat = ca.vertcat(
    ca.horzcat(ca.cos(theta), -ca.sin(theta)),
    ca.horzcat(ca.sin(theta),  ca.cos(theta))
)

B = f_thruster*ca.vertcat(
    ca.horzcat(1/M, -1/M, 1/M, -1/M, 0, 0, 0, 0),
    ca.horzcat(0, 0, 0, 0, 1/M, -1/M, 1/M, -1/M),
    ca.horzcat(rd/J, -rd/J, -rd/J, rd/J, rd/J, -rd/J, -rd/J, rd/J)
)

# Define the dynamics
r_x_ddot = (rotmat @ B[0:2, :] @ u)[0]

r_y_ddot = (rotmat @ B[0:2, :] @ u)[1]
theta_ddot = (B[2, :] @ u)

# Combine dynamics into a single vector
x_dot = ca.vertcat(
    r_x_dot,
    r_x_ddot,
    r_y_dot,
    r_y_ddot,
    theta_dot,
    theta_ddot
)

# Define a function for the dynamics
f = ca.Function('f', [x, u], [x_dot])

# Discretise the dynamics using Euler integration
x_next = x + dt * f(x, u)
f_discrete = ca.Function('f_discrete', [x, u], [x_next])

res_1 = f_discrete([0, 0, 0, 0, 0, 0], [0.5, 0, 0.5, 0, 1, 0, 0, 0])
res_2 = f_discrete([1, 0, 2, 0.5, 0, 0.2], [0, 0.2, 0, 0.5, 0.3, 0.2, 0, 0])
res_3 = f_discrete([1, 0.5, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0])
print(res_1)
print(res_2)
print(res_3)