from casadi import *
from ahrs.filters import AngularRate

mass = 1
mass = 1
Ixx, Iyy, Izz = 1, 1, 1
I = np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
f_thruster = 1
dx, dy = 1, 1
u_min = 0
u_max = 1
dt=1

x = MX.sym('x')
y = MX.sym('y')
z = MX.sym('x')
x_dot = MX.sym('x_dot')
y_dot = MX.sym('y_dot')
z_dot = MX.sym('z_dot')
q0 = MX.sym('q0')
q1 = MX.sym('q1')
q2 = MX.sym('q2')
q3 = MX.sym('q3')
omega1 = MX.sym('omega1')
omega2 = MX.sym('omega2')
omega3 = MX.sym('omega3')

# State Vector Concatenation
states = vertcat(x, y, z, x_dot, y_dot, z_dot, q0, q1, q2, q3, omega1, omega2, omega3)
n = states.size1()

# Control Variables Initialization

u = MX.sym('u', 8)
controls = u
m = controls.size1()

# Quaternion Rotation Matrix
Rot = vertcat(
    horzcat(1-2*q2**2-2*q3**2, 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2),
    horzcat(2*q1*q2+2*q0*q3, 1-2*q1**2-2*q3**2, 2*q2*q3-2*q0*q1),
    horzcat(2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 1-2*q1**2-2*q2**2)
)

B_pos = vertcat(
    horzcat(1, -1, 1, -1, 0, 0, 0, 0),
    horzcat(0, 0, 0, 0, 1, -1, 1, -1),
    horzcat(0, 0, 0, 0, 0, 0, 0, 0)
)

# Model Equations
pos_ddot = (1/mass) * mtimes([Rot, B_pos, controls])
x_ddot = pos_ddot[0]
y_ddot = pos_ddot[1]
z_ddot = pos_ddot[2]

Omega = vertcat(
    horzcat(0, -omega1, -omega2, -omega3),
    horzcat(omega1, 0, omega3, -omega2),
    horzcat(omega2, -omega3, 0, omega1),
    horzcat(omega3, omega2, -omega1, 0)
)

quat = vertcat(q0, q1, q2, q3)
quat_dot = 0.5 * mtimes(Omega, quat)
q0_dot = quat_dot[0]
q1_dot = quat_dot[1]
q2_dot = quat_dot[2]
q3_dot = quat_dot[3]

T_matrix = vertcat(
    horzcat(0, 0, 0, 0, 0, 0, 0, 0),
    horzcat(0, 0, 0, 0, 0, 0, 0, 0),
    horzcat(dx, -dx, -dx, dx, dy, -dy, -dy, dy)
)

omegas = vertcat(omega1, omega2, omega3)
omega_tilde = vertcat(
    horzcat(0, -omega3, omega2),
    horzcat(omega3, 0, -omega1),
    horzcat(-omega2, omega1, 0)
)

omega_dot = mtimes(inv(I), mtimes(T_matrix, controls) - mtimes(omega_tilde,I,omegas))
#omega_dot = mtimes(inv(I), mtimes(T_matrix, controls) - mtimes([omegas.T, I, omegas]))
omega_dot = mtimes(inv(I), mtimes(T_matrix, controls))
omega1_dot = omega_dot[0]
omega2_dot = omega_dot[1]
omega3_dot = omega_dot[2]

dynamics = vertcat(x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot, q0_dot, q1_dot, q2_dot, q3_dot, omega1_dot, omega2_dot, omega3_dot)



# Initial and Target State 
x_ref = MX.sym('x_ref', 13)
x0 = MX.sym('x0', 13)

# Continuous Dynamics Function
f = Function('f', [states, controls], [dynamics])
## Discretization 

# Euler Forward
next_state = states + dt * f(states, controls)
F = Function('F', [states, controls], [next_state])

res = f(states, controls)
#next_quaternions = mtimes(exp(0.5 * Omega * dt),res[6:10])
next_quaternions = mtimes(expm1(0.5 * Omega), quat)
next_quaternions2 = mtimes(cos(0.5 * norm_2(omegas)) * MX.eye(4) + (0.1 * norm_2(omegas))*sin(0.5*norm_2(omegas))*Omega, quat)
F_quat = Function('F_quat', [states, controls], [next_quaternions])
F_quat2 = Function('F_quat2', [states, controls], [next_quaternions2])




res1 = F([1, 1, 0, 0, 0, 0, 1, 0, 0, 0.392699, 0, 0, 0.785398], [0,0,0,0,0,0,0,0])
res2 = F_quat([1,1,0,0,0,0,1,0,0,0,0,0,pi/4], [0,0,0,0,0,0,0,0])
res3 = f([1,1,0,0,0,0,1,0,0,0,0,0,1], [0,0,0,0,0,0,0,0])
res4 = F_quat2([1,1,0,0,0,0,1,0,0,0,0,0,pi/8], [0,0,0,0,0,0,0,0])


quat1 = res1[6:10]/(sqrt(sum1(res1[6:10]**2)))


quat2 = res4/norm_2(res4)
from ahrs.filters import AngularRate
gyro_data = [0,0,pi/8]


      # Allocation of quaternions
Q = [1.0, 0.0, 0.0, 0]         # Initial attitude as a quaternion
angular_rate = AngularRate()
Qnext = angular_rate.update(Q, gyro_data)

F_quat

print('QNEXT',Qnext)


print(res1)
print('RES2', res2)

print(res4)

print('QUAT1', quat1)
print('QUAT2asério', quat2)