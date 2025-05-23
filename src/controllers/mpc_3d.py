from casadi import *
from src.util.quaternion_inverse import quaternion_inverse
from src.util.quaternion_multiplication import quaternion_product

class MPCController:
    def __init__(self, time_horizon, c_horizon, mass, I, dx, dy, dt, Q, R, P, u_min, u_max):
        
        self.p_horizon = p_horizon = int(time_horizon/dt)
        self.c_horizon = c_horizon
        self.u_min = u_min
        self.u_max = u_max

        # States Variables Initialization
        x = SX.sym('x')
        y = SX.sym('y')
        z = SX.sym('z')
        x_dot = SX.sym('x_dot')
        y_dot = SX.sym('y_dot')
        z_dot = SX.sym('z_dot')
        q0 = SX.sym('q0')
        q1 = SX.sym('q1')
        q2 = SX.sym('q2')
        q3 = SX.sym('q3')
        omega1 = SX.sym('omega1')
        omega2 = SX.sym('omega2')
        omega3 = SX.sym('omega3')

        # State Vector Concatenation
        states = vertcat(x, y, z, x_dot, y_dot, z_dot, q0, q1, q2, q3, omega1, omega2, omega3)
        self.n = states.size1()

        # Control Variables Initialization
        u = SX.sym('u', 8)
        controls = u
        self.m = controls.size1()

        # Quaternion Rotation Matrix
        Rot = vertcat(
            horzcat(1-2*q2**2-2*q3**2, 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2),
            horzcat(2*q1*q2+2*q0*q3, 1-2*q1**2-2*q3**2, 2*q2*q3-2*q0*q1),
            horzcat(2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 1-2*q1**2-2*q2**2)
        )

        # Input Matrix
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

        # Expanded Skew-Symmetric Omega Matrix
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

        # Torque Matrix
        T_matrix = vertcat(
            horzcat(0, 0, 0, 0, 0, 0, 0, 0),
            horzcat(0, 0, 0, 0, 0, 0, 0, 0),
            horzcat(dx, -dx, -dx, dx, dy, -dy, -dy, dy)
        )

        omegas = vertcat(omega1, omega2, omega3)

        # Omega Skew-Symmetric
        omega_tilde = vertcat(
            horzcat(0, -omega3, omega2),
            horzcat(omega3, 0, -omega1),
            horzcat(-omega2, omega1, 0)
        )

        omega_dot = mtimes(inv(I), mtimes(T_matrix, controls) - mtimes([omega_tilde,I,omegas]))
        omega1_dot = omega_dot[0]
        omega2_dot = omega_dot[1]
        omega3_dot = omega_dot[2]
        
        dynamics = vertcat(x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot, q0_dot, q1_dot, q2_dot, q3_dot, omega1_dot, omega2_dot, omega3_dot)

        # Initial and Target State 
        x_ref = SX.sym('x_ref', 13)
        x0 = SX.sym('x0', 13)

        # Quaternion Matrix to calculate its variation
        quat_A = vertcat(
            horzcat(x_ref[6], x_ref[7], x_ref[8], x_ref[9]),
            horzcat(-x_ref[7], x_ref[6], -x_ref[9], -x_ref[8]),
            horzcat(-x_ref[8], -x_ref[9], x_ref[6], x_ref[7]),
            horzcat(-x_ref[9], x_ref[8], -x_ref[7], x_ref[6]))

        # Continuous Dynamics Function
        f = Function('f', [states, controls], [dynamics])

        ## Discretization 
        # RK4
        # k1 = f(states, controls)
        # k2 = f(states + 0.5 * dt * k1, controls)
        # k3 = f(states + 0.5 * dt * k2, controls)
        # k4 = f(states + dt * k3, controls)
        # F = Function('F', [states, controls], [states + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)])

        # Euler Forward
        next_state = states + dt * f(states, controls)
        F = Function('F', [states, controls], [next_state])

        # Quaternions Update (not functional yet)
        next_quaternions = mtimes(cos(0.5 * norm_2(omegas) * dt) * SX.eye(4) + (1/norm_2(omegas)) * sin(0.5 * norm_2(omegas) * dt) * Omega, quat)
        # next_quaternions = mtimes(SX.eye(4) + 0.5 * Omega * dt, quat)
        F_quat = Function('F_quat', [states, controls], [next_quaternions])  

        U = SX.sym('U', self.m, self.c_horizon)
        X = x0 # Initial State
        J = 0 # Cost
        
        for k in range(self.p_horizon):

            U_k = U[:, min(k, c_horizon-1)]# From the control horizon, U_k is set as U_c_horizon

            pos_vel_delta = X[0:6] - x_ref[0:6] # Position and Velocity Deviation
            omega_delta = X[10:13] - x_ref[10:13] # Angular Rate Deviation
            #quat_err = X[6:10] - x_ref[6:10]
            #quat_err = quaternion_product(X[6:10], quaternion_inverse(x_ref[6:10]))
            quat_err = quaternion_product(x_ref[6:10], quaternion_inverse(X[6:10]))
            quat_delta = vertcat(1-quat_err[0], (norm_2(quat_err[1:])/9),(norm_2(quat_err[1:])/9),(norm_2(quat_err[1:])/9))
            f = Function('f', [x_ref], [x_ref])
            quat_dot_product = dot(X[6:10], x_ref[6:10])
            quat_dot_product_clamped = fmax(fmin(quat_dot_product, 1), -1)
            #quat_dot_product_clamped = fmin(fmax(quat_dot_product, 1), -1)
            

            #quat_delta = 2 * acos(fabs(quat_dot_product_clamped))
            quat_delta = 2 * pi - (2 * acos(quat_dot_product_clamped))

            
            #quat_err = mtimes(quat_A, vertcat(X[6:10])) # Quaternion Deviation

            x_delta = vertcat(pos_vel_delta, quat_delta, omega_delta)
            #J += mtimes([x_delta.T, Q, x_delta]) # State Deviation Cost
            J += quat_delta ** 2 * (Q[6,6] + Q[7,7] + Q[8,8] + Q[9,9])
          
            J += mtimes([pos_vel_delta.T, Q[0:6,0:6], pos_vel_delta])
            # J += (1-quat_err[0]) ** 2 * Q[6,6]
            #J += norm_2(quat_err[1:]) * Q[7:10,7:10]
            # J += sum1((quat_err[1:] ** 2) * diag(Q[7:10, 7:10]))
            #J += fmin(norm_2(x_ref[6:10]-X[6:10]), norm_2(x_ref[6:10]+X[6:10])) * Q[6,6]
            J += mtimes([omega_delta.T, Q[10:13, 10:13], omega_delta])

            J += mtimes([U_k.T, R, U_k]) # Input Cost
            
            X_next = F(X, U_k) # Next State Computation
            # X_next[6:10] = if_else(norm_2(omegas) == 0, X_next[6:10], F_quat(X, U_k))
            X_next[6:10] = F_quat(X, U_k)
            X_next[6:10] = X_next[6:10]/(norm_2(X_next[6:10]) + 1e-16) # Quaternions Normalization
            X = X_next # State Update
        
        # Terminal Cost
        quat_err_ter = quaternion_product(X[6:10], quaternion_inverse(x_ref[6:10]))
        J += mtimes([(X[0:6] - x_ref[0:6]).T, P[0:6,0:6], (X[0:6] - x_ref[0:6])])
        J += mtimes([quat_err_ter.T, P[6:10,6:10], quat_err_ter])
        J += mtimes([(X[10:13] - x_ref[10:13]).T, P[10:13,10:13], (X[10:13] - x_ref[10:13])])
        p = vertcat(x0,x_ref)
        
        # Solver Design
        nlp = {'x': reshape(U, -1, 1), 'f': J, 'p': p}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes', 'ipopt.max_iter': 100, 'ipopt.tol': 1e-6}

        # Solver Initialization
        self.solver = nlpsol('solver', 'ipopt', nlp, opts) # with IPOPT

    # Optimal Input Calculation
    def get_optimal_input(self, x0, x_ref, u_guess):

        # Cost Evolution Initialization
        cost_iter = []
        
        # Initial guess and bounds for the solver
        arg = {}
        arg["x0"] = u_guess
        arg["lbx"] = np.tile([self.u_min]*self.m, self.c_horizon)
        arg["ubx"] = np.tile([self.u_max]*self.m, self.c_horizon)
        arg["p"] = np.concatenate((x0, x_ref))
        
        # Solve the problem
        res = self.solver(**arg)
        u_opt = res['x'].full().reshape(self.c_horizon, self.m)
        cost_iter.append(float(res["f"]))

        return u_opt[0, :], cost_iter       