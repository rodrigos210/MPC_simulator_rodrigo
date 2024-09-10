from casadi import *
from src.util.quaternion_inverse import quaternion_inverse

class MPCController:
    def __init__(self, time_horizon, c_horizon, mass, I, dx, dy, dt, Q, R, P, u_min, u_max):
        
        self.p_horizon = p_horizon = int(time_horizon/dt)
        self.c_horizon = c_horizon
        self.u_min = u_min
        self.u_max = u_max

        self.cone_apex = MX.sym('cone_apex', 3)  # Target position
        self.cone_axis = MX.sym('cone_axis', 3)  # Cone axis direction
        self.cone_angle = MX.sym('cone_angle')  # Cone angle in radians
        
        # Convert cone angle to cosine value
        self.cone_cos_angle = cos(self.cone_angle)


        # States Variables Initialization
        x = MX.sym('x')
        y = MX.sym('y')
        z = MX.sym('z')
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
        self.n = states.size1()

        # Control Variables Initialization
        u = MX.sym('u', 8)
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
        x_ref = MX.sym('x_ref', 13)
        x0 = MX.sym('x0', 13)

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
        next_quaternions = mtimes(cos(0.5 * norm_2(omegas) * dt) * MX.eye(4) + (1/norm_2(omegas)) * sin(0.5 * norm_2(omegas) * dt) * Omega, quat)
        # next_quaternions = mtimes(MX.eye(4) + 0.5 * Omega * dt, quat)
        F_quat = Function('F_quat', [states, controls], [next_quaternions])  

        U = MX.sym('U', self.m, self.c_horizon)
        X = x0 # Initial State
        J = 0 # Cost
        
        for k in range(self.p_horizon):

            U_k = U[:, min(k, c_horizon-1)]# From the control horizon, U_k is set as U_c_horizon

            pos_vel_delta = X[0:6] - x_ref[0:6] # Position and Velocity Deviation
            omega_delta = X[10:13] - x_ref[10:13] # Angular Rate Deviation
            quat_err = mtimes(quat_A, vertcat(X[6:10])) # Quaternion Deviation

            x_delta = vertcat(pos_vel_delta, quat_err, omega_delta)
            J += mtimes([x_delta.T, Q, x_delta]) # State Deviation Cost
            J += mtimes([U_k.T, R, U_k]) # Input Cost
            
            X_next = F(X, U_k) # Next State Computation
            # X_next[6:10] = if_else(norm_2(omegas) == 0, X_next[6:10], F_quat(X, U_k))
            quat_next = F_quat(X, U_k)
            X_next[6:10] = X_next[6:10]/norm_2(X_next[6:10]) # Quaternions Normalization
            X = X_next # State Update
        
        # Terminal Cost
        quat_err_ter = mtimes(quat_A, vertcat(X[6:10]))
        J += mtimes([(X[0:6] - x_ref[0:6]).T, P[0:6,0:6], (X[0:6] - x_ref[0:6])])
        J += mtimes([quat_err_ter.T, P[6:10,6:10], quat_err_ter])
        J += mtimes([(X[10:13] - x_ref[10:13]).T, P[10:13,10:13], (X[10:13] - x_ref[10:13])])
        p = vertcat(x0,x_ref, self.cone_apex, self.cone_axis, self.cone_angle)


        # Include cone constraint in the cost function or constraints
        spacecraft_position = x0[:3]  # Assume spacecraft position is in the first 3 elements of x0
        cone_constraint = self.compute_cone_constraint(spacecraft_position)
        
        # Add the constraint to the problem
        # Modify this part based on your specific solver requirements

        # Example constraint setup (depends on solver specifics)
        constraints = {'g': cone_constraint}
        nlp = {'x': reshape(U, -1, 1), 'f': J, 'p': p, **constraints}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes', 'ipopt.max_iter': 100, 'ipopt.tol': 1e-6}

        # Solver Initialization
        self.solver = nlpsol('solver', 'ipopt', nlp, opts)

    def compute_cone_constraint(self, spacecraft_position):
        # Vector from spacecraft to target
        v = self.cone_apex - spacecraft_position

        # Normalize the vector
        v_norm = v / norm_2(v)

        # Dot product with cone axis
        dot_product = mtimes(v_norm.T, self.cone_axis)

        # Constraint: dot_product >= cos(cone_angle)
        constraint = dot_product - self.cone_cos_angle

        return constraint

    # Optimal Input Calculation
    def get_optimal_input(self, x0, x_ref, u_guess, cone_apex, cone_axis, cone_angle):

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