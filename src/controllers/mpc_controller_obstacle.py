from casadi import *
from src.util.quaternion_inverse import quaternion_inverse
from src.util.quaternion_update import quaternion_update_ca
from src.util.quaternion_rotation import quaternion_to_rotation_matrix_casadi
from src.util.quaternion_ahrs import quaternion_ahrs

class MPCController:
    def __init__(self, time_horizon, c_horizon, mass, I, dx, dy, dt, Q, R, P, u_min, u_max, x_obstacle, radius_obstacle, rho, radius_spacecraft, vertices):
        
        self.p_horizon = p_horizon = int(time_horizon/dt)
        self.c_horizon = c_horizon
        self.u_min = u_min
        self.u_max = u_max
        self.radius_obstacle = radius_obstacle
        self.radius_spacecraft = radius_spacecraft
        self.vertices = vertices

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

        # Slack variables initialization
        xi_obstacle = MX.sym('xi_obstacle', self.p_horizon)  # Obstacle Margin Scack Variable
        eta_target = MX.sym('eta_target', self.n) # Terminal Cost Position Slack Variable

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

        # # Discretization 
        # RK4 
        # k1 = f(states, controls)
        # k2 = f(states + 0.5 * dt * k1, controls)
        # k3 = f(states + 0.5 * dt * k2, controls)
        # k4 = f(states + dt * k3, controls)
        # next_states = states + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # F = Function('F', [states, controls], [next_states])
        #F = Function('F', [states, controls], [states + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)])

        # Euler Forward
        next_state = states + dt * f(states, controls)
        F = Function('F', [states, controls], [next_state])
        next_quaternions = quaternion_update_ca(states[6:10], states[10:13], dt)
        F_quat = Function('F_quat', [states, controls], [next_quaternions]) 


        U = MX.sym('U', self.m, self.c_horizon)
        X = x0  # Initial State
        J = 0 # Cost
        g = []  # constraints list

        for k in range(self.p_horizon):
            
            U_k = U[:, min(k, c_horizon-1)] # From the control horizon, U_k is set as U_c_horizon
            
            pos_delta = X[0:3] - x_obstacle[0:3] # Position Deviation
            distance_to_obstacle = norm_2(pos_delta) # Distance to the Obstacle
            
            effective_obstacle_radius = radius_obstacle + radius_spacecraft
            effective__margin_radius = radius_obstacle*2 +radius_spacecraft

            obstacle_margin_constraint = distance_to_obstacle - effective__margin_radius + xi_obstacle[k] # Outer Circle Constraint (SOFT)
            obstacle_constraint = distance_to_obstacle - effective_obstacle_radius # Obstacle Constraint (HARD)
            
            # Appending Constraints
            g.append(obstacle_constraint)
            g.append(obstacle_margin_constraint)
            
            pos_vel_delta = X[0:6] - x_ref[0:6] # Position and Velocity Deviation
            omega_delta = X[10:13] - x_ref[10:13] # Angular Rate Deviation
            quat_err = mtimes(quat_A, quaternion_inverse(X[6:10])) # Quaternion Deviation

            x_delta = vertcat(pos_vel_delta, quat_err, omega_delta)
            J += mtimes([x_delta.T, Q, x_delta]) # State Deviation Cost 
            J += mtimes([U_k.T, R, U_k]) # Input Cost 
            J += rho * xi_obstacle[k]**2 # Obstacle Margin Constraint Cost

            X_next = F(X, U_k) # Next State Computation
            X_next[6:10] = F_quat(X, U_k) #(Only working if norm of initial omegas aren't 0!!!)
            #X_next[6:10] = X_next[6:10]/norm_2(X_next[6:10]) # Quaternions Normalization
            X = X_next # State Update
        
        # Terminal Cost
        quat_err_ter = mtimes(quat_A, vertcat(X[6:10])) #
        J += mtimes([(X[0:6] - x_ref[0:6] + eta_target[0:6]).T, P[0:6,0:6], (X[0:6] - x_ref[0:6] + eta_target[0:6])])
        #J += mtimes([(X[0:6] - x_ref[0:6]).T, P[0:6,0:6], (X[0:6] - x_ref[0:6])])
        J += mtimes([(quat_err_ter + eta_target[6:10]).T , P[6:10,6:10], (quat_err_ter + eta_target[6:10])])
        J += mtimes([(X[10:13] - x_ref[10:13] + eta_target[10:13]).T, P[10:13,10:13], (X[10:13] - x_ref[10:13] + eta_target[10:13])])
        p = vertcat(x0,x_ref)

        # Solver Design
        nlp = {'x': vertcat(reshape(U, -1, 1), xi_obstacle, eta_target), 'f': J, 'g': vertcat(*g), 'p': p}
        #nlp = {'x': vertcat(reshape(U, -1, 1), xi_obstacle), 'f': J, 'g': vertcat(*g), 'p': p}

        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes', 'ipopt.max_iter': 100, 'ipopt.tol': 1e-4, 'ipopt.acceptable_tol': 1e-3, 'ipopt.constr_viol_tol': 1e-4}

        self.solver = nlpsol('solver', 'ipopt', nlp, opts) # Solver Initiation with IPOPT
        #self.solver = nlpsol('solver', 'sqpmethod', nlp) # Solver Initiation with SQP Method (NOT WORKING)

    # Optimal Input Calculation
    def get_optimal_input(self, x0, x_ref, u_guess):

        # Cost and Slack Variables Evolution Initialization
        cost_iter = []
        xi_optimal = []
        eta_optimal = []

        # Bounds Organization
        lbx_u = np.tile([self.u_min]*self.m, self.c_horizon)
        ubx_u = np.tile([self.u_max] * self.m, self.c_horizon)
        lbx_xi = [0] * self.p_horizon
        ubx_xi = [float(inf)] * self.p_horizon
        lbx_eta = [float(-inf)] * self.n
        ubx_eta = [float(inf)] * self.n

        # Solver Bounds, Parameters, and Initial States Definition
        arg = {}
        arg["x0"] = np.concatenate((u_guess.flatten(), np.zeros(self.p_horizon), np.zeros(self.n)))
        #arg["x0"] = np.concatenate((u_guess.flatten(), np.zeros(self.p_horizon))) # To test without \eta
        arg["lbx"] = np.concatenate((lbx_u,lbx_xi, lbx_eta))
        arg["ubx"] = np.concatenate((ubx_u,ubx_xi, ubx_eta))
        #arg["lbx"] = np.concatenate((lbx_u,lbx_xi)) # To test without \eta
        #arg["ubx"] = np.concatenate((ubx_u,ubx_xi)) # To test without \eta
        arg["lbg"] = [0, 0] * self.p_horizon
        arg["ubg"] = [float('inf'), float('inf')] * self.p_horizon  
        #arg["lbg"] = [0] * self.p_horizon  # To test without one of obstacle constraints
        #arg["ubg"] = [float('inf')] * self.p_horizon  # To test without one of the obstacle constraints
        arg["p"] = np.concatenate((x0, x_ref))
        
        # Solve the problem
        res = self.solver(**arg)
        u_opt = res['x'].full().reshape(-1)[:self.c_horizon*self.m].reshape(self.c_horizon, self.m)

        # Evolutions Track
        xi_optimal.append(res['x'].full().reshape(-1)[self.c_horizon*self.m:self.c_horizon * self.m + self.p_horizon])
        eta_optimal.append(res['x'].full().reshape(-1)[self.c_horizon * self.m + self.p_horizon:])
        cost_iter.append(float(res["f"]))

        return u_opt, xi_optimal, eta_optimal, cost_iter
