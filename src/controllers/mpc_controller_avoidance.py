from casadi import *
from src.util.quaternion_inverse import quaternion_inverse
from src.util.quaternion_update import quaternion_update_ca

class MPCController:
    def __init__(self, time_horizon, c_horizon, mass, I, dx, dy, dt, Q, R, P, u_min, u_max, entry_radius, rho):
        
        self.p_horizon = p_horizon = int(time_horizon/dt)
        self.c_horizon = c_horizon
        self.u_min = u_min
        self.u_max = u_max
        self.entry_radius = entry_radius


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
        xi1 = SX.sym('xi', p_horizon) # slack variable
        xi2 = SX.sym('xi', p_horizon) # slack variable
        eta_target = SX.sym('eta_target', self.n) # Terminal Cost Position Slack Variable

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
        # next_state = states + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # F = Function('F', [states, controls], [next_state])

        # # Euler Forward
        next_state = states + dt * f(states, controls)
        F = Function('F', [states, controls], [next_state])
        next_quaternions = quaternion_update_ca(states[6:10], states[10:13], dt)
        F_quat = Function('F_quat', [states, controls], [next_quaternions]) 

        # # Quaternions Update (not functional yet)
        # next_quaternions = mtimes(cos(0.5 * norm_2(omegas) * dt) * SX.eye(4) + (1/(norm_2(omegas) + 1e-8)) * sin(0.5 * norm_2(omegas) * dt) * Omega, quat)
        # # next_quaternions = mtimes(SX.eye(4) + 0.5 * Omega * dt, quat)
        # F_quat = Function('F_quat', [states, controls], [next_quaternions])  

        U = SX.sym('U', self.m, self.c_horizon)
        X = x0 # Initial State
        J = 0 # Cost
        g = []
        for k in range(self.p_horizon):

            U_k = U[:, min(k, c_horizon-1)]# From the control horizon, U_k is set as U_c_horizon

            pos_vel_delta = X[0:6] - x_ref[0:6] # Position and Velocity Deviation
            omega_delta = X[10:13] - x_ref[10:13] # Angular Rate Deviation
            quat_err = mtimes(quat_A, quaternion_inverse((X[6:10]))) # Quaternion Deviation

            x_delta = vertcat(pos_vel_delta, quat_err, omega_delta)

            vel_sc = X[3:6]
            rot_states = vertcat(
            horzcat(1-2*X[8]**2-2*X[9]**2, 2*X[7]*X[8]-2*X[6]*X[9], 2*X[7]*X[9]+2*X[6]*X[8]),
            horzcat(2*X[7]*X[8]+2*X[6]*X[9], 1-2*X[7]**2-2*X[9]**2, 2*X[8]*X[9]-2*X[6]*X[7]),
            horzcat(2*X[7]*X[9]-2*X[6]*X[8], 2*X[8]*X[9]+2*X[6]*X[7], 1-2*X[7]**2-2*X[8]**2)
            )

            pos_delta = X[0:3] - x_ref[0:3] # Position Deviation
            distance_to_obstacle = norm_2(pos_delta) # Distance to the Obstacle

        
            vel_sc_inertial = mtimes(rot_states, vel_sc)

            #y_constraint = X[1] - x_ref[1] + xi[k]
            slope = (X[1] - x_ref[1])/(X[0] - x_ref[0])
            #angle_constraint = cone_angle - atan(slope) + xi[k]
            angle = atan(slope)
            #angle_constraint = sin(cone_angle) - sin(angle) + xi[k]

            #entry_constraint = if_else(distance_to_obstacle > entry_radius, 0, X[1] - fabs(X[0] - x_ref[0]) - (9.8) + xi[k])
           # entry_constraint = if_else(distance_to_obstacle > entry_radius, 0, X[1] - fabs(X[0] - x_ref[0]) - (9.8))
            #entry_constraint = if_else(distance_to_obstacle > entry_radius, 0, X[1] - fabs(X[0] - x_ref[0]) - (x_ref[1]) + xi[k])
            #entry_constraint = X[1] - fabs(X[0] - x_ref[0]) - 9.8
            #entry_constraint1 = (X[1] - (X[0] - x_ref[0]) - 9.8) * heaviside(distance_to_obstacle - entry_radius)
            #entry_constraint1 = (distance_to_obstacle - entry_radius) * heaviside(X[1] - (X[0] - x_ref[0]) - 9.8) 
            #entry_constraint2 = (distance_to_obstacle - entry_radius) * heaviside(X[1] + (X[0] - x_ref[0]) - 9.8)
            #entry_constraint1 = (X[1] - (X[0] - x_ref[0]) - 9.8 + xi1[k]) * heaviside(entry_radius - distance_to_obstacle)
            #entry_constraint2 = (X[1] + (X[0] - x_ref[0]) - 9.8) * heaviside(distance_to_obstacle - entry_radius)
            #entry_constraint2 = (X[1] + (X[0] - x_ref[0]) - 9.8 + xi2[k]) * heaviside(entry_radius - distance_to_obstacle)
            #entry_constraint1 = (X[1] - (X[0] - x_ref[0]) - 9.8) 
            #entry_constraint2 = (X[1] + (X[0] - x_ref[0]) - 9.8) 
            entry_constraint1 = if_else(X[1] - (X[0] - x_ref[0]) - 9.8 > 0, 0, (distance_to_obstacle - entry_radius))
            entry_constraint2 = if_else(X[1] + (X[0] - x_ref[0]) - 9.8 > 0, 0, (distance_to_obstacle - entry_radius))
            g.append(entry_constraint1)
            g.append(entry_constraint2)
            #g.append(angle_constraint)

            J += mtimes([x_delta.T, Q, x_delta]) # State Deviation Cost
            J += mtimes([U_k.T, R, U_k]) # Input Cost
            #J += rho * (xi1[k]**2 + xi2[k]**2)
            
            X_next = F(X, U_k) # Next State Computation
            X_next[6:10] = F_quat(X, U_k)
            X_next[6:10] = X_next[6:10]/(norm_2(X_next[6:10])) # Quaternions Normalization
            X = X_next # State Update
    
        # Terminal Cost
        quat_err_ter = mtimes(quat_A, vertcat(X[6:10]))
        # J += mtimes([(X[0:6] - x_ref[0:6] + eta_target[0:6]).T, P[0:6,0:6], (X[0:6] - x_ref[0:6] + eta_target[0:6])])
        # J += mtimes([(quat_err_ter + eta_target[6:10]).T, P[6:10,6:10], (quat_err_ter + eta_target[6:10])])
        # J += mtimes([(X[10:13] - x_ref[10:13] + eta_target[10:13]).T, P[10:13,10:13], (X[10:13] - x_ref[10:13] + eta_target[10:13])])
        p = vertcat(x0,x_ref)

        # Without ETA
        J += mtimes([(X[0:6] - x_ref[0:6]).T, P[0:6,0:6], (X[0:6] - x_ref[0:6])])
        J += mtimes([(quat_err_ter).T , P[6:10,6:10], quat_err_ter])
        J += mtimes([(X[10:13] - x_ref[10:13]).T, P[10:13,10:13], (X[10:13] - x_ref[10:13])])
        
        # Solver Design
        #nlp = {'x': vertcat(reshape(U, -1, 1), eta_target), 'f': J, 'p': p, 'g': vertcat(*g)}
        #nlp = {'x': vertcat(reshape(U, -1, 1)), 'f': J, 'p': p, 'g': vertcat(*g)}
        nlp = {'x': vertcat(reshape(U, -1, 1)), 'f': J, 'p': p, 'g': vertcat(*g)}
        opts = {'ipopt.print_level': 0, 
                'print_time': 0, 
                'ipopt.sb': 'yes', 
                'ipopt.max_iter': 100, 
                'ipopt.tol': 1e-3, 
                'ipopt.constr_viol_tol': 1e-1,
                'ipopt.acceptable_constr_viol_tol': 1e-1, 
                'ipopt.honor_original_bounds' : 'yes',
                'ipopt.bound_relax_factor': 0}
        
        # s_opts = {"max_cpu_time": 0.1, 
		# 		  "print_level": 0, 
		# 		  "tol": 5e-1, 
		# 		  "dual_inf_tol": 5.0, 
		# 		  "constr_viol_tol": 1e-1,
		# 		  "compl_inf_tol": 1e-1, 
		# 		  "acceptable_tol": 1e-2, 
		# 		  "acceptable_constr_viol_tol": 0.01, 
		# 		  "acceptable_dual_inf_tol": 1e10,
		# 		  "acceptable_compl_inf_tol": 0.01,
		# 		  "acceptable_obj_change_tol": 1e20,
		# 		  "diverging_iterates_tol": 1e20}

        # Solver Initialization
        self.solver = nlpsol('solver', 'ipopt', nlp, opts) # with IPOPT

    # Optimal Input Calculation
    def get_optimal_input(self, x0, x_ref, u_guess):

        # Cost Evolution Initialization
        cost_iter = []
        constraint_iter = []
        xi_optimal = []
        lbx_u = np.tile([self.u_min]*self.m, self.c_horizon)
        ubx_u = np.tile([self.u_max]*self.m, self.c_horizon)
        #lbx_xi = [0] * self.p_horizon
        #ubx_xi = [float('inf')] * self.p_horizon
        #lbx_xi = [0, 0] * self.p_horizon
        #ubx_xi = [float('inf'), float('inf')] * self.p_horizon
        #lbx_eta = [float('-inf')] * self.n
        #ubx_eta = [float('inf')] * self.n
        
        
        # Initial guess and bounds for the solver
        arg = {}
        #arg["x0"] = np.concatenate((u_guess.flatten(), np.zeros(self.p_horizon), np.zeros(self.p_horizon)))
        # arg["x0"] = np.concatenate((u_guess.flatten(), np.zeros(self.n)))
        arg["x0"] = u_guess.flatten()
        #arg["lbx"] = np.concatenate((lbx_u, lbx_xi))
        #arg["ubx"] = np.concatenate((ubx_u, ubx_xi))
        #arg["lbx"] = np.concatenate((lbx_u, lbx_eta))
        #arg["ubx"] = np.concatenate((ubx_u, ubx_eta))
        arg["lbx"] = lbx_u
        arg["ubx"] = ubx_u
        arg["p"] = np.concatenate((x0, x_ref))
        #arg["lbg"] = [0] * self.p_horizon
        #arg["ubg"] = [float('inf')] * self.p_horizon
        arg["lbg"] = [0, 0] * self.p_horizon
        arg["ubg"] = [float('inf'), float('inf')] * self.p_horizon

        # Solve the problem
        res = self.solver(**arg)
        u_opt = res['x'].full().reshape(-1)[:self.c_horizon*self.m].reshape(self.c_horizon, self.m)
        res_g = res['g'].full()
        
        #print(res_g)

        # Evolutions Tracking
        xi_optimal.append(res['x'].full().reshape(-1)[self.c_horizon*self.m:])
        cost_iter.append(float(res["f"]))
        constraint_iter.append(res_g)

        return u_opt[0, :], cost_iter, constraint_iter