from casadi import *
from src.util.quaternion_inverse import quaternion_inverse
from src.util.quaternion_update import quaternion_update_ca
from src.util.quat2eul import ca_quaternion_to_euler
from src.util.quaternion_rotation import quaternion_to_rotation_matrix_casadi
from src.util.quaternion_rotation import pos_prime_rot_casadi
from src.util.custom_heaviside import custom_heaviside

class MPCController:
    def __init__(self, 
                 time_horizon=14,
                 c_horizon=4,
                 mass=15,
                 I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.5 * 15 * (0.2**2)]]),
                 dx = 0.12,
                 dy = 0.12,
                 dt = 1.0,
                 Q = 1e2 * np.eye(13),
                 R = 1 * 1e6 * np.eye(8),
                 P = 1e3 * np.eye(13),
                 u_min = 0.0,
                 u_max = 1.0,
                 r_chaser = 0.2,
                 r_target = 0.2,
                 chaser_vector_C = np.array([1, 0, 0]),
                 target_vector_T = np.array([0, -1, 0]),
                 x_obstacle = np.array([1.5, 1.5, 0]),
                 r_obstacle = 0.2,
                 flags = [True, True]):

        self.p_horizon = int(time_horizon/dt)
        self.c_horizon = c_horizon
        self.u_min = u_min
        self.u_max = u_max
        self.r_chaser = r_chaser
        self.chase_vector_C = chaser_vector_C
        self.target_vector_T = target_vector_T
        self.r_obstacle = r_obstacle

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

        dynamics = vertcat(x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot, 0, 0, 0, 0, omega1_dot, omega2_dot, omega3_dot)

        # Initial and Target State 
        x_ref = SX.sym('x_ref', 13)
        x_initial = SX.sym('x0', 13)

        # Quaternion Matrix to calculate its variation
        quat_A = vertcat(
            horzcat(x_ref[6], x_ref[7], x_ref[8], x_ref[9]),
            horzcat(-x_ref[7], x_ref[6], -x_ref[9], -x_ref[8]),
            horzcat(-x_ref[8], -x_ref[9], x_ref[6], x_ref[7]),
            horzcat(-x_ref[9], x_ref[8], -x_ref[7], x_ref[6]))
        
        # Continuous Dynamics Function
        f = Function('f', [states, controls], [dynamics])

        # # Discretization 
        # Euler Forward
        next_state = states + dt * f(states, controls)
        F = Function('F', [states, controls], [next_state])
        next_quaternions = quaternion_update_ca(states[6:10], states[10:13], dt)
        F_quat = Function('F_quat', [states, controls], [next_quaternions]) 

        U = SX.sym('U', self.m, self.c_horizon)
        X = x_initial  # Initial State
        J = 0 # Cost
        g = []  # constraints list
        x_docking = DM.zeros(13)
        x_target = DM.zeros(13)
        for k in range(self.p_horizon):
            
            U_k = U[:, min(k, c_horizon-1)] # From the control horizon, U_k is set as U_c_horizon

            
            
            
            x_target[0] = DM(x_ref[0])
            x_target[1] = DM(x_ref[1])
            x_target[2] = DM(x_ref[2])
            #target_delta = X[0:3] - (x_target[0:3])
            target_delta = X[0:3] - (x_ref[0:3])
            
            distance_to_target = norm_2(target_delta)

            chaser_vector_T = pos_prime_rot_casadi(x_target[6], x_target[7], x_target[8], x_target[9], self.chase_vector_C[0], self.chase_vector_C[1], self.chase_vector_C[2])
            chaser_vector_N = pos_prime_rot_casadi(X[6], X[7], X[8], X[9], self.chase_vector_C[0], self.chase_vector_C[1], self.chase_vector_C[2])
            target_vector_N = pos_prime_rot_casadi(x_target[6], x_target[7], x_target[8], x_target[9], self.target_vector_T[0], self.target_vector_T[1], self.target_vector_T[2])
            chaser_target_cross = cross(chaser_vector_T, self.target_vector_T)
            norm_c_t_cross = norm_2(chaser_target_cross)
            
            euler = ca_quaternion_to_euler(X[6:10])
            x_prime = X[0] - x_target[0] 
            y_prime = X[1] - x_target[1] 
            z_prime = X[2] - x_target[2]
            effective_target_radius = r_target + r_chaser

            pos_prime_rotated = pos_prime_rot_casadi(x_target[6], x_target[7], x_target[8], x_target[9], x_prime, y_prime, z_prime)
            #x_docking[0:3] = [x_ref[0], x_ref[1] + r_target + r_chaser, x_ref[2]] 
            x_docking[0] = DM(x_ref[0])
            x_docking[1] = DM(x_ref[1] + r_target + r_chaser)
            x_docking[2] = DM(x_ref[2])
            
            pos_docking = pos_prime_rot_casadi(x_target[6],x_target[7],x_target[8],x_target[9],x_docking[0], x_docking[1], x_docking[2])

            entry_constraint1_condition = pos_prime_rotated[1] - pos_prime_rotated[0] 
            entry_constraint2_condition = pos_prime_rotated[1] + pos_prime_rotated[0] 
            entry_constraint1 = custom_heaviside(entry_constraint1_condition) * (distance_to_target - 2 * effective_target_radius )
            entry_constraint2 = custom_heaviside(entry_constraint2_condition) * (distance_to_target - 2 * effective_target_radius )

            # Test Bed Size Constraints
            edge_constraint1 = 3.32 - X[1] - r_chaser
            edge_constraint2 = X[1] - r_chaser
            edge_constraint3 = X[0] - r_chaser
            edge_constraint4 = 4.10 - X[0] - r_chaser
            
            # Obstacle Constraint
            obstacle_delta = X[0:3] - x_obstacle[0:3] # Position Deviation
            distance_to_obstacle = norm_2(obstacle_delta) # Distance to the Obstacle
            
            effective_obstacle_radius = r_obstacle + r_chaser
            effective__margin1_radius = r_obstacle*2 + r_chaser

            # obstacle_margin_constraint = distance_to_obstacle - effective__margin1_radius + xi_obstacle[k] # Outer Circle Constraint (SOFT)
            obstacle_constraint = distance_to_obstacle - effective_obstacle_radius # Obstacle Constraint (HARD)
            
            # Appending Constraints
            
            g.append(if_else(flags[0] == True, entry_constraint1, 0))
            g.append(if_else(flags[0] == True, entry_constraint2, 0))
            
            g.append(edge_constraint1)
            g.append(edge_constraint2)
            g.append(edge_constraint3)
            g.append(edge_constraint4)

            g.append(if_else(flags[1] == True, obstacle_constraint, 0))

            # Cost Function Composition
        
            effective_target_radius = r_target + r_chaser

            #pos_delta = X[0:3] - x_docking[0:3]
            pos_delta = X[0:3] - x_ref[0:3]
            vel_delta = X[3:6] - x_ref[3:6]
            omega_delta = X[10:13] - x_ref[10:13] # Angular Rate Deviation
            quat_err = mtimes(quat_A, quaternion_inverse(X[6:10])) # Quaternion Deviation
            x_delta = vertcat(pos_delta, vel_delta, quat_err, omega_delta)
            distance_to_target_squared = (X[0] - x_target[0])**2 + (X[1] - x_target[1])**2

            J += mtimes([x_delta.T, Q, x_delta]) #* (1/(distance_to_target))  # State Deviation Cost 
            J += mtimes([U_k.T, R, U_k])#Input Cost 

            X_next = F(X, U_k) # Next State Computation
            X_next[6:10] = F_quat(X, U_k) #(Only working if norm of initial omegas aren't all 0!!!)
            X_next[6:10] = X_next[6:10]/norm_2(X_next[6:10]) # Quaternions Normalization
            X = X_next # State Update
        
        ## Terminal Cost
        quat_err_ter = mtimes(quat_A, quaternion_inverse(X[6:10])) #

        #J += mtimes([(X[0:3] - x_docking[0:3]).T, P[0:3, 0:3],(X[0:3] - x_docking[0:3])])
        J += mtimes([(X[0:3] - x_ref[0:3]).T, P[0:3, 0:3],(X[0:3] - x_ref[0:3])])
        J += mtimes([(X[3:6] - x_ref[3:6]).T, P[3:6,3:6], (X[3:6] - x_ref[3:6])])
        J += mtimes([(quat_err_ter).T , P[6:10,6:10], quat_err_ter])
        J += mtimes([(X[10:13] - x_ref[10:13]).T, P[10:13,10:13], (X[10:13] - x_ref[10:13])])

        p = vertcat(x_initial,x_ref)

        # Solver Design
        nlp = {'x': vertcat(reshape(U, -1, 1)), 'f': J, 'p': p, 'g': vertcat(*g)}
       
        opts = {'ipopt.print_level': 0, 
                'print_time': 0, 
                'ipopt.sb': 'yes', 
                'ipopt.max_iter': 100, 
                'ipopt.tol': 1e-3, 
                'ipopt.constr_viol_tol': 1e-4,
                'ipopt.acceptable_constr_viol_tol': 1e-3, 
                'ipopt.honor_original_bounds' : 'no',
                'ipopt.bound_relax_factor': 0}

        self.solver = nlpsol('solver', 'ipopt', nlp, opts) # Solver Initiation with IPOPT

    # Optimal Input Calculation
    def get_optimal_input(self, x0, x_ref, u_guess):

        # Cost and Slack Variables Evolution Initialization
        cost_iter = []
        eta_optimal = []

        # Bounds Organization
        lbx_u = np.tile([self.u_min]*self.m, self.c_horizon)
        ubx_u = np.tile([self.u_max]*self.m, self.c_horizon)

        # Solver Bounds, Parameters, and Initial States Definition
        arg = {}
        
        arg["x0"] = u_guess.flatten()
        arg["lbx"] = lbx_u
        arg["ubx"] = ubx_u
        arg["p"] = np.concatenate((x0, x_ref))
        arg["lbg"] = [0, 0, 0, 0, 0, 0, 0] * self.p_horizon
        arg["ubg"] = [float('inf'),float('inf'),float('inf'),float('inf'),float('inf'),float('inf'),float('inf')] * self.p_horizon  
        
        # Solve the problem
        res = self.solver(**arg)
        u_opt = res['x'].full().reshape(-1)[:self.c_horizon*self.m].reshape(self.c_horizon, self.m)

        # Evolutions Track
        eta_optimal.append(res['x'].full().reshape(-1)[self.c_horizon * self.m:])
        cost_iter.append(float(res["f"]))

        return u_opt, eta_optimal, cost_iter
