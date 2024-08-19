from casadi import *

class MPCController:
    def __init__(self, time_horizon, c_horizon, mass, I, dx, dy, dt, Q, R, P, u_min, u_max, x_obstacle, radius_obstacle, rho):
        
        self.p_horizon = p_horizon = int(time_horizon/dt)
        self.c_horizon = c_horizon
        self.u_min = u_min
        self.u_max = u_max
        self.radius_obstacle = radius_obstacle


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

        # Slack variable initialization
        xi_obstacle = MX.sym('xi_obstacle', self.p_horizon)  # One slack variable for each time step

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

        omega_dot = mtimes(inv(I), mtimes(T_matrix, controls) - mtimes([omega_tilde,I,omegas]))
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

        # RK4 
        # k1 = f(states, controls)
        # k2 = f(states + 0.5 * dt * k1, controls)
        # k3 = f(states + 0.5 * dt * k2, controls)
        # k4 = f(states + dt * k3, controls)
        # F = Function('F', [states, controls], [states + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)])

        # # Euler Forward
        next_state = states + dt * f(states, controls)
        F = Function('F', [states, controls], [next_state])

        U = MX.sym('U', self.m, self.c_horizon)
        X = x0  # Initial State
        J = 0
        g = []  # constraints list
        quat_A = vertcat(
            horzcat(x_ref[0], x_ref[1], x_ref[2], x_ref[3]),
            horzcat(-x_ref[1], x_ref[0], -x_ref[3], -x_ref[2]),
            horzcat(-x_ref[2], -x_ref[3], x_ref[0], x_ref[1]),
            horzcat(-x_ref[3], x_ref[2], -x_ref[1], x_ref[0])
        )
        for k in range(self.p_horizon):
            if k < c_horizon:
                U_k = U[:, k]
            else:
                U_k = U[:, c_horizon-1]

            X_next = F(X, U_k) 

            X_next[6:10] = X_next[6:10]/norm_2(X_next[6:10])
            
            pos_delta = X[0:3] - x_obstacle[0:3]
            distance_to_obstacle = norm_2(pos_delta)

            obstacle_margin_constraint = distance_to_obstacle - radius_obstacle * 2 + xi_obstacle[k]
            obstacle_constraint = distance_to_obstacle - radius_obstacle

            
            g.append(obstacle_constraint)
            g.append(obstacle_margin_constraint)
            
            pos_vel_delta = X[0:6] - x_ref[0:6]
            omega_delta = X[10:13] - x_ref[10:13]
            
            quat_err = mtimes(quat_A, vertcat(X[6:10]))
            x_delta = vertcat(pos_vel_delta, quat_err, omega_delta)
            J += mtimes([x_delta.T, Q, x_delta])
            J += mtimes([U_k.T, R, U_k])
            J += rho * xi_obstacle[k]**2
            
            X = X_next
        
        quat_err_ter = mtimes(quat_A, vertcat(X[6:10]))
        J += mtimes([(X[0:6] - x_ref[0:6]).T, P[0:6,0:6], (X[0:6] - x_ref[0:6])])
        J += mtimes([quat_err_ter.T, P[6:10,6:10], quat_err_ter])
        J += mtimes([(X[10:13] - x_ref[10:13]).T, P[10:13,10:13], (X[10:13] - x_ref[10:13])])
        p = vertcat(x0,x_ref)

        # Solver Design
        nlp = {'x': vertcat(reshape(U, -1, 1), xi_obstacle), 'f': J, 'g': vertcat(*g), 'p': p}
        #nlp = {'x': reshape(U, -1, 1), 'f': J, 'g': vertcat(*g), 'p': p}
        #nlp = {'x': vertcat(reshape(U, -1, 1)), 'f': J, 'p': p}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes', 'ipopt.max_iter': 100, 'ipopt.tol': 1e-6}
        # opts = {
        #     'ipopt.print_level': 0, 
        #     'print_time': 0, 
        #     'ipopt.sb': 'yes', 
        #     'ipopt.max_iter': 500, 
        #     'ipopt.tol': 1e-5,
        #     'ipopt.acceptable_tol': 1e-4,
        #     'ipopt.constr_viol_tol': 1e-3,
        # }


        self.solver = nlpsol('solver', 'ipopt', nlp, opts)
        #self.solver = nlpsol('solver', 'sqpmethod', nlp)


    def get_optimal_input(self, x0, x_ref, u_guess):
        cost_iter = []
        xi_optimal = []
        # Initial guess and bounds for the solver
        lbx_u = np.tile([self.u_min]*self.m, self.c_horizon)
        ubx_u = np.tile([self.u_max] * self.m, self.c_horizon)
        lbx_xi = [0] * self.p_horizon
        ubx_xi = [float(inf)] * self.p_horizon
        arg = {}
        arg["x0"] = np.concatenate((u_guess.flatten(), np.zeros(self.p_horizon)))
        #arg["x0"] = u_guess.flatten()
        arg["lbx"] = np.concatenate((lbx_u,lbx_xi))
        arg["ubx"] = np.concatenate((ubx_u,ubx_xi))
        #arg["lbx"] = lbx_u
        #arg["ubx"] = ubx_u
        #arg["lbg"] = [0] * self.p_horizon  
        arg["lbg"] = [0, 0] * self.p_horizon
        #arg["ubg"] = [float('inf')] * self.p_horizon  
        arg["ubg"] = [float('inf'), float('inf')] * self.p_horizon  
        arg["p"] = np.concatenate((x0, x_ref))
        
        # Solve the problem
        res = self.solver(**arg)
        u_opt = res['x'].full().reshape(-1)[:self.c_horizon*self.m].reshape(self.c_horizon, self.m)
        #u_opt = res['x'].full().reshape(self.c_horizon,self.m)
        xi_optimal.append(res['x'].full().reshape(-1)[self.c_horizon*self.m:])
        cost_iter.append(float(res["f"]))

        return u_opt, xi_optimal, cost_iter
