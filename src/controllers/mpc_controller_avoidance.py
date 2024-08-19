from casadi import *

class MPCController:
    def __init__(self, time_horizon, c_horizon, mass, I, dx, dy, dt, Q, R, P, u_min, u_max, apex, azimuth_bounds, elevation_bounds):
        
        self.p_horizon = p_horizon = int(time_horizon/dt)
        self.c_horizon = c_horizon
        self.u_min = u_min
        self.u_max = u_max
        self.cone_apex = apex
        self.azimuth_bounds = azimuth_bounds
        self.elevation_bounds = elevation_bounds

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
        omega_dot = mtimes(inv(I), mtimes(T_matrix, controls) - mtimes([omegas.T, I, omegas]))
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
        k1 = f(states, controls)
        k2 = f(states + 0.5 * dt * k1, controls)
        k3 = f(states + 0.5 * dt * k2, controls)
        k4 = f(states + dt * k3, controls)
        F = Function('F', [states, controls], [states + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)])
        # Euler Forward
        next_state = states + dt * f(states, controls)
        F = Function('F', [states, controls], [next_state])

        U = MX.sym('U', self.m, self.c_horizon)
        X = x0 # Initial State
        J = 0
        g = []  # constraints list
        r_threshold = 1
        
        for k in range(self.p_horizon):
            if k < c_horizon:
                U_k = U[:, k]
            else:
                U_k = U[:, c_horizon-1]

            X_next = F(X, U_k) 
            X_next[6:10] = X_next[6:10]/norm_2(X_next[6:10])
            
            distance_to_obstacle = sqrt(sum1((X_next[:3] - self.cone_apex)**2))
            print('ANTIGO',distance_to_obstacle)
            distance_to_obstacle = norm_2(X_next[:3]-self.cone_apex)
            print('NOVO', distance_to_obstacle)
            # Calculate the relative position vector
            relative_position = X_next[:3] - self.cone_apex
            
            # Calculate azimuth angle (in radians)
            azimuth_angle = atan2(relative_position[1], relative_position[0])
            
            # Calculate elevation angle (in radians)
            elevation_angle = atan2(relative_position[2], sqrt(relative_position[0]**2 + relative_position[1]**2))
            azimuth_constraint = if_else(distance_to_obstacle<r_threshold,azimuth_angle, 0)
            # Add the azimuth angle constraint
            g.append(azimuth_constraint)
            # Add the elevation angle constraint

            
            #g.append(elevation_angle)

            x_delta = X - x_ref
            J += mtimes([x_delta.T, Q, x_delta])
            J += mtimes([U_k.T, R, U_k])
            
            X = X_next

        J += mtimes([(X - x_ref).T, P, (X - x_ref)])
        p = vertcat(x0, x_ref)
        
        # Solver Design
        nlp = {'x': reshape(U, -1, 1), 'f': J, 'g': vertcat(*g), 'p': p}
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.sb': 'yes', 
            'ipopt.max_iter': 500, 
            'ipopt.tol': 1e-5,  
            'ipopt.acceptable_tol': 1e-4,  
            'ipopt.constr_viol_tol': 1e-3,  
        }

        self.solver = nlpsol('solver', 'ipopt', nlp, opts)

    def get_optimal_input(self, x0, x_ref, u_guess):
        cost_iter = []
        # Initial guess and bounds for the solver
        arg = {}
        arg["x0"] = u_guess
        arg["lbx"] = np.tile([self.u_min]*self.m, self.c_horizon)
        arg["ubx"] = np.tile([self.u_max]*self.m, self.c_horizon)
        arg["p"] = np.concatenate((x0, x_ref))
        
        # Set the bounds for the azimuth and elevation constraints
        # arg["lbg"] = [self.azimuth_bounds[0], self.elevation_bounds[0]] * self.p_horizon
        # arg["ubg"] = [self.azimuth_bounds[1], self.elevation_bounds[1]] * self.p_horizon
        arg["lbg"] = [self.azimuth_bounds[0]] * self.p_horizon
        arg["ubg"] = [self.azimuth_bounds[1]] * self.p_horizon
        
        # Solve the problem
        res = self.solver(**arg)
        u_opt = res['x'].full().reshape(self.c_horizon,self.m)
        cost_iter.append(float(res["f"]))

        return u_opt[0, :], cost_iter
