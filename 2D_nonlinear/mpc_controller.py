from casadi import *

class MPCController:
    def __init__(self, time_horizon, c_horizon, mass, d, dt, Q, R, P, u_min, u_max):
        
        self.p_horizon = p_horizon = int(time_horizon/dt)
        self.c_horizon = c_horizon
        self.u_min = u_min
        self.u_max = u_max

        # States Variables Initialization
        x = MX.sym('x')
        x_dot = MX.sym('x_dot')
        y = MX.sym('y')
        y_dot = MX.sym('y_dot')
        theta = MX.sym('theta')
        theta_dot = MX.sym('theta_dot')

        # State Vector Concatenation
        states = vertcat(x, x_dot, y, y_dot, theta, theta_dot)
        self.n = states.size1()

        # Control Variables Initialization
        u = MX.sym('u', 8)
        controls = u
        self.m = controls.size1()

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

        ## Discretization 

        # Euler Forward
        next_state = states + dt * f(states, controls)
        F = Function('F', [states, controls], [next_state])

        # ## Range-Kutta 4 NOT WORKING
        # k1 = f(states, controls)
        # k2 = f(states + 0.5 * dt * k1, controls)
        # k3 = f(states + 0.5 * dt * k2, controls)
        # k4 = f(states + dt * k3, controls)
        # next_state = states + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # F = Function('F', [states, controls], [next_state])

        U = MX.sym('U', self.m, self.c_horizon)
        X = x0 # Initial State
        J = 0
        
        for k in range(self.p_horizon):
            # U_k = U[:, min(k, c_horizon-1)]
            if k < c_horizon:
                U_k = U[:, k]
            else:
                U_k = U[:, c_horizon-1]

            X_next = F(X, U_k) 
            
            x_delta = X - x_ref
            J += mtimes([x_delta.T, Q, x_delta])
            J += mtimes([U_k.T, R, U_k])
            
            X = X_next
        

        J += mtimes([(X - x_ref).T, P, (X - x_ref)])
        p = vertcat(x0,x_ref)
        
        # Solver Design
        nlp = {'x': reshape(U, -1, 1), 'f': J, 'p': p}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes', 'ipopt.max_iter': 100, 'ipopt.tol': 1e-6}
        self.solver = nlpsol('solver', 'ipopt', nlp, opts)

    def get_optimal_input(self, x0, x_ref, u_guess):
        cost_iter = []
        # Initial guess and bounds for the solver
        arg = {}
        arg["x0"] = u_guess
        arg["lbx"] = np.tile([self.u_min]*self.m, self.c_horizon)
        arg["ubx"] = np.tile([self.u_max]*self.m, self.c_horizon)
        arg["p"] = np.concatenate((x0, x_ref))
        
        # Solve the problem
        res = self.solver(**arg)
        u_opt = res['x'].full().reshape(self.c_horizon,self.m)
        cost_iter.append(float(res["f"]))

        return u_opt[0, :], cost_iter       