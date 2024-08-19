import numpy as np
import matplotlib.pyplot as plt
from src.controllers.mpc_controller_obstacle import MPCController
from src.dynamics.dynamics_3d import rk4_step
from matplotlib.animation import FuncAnimation
from src.util.quat2eul import quaternion_to_euler
from src.util.eul2quat import euler_to_quaternion
import time
start_time = time.time()


# Constants
mass = 1
Ixx, Iyy, Izz = 1, 1, 1
I = np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
f_thruster = 1
dx, dy = 1, 1
u_min = 0
u_max = 1

# MPC Parameters
dt_MPC = 1
T_horizon = 20
c_horizon = 1
Q = 10 * np.eye(13) # State Weighting Matrix
#Q[6:10,:] = 0
R = 1 * np.eye(8)   # Control weighting matrix
P = 100 * np.eye(13) # Terminal Cost Weighting Matrix
#P[6:10,:] = 0
pho = 1e7
MPC_freq = 1

# Simulation parameters
simulation_time = 100  # Total simulation time in seconds
dt_sim = 1  # Time step
num_steps = int(simulation_time / dt_sim) # Number of simulation steps
x0 = np.zeros(13)
x0[6:10] = euler_to_quaternion(0,0,0)
predicted_states = np.zeros((num_steps, c_horizon, 13))
predicted_inputs = np.zeros((num_steps, c_horizon, 8))
x_obstacle = [2, 3, 0]
r_obstacle = 0.5
static_reference = False

# Reference State
x_ref_static = np.zeros(13) 

x_ref_static[0:2] = 10
x_ref_static[6:10] = euler_to_quaternion(0, 0, 0) # Yaw, Pitch = 0, Roll = 0
print(x_ref_static)

# Reference Trajectory Intial State
x_ref_dyn_initial = np.zeros(13)
x_ref_dyn_initial[6:10] = euler_to_quaternion(0,0,0)
print(x_ref_dyn_initial)


# Storage for states and inputs
states = np.zeros((num_steps + 1, 13))
inputs = np.zeros((num_steps, 8))
states_euler = np.zeros((num_steps + 1, 3))

# Set initial state
states[0, :] = x0
states_euler[0, :] = quaternion_to_euler(states[0, 6:10])
cost_evolution=[]
xi_evolution = []

def target_dynamics(t):

    if static_reference == True:
        x_ref = x_ref_static

    else:
        x_ref = x_ref_dyn_initial.copy()
        if t <= num_steps/2:
            # Simple Tranlastion
            x_ref[0] += (0.1) * t 
            x_ref[1] += (0.2) * t

            # Circular Motion
            # x_ref[0] += np.cos((np.pi/4)*t)
            # x_ref[1] += np.sin((np.pi/4)*t)

            # Decaying Eliptical Motion
            # x_ref[0] += 20 * np.exp(-0.01 * t) * np.cos(0.1*t)
            # x_ref[1] += 30 * np.exp(-0.01 * t) * np.sin(0.1*t) 
        else:
            x_ref[0] += 10
            x_ref[1] += 10
    return x_ref

def main():
    controller = MPCController(T_horizon, c_horizon, mass, I, dx, dy, dt_MPC, Q, R, P, u_min, u_max, x_obstacle, r_obstacle, pho)
    u_guess = np.zeros((c_horizon * 8, 1))

    # Simulate the system
    for t in range(num_steps):
        x_ref = target_dynamics(t)
        if t % int(1/(MPC_freq*dt_sim)) == 0:
            # Get the optimal control input
            u, xi_optimal, cost_iter = controller.get_optimal_input(states[t, :], x_ref, u_guess)
            predicted_inputs[t, :, :] = u
            
            # Use the model to predict the future trajectory
            X_pred = states[t, :].copy()
            for k in range(c_horizon):
                X_pred = rk4_step(X_pred, predicted_inputs[t, k, :], dt_sim)
                predicted_states[t, k, :] = X_pred
            cost_evolution.append(cost_iter[-1])
            xi_evolution.append(xi_optimal[-1])
        # Apply the control input to get the next state
        x_next = states[t + 1, :] = rk4_step(states[t, :], u[0,:], dt_sim)
        # Store the input and the next state
        states[t + 1, :] = x_next
        inputs[t, :] = u[0, :]
        u_guess = np.tile(u[0,:], (c_horizon, 1)).reshape(c_horizon * 8, 1)
        states_euler[t + 1, :] = quaternion_to_euler(x_next[6:10])
        
    
    # Plotting
    time = np.linspace(0, simulation_time, num_steps + 1)
    x_ref_evolution = []
    
    for t in range(num_steps):
        x_ref_evolution.append(target_dynamics(t))

    plt.figure(figsize=(12, 8))

    # Plot states
    plt.subplot(1, 1, 1)
    plt.plot(time, states[:, 0], label='r_x')
    plt.plot(time, states[:, 1], label='r_y')
    plt.plot(time, states_euler[:, 0], label='yaw')
    plt.xlabel('Time [s]')
    plt.ylabel('States')
    plt.legend()
    plt.grid()

    # Plot inputs
    plt.figure(figsize=(12, 8))
    time_inputs = np.linspace(0, simulation_time - dt_sim, num_steps)
    for i in range(1, 9): 
        plt.subplot(5, 2, i)
        plt.step(time_inputs, inputs[:, i-1], label=f'u{i}')
        plt.xlabel('Time [s]')
        plt.ylabel('Inputs')
        plt.legend()
        plt.grid()

    u_cumsum = np.cumsum(np.sum(inputs, axis=1), axis=0)
    plt.subplot(5, 1, 5)
    plt.step(time_inputs, u_cumsum, label=f'Total input')
    plt.xlabel('Time [s]')
    plt.ylabel('Total input')
    plt.legend()
    plt.grid()

    plt.tight_layout()

    # Plot cost history
    plt.figure(figsize=(12, 8))
    time_cost = np.linspace(0, simulation_time - dt_sim, len(cost_evolution))
    plt.plot(time_cost, cost_evolution)
    plt.xlabel('Time [s]')
    plt.ylabel('Cost')
    plt.title('Cost Evolution')
    plt.grid()

    # Plot slack history
    plt.figure(figsize=(12, 8))
    time_xi = np.linspace(0, simulation_time - dt_sim, len(xi_evolution))
    plt.plot(time_xi, xi_evolution)
    plt.xlabel('Time [s]')
    plt.ylabel('Cost')
    plt.title('Slack Variable Evolution')
    plt.grid()

    # Plot trajectory
    
    plt.figure(figsize=(8, 6))
    plt.plot(states[:, 0], states[:, 1])
    x_ref_evolution = np.array(x_ref_evolution)
    plt.plot(x_ref_evolution[:, 0], x_ref_evolution[:, 1],"--")
    body = plt.Circle((x_obstacle[0], x_obstacle[1]), r_obstacle, color='#303030', fill=True)
    circle = plt.Circle((x_obstacle[0], x_obstacle[1]), r_obstacle * 2, color='#B7B6B6', linestyle='dotted' , fill=True)
    plt.gca().add_patch(circle)
    plt.gca().add_patch(body)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory')
    plt.grid()


    plt.figure(figsize=(8,6))
    plt.plot(time, states[:, 6], label= 'q0')
    plt.plot(time, states[:, 7], label= 'q1')
    plt.plot(time, states[:, 8], label= 'q2')
    plt.plot(time, states[:, 9], label= 'q3')
    plt.xlabel('time')
    plt.ylabel('quaternions')
    plt.grid()
    plt.show() 

if __name__ == "__main__":
    main()
    # animate_trajectory(states, simulation_time, dt_sim)
    print("Process finished --- %s seconds ---" % (time.time() - start_time))
    

