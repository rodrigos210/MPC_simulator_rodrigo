## Docking through defined entry-angle scenario ##
# Not tuned 

import numpy as np
import matplotlib.pyplot as plt
from src.controllers.mpc_controller_entry_angle import MPCController
from src.dynamics.dynamics_3d import rk4_step
from src.util.eul2quat import euler_to_quaternion
from src.util.quat2eul import quaternion_to_euler
from matplotlib.animation import FuncAnimation
import time
start_time = time.time()

# Constants
mass = 1 #[kg]
Ixx, Iyy, Izz = 1, 1, 1
I = np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
f_thruster = 1 # Maximum Thrust [N]
dx, dy = 1, 1 # Distance from the CoM to the Thrusters [m]
u_min = 0 # Lower Thrust Bound
u_max = 1 # Upper Thrust Bound

# MPC Parameters
dt_MPC = 1
T_horizon = 12 # Prediction Horizon = T_horizon / dt_MPC
c_horizon = 3 # Control Horizon
# Q = np.diag([1e2, 1e2, 0,
#              1, 1, 1,
#              10, 10, 10, 10,
#              1, 1, 1])
Q = 1e1 * np.eye(13) # State Weighting Matrix #1e1
# #Q[1,1] = 1e7
Q[3,3] = 1e2 #1e3
Q[4,4] = 1e2 #1e3
Q[6:10,:] = 0 #0
Q[10:13,10:13] = 1e0 * np.eye(3) #1e0 
R = 1e-1 * np.eye(8) # Control Weighting matrix #1e-1
P = 1e2 * np.eye(13) # Terminal Cost Weighting Matrix 1e2
P[6:10,:] = 0 #0
rho = 0
MPC_freq = 1

# Simulation parameters
simulation_time = 150  # Total simulation time in seconds
dt_sim = 0.1  # Time step
num_steps = int(simulation_time / dt_sim) # Number of simulation steps
x0 = np.zeros(13) # Initial State Initialization
x0[0:2] = [10, 15]
x0[6:10] = euler_to_quaternion(0,0,0)
x0[10:13] = [0, 1e-24, 0]

# Static Reference Scenario Parameters
x_ref_static = np.zeros(13) 
x_ref_static[0:2] = 10
x_ref_static[6:10] = euler_to_quaternion(0, 0, 0) # Yaw, Pitch = 0, Roll = 0
x_ref_static[10:13] = [0, 1e-24, 0]

apex = x_ref_static
cone_vector = np.array([1, 0, 0])
cone_angle = np.deg2rad(60) 
entry_radius = 1

# Dynamic Reference Trajectory / Path Following Parameters
x_ref_dyn_initial = np.zeros(13)
x_ref_dyn_initial[6:10] = euler_to_quaternion(0,0,0)

# Storage for states and inputs
states = np.zeros((num_steps + 1, 13))
inputs = np.zeros((num_steps, 8))
states_euler = np.zeros((num_steps + 1, 3))

# Path Following Condition (True -> Static Reference Target, False -> Path Following Scenario)
static_reference = True

# Target x_ref definition
def target_dynamics(t):
    if static_reference == True:
        x_ref = x_ref_static
    else:
        x_ref = x_ref_dyn_initial.copy()
        # Simple Tranlastion
        # x_ref[0] += (0.1) * t 
        # x_ref[2] += (0.1) * t

        # Circular Motion
        x_ref[0] += 0.1 * np.cos(0.01*t)
        x_ref[1] += 0.1 * np.sin(0.01*t)

        # Decaying Eliptical Motion
        # x_ref[0] += 20 * np.exp(-0.01 * t) * np.cos(0.1*t)
        # x_ref[1] += 30 * np.exp(-0.01 * t) * np.sin(0.1*t) 
    return x_ref

# Set initial state
states[0, :] = x0
states_euler[0, :] = quaternion_to_euler(states[0, 6:10])
cost_evolution=[]
constraint_evolution = []

def main():
    controller = MPCController(T_horizon, c_horizon, mass, I, dx, dy, dt_MPC, Q, R, P, u_min, u_max, entry_radius, rho)
    u_guess = np.zeros((c_horizon * 8, 1))
    # Simulate the system
    for t in range(num_steps):
        x_ref = target_dynamics(t)
        if t % int(1/(MPC_freq*dt_sim)) == 0:
            # Get the optimal control input
            u, cost_iter, constraint_iter = controller.get_optimal_input(states[t, :], x_ref, u_guess)
            
            cost_evolution.append(cost_iter[-1])
            constraint_evolution.append(constraint_iter[-1])

        # Apply the control input to get the next state
        x_next = states[t + 1, :] = rk4_step(states[t, :], u, dt_sim)
        # Store the input and the next state
        states[t + 1, :] = x_next
        inputs[t, :] = u
        u_guess = np.tile(u, (c_horizon, 1)).reshape(c_horizon * 8, 1)
        #u_guess = np.tile(u, (c_horizon, 1))
        #u_guess = np.tile(np.zeros(8), (c_horizon, 1)).reshape(c_horizon * 8, 1)
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
    for i in range(1, 9): #change 5 to 9 when full thrusters
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


    # Plot trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(states[:, 0], states[:, 1])
    x_ref_evolution = np.array(x_ref_evolution)
    plt.plot(x_ref_evolution[:, 0], x_ref_evolution[:, 1],"--")
    plt.plot([x_ref[0], x_ref[0] + 1], [x_ref[1], x_ref[1] + 1], 'r--')
    plt.plot([x_ref[0], x_ref[0] - 1], [x_ref[1], x_ref[1] + 1], 'r--')
    threshold = plt.Circle((x_ref[0], x_ref[1]), entry_radius, color='#303030', fill=False)
    plt.gca().add_patch(threshold)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory')
    plt.grid()

    # Plot Quaternions
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
    print("Process finished --- %s seconds ---" % (time.time() - start_time))
