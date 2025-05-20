## Translation and Path Following Scenario ##
# Set static_reference = True for Translation; static_reference = False for Path Following.

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from src.controllers.mpc_3d import MPCController
from src.dynamics.dynamics_3d import rk4_step
from src.util.eul2quat import euler_to_quaternion
from src.util.quat2eul import np_quaternion_to_euler
from matplotlib.animation import FuncAnimation
import time

# Start Timer
start_time = time.time()

# Flags for plotting
plt_save = False  # Save the plots
plt_show = True # Show the plots

# Constants
mass = 15 #[kg]
f_thruster = 1 # Maximum Thrust [N]
dx, dy = 0.12, 0.12 # Distance from the CoM to the Thrusters [m]
u_min = 0 # Lower Thrust Bound
u_max = 1 # Upper Thrust Bound
L = 0.2828 # Length of the Robot (L x L) 
r_chaser = r_target = 0.5 * (L * np.sqrt(2)) # For approximating the shape of the sc to a circle
chaser_vector_C = np.array([1,0,0]) # Chaser Unity Direction Vector in the
target_vector_T = np.array([0,-1,0]) # Target Unity Direction Vector in the N
Ixx, Iyy = 1, 1
Izz = 0.5 * mass * (r_chaser**2) # Approximated to a cylinder shape
I = np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]]) # Moment of Inertia Matrix

# MPC Parameters
dt_MPC = 1
T_horizon = 15 # Prediction Horizon = T_horizon / dt_MPC
c_horizon = 5 # Control Horizon
Q = 1 * np.eye(13) # State Weighting Matrix
Q[6:10,6:10] = 0 * 1e-1 * np.eye(4)
Q[10:13,10:13] = 0
R = 100 * np.eye(8) # Control Weighting matrix
P = 1e1 * np.eye(13) # Terminal Cost Weighting Matrix
P[6:10,6:10] = 0 * 1e0 * np.eye(4)
P[10:13,10:13] = 0
MPC_freq = 1

# Simulation parameters
simulation_time = 200  # Total simulation time in seconds
dt_sim = 0.1  # Time step
num_steps = int(simulation_time / dt_sim) # Number of simulation steps
x0 = np.zeros(13) # Initial State Initialization
x0[6:10] = euler_to_quaternion(0,0,0)
x0[10:13] = [0, 1e-16, 0]

# Static Reference Scenario Parameters
x_ref_static = np.zeros(13) 
x_ref_static[0:2] = [3.5,3.5]
x_ref_static[6:10] = euler_to_quaternion(0, 0, 0) # Yaw, Pitch = 0, Roll = 0
#x_ref_static[6:10] = [0.707, 0, 0.0, 0.707]
x_ref_static[10:13] = [0, 1e-8, 0]

# Dynamic Reference Trajectory / Path Following Parameters
x_ref_dyn_initial = np.zeros(13)
x_ref_dyn_initial[6:10] = euler_to_quaternion(0,0,0)
x_ref_dyn_initial[10:13] = [0, 1e-8, 0]
# Storage for states and inputs
states = np.zeros((num_steps + 1, 13))
inputs = np.zeros((num_steps, 8))
states_euler = np.zeros((num_steps + 1, 3))

# Path Following Condition (True -> Static Reference Target, False -> Path Following Scenario)
static_reference = False

# Target x_ref definition
def target_dynamics(t):
    if static_reference == True:
        x_ref = x_ref_static
    else:
        x_ref = x_ref_dyn_initial.copy()
        # Simple Tranlastion
        #x_ref[0] += (1) * t 
        #x_ref[1] += (2) * t

        # Circular Motion
        # x_ref[0] += 0.1 * np.cos(0.01*t)
        # x_ref[1] += 0.1 * np.sin(0.01*t)

        # Decaying Eliptical Motion
        x_ref[0] = 2 * np.exp(-0.01 * t) * np.cos(0.1*t)
        x_ref[1] = 3 * np.exp(-0.01 * t) * np.sin(0.1*t) 
    return x_ref

# Set initial state
states[0, :] = x0
states_euler[0, :] = np_quaternion_to_euler(states[0, 6:10])
cost_evolution=[]

def simulation():
    controller = MPCController(T_horizon, c_horizon, mass, I, dx, dy, dt_MPC, Q, R, P, u_min, u_max)
    u_guess = np.zeros((c_horizon * 8, 1))
    
    # Simulate the system
    for t in range(num_steps):
        x_ref = target_dynamics(t)
        print(x_ref)
        if t % int(1/(MPC_freq*dt_sim)) == 0:
            # Get the optimal control input
            u, cost_iter = controller.get_optimal_input(states[t, :], x_ref, u_guess)
            
            cost_evolution.append(cost_iter[-1])
        # Apply the control input to get the next state
        x_next = states[t + 1, :] = rk4_step(states[t, :], u, dt_sim)
        # Store the input and the next state
        states[t + 1, :] = x_next
        inputs[t, :] = u
        u_guess = np.tile(u, (c_horizon, 1)).reshape(c_horizon * 8, 1)
        states_euler[t + 1, :] = np_quaternion_to_euler(x_next[6:10])

def save_simulation_parameters(filename):
    params = {
        "mass": mass,
        "Ixx": Ixx,
        "Iyy": Iyy,
        "Izz": Izz,
        "Thrust bounds": [u_min, u_max],
        "MPC Horizon": T_horizon,
        "Control Horizon": c_horizon,
        "State Weighting Matrix Q": Q,
        "Control Weighting Matrix R": R,
        "Terminal Cost Weighting Matrix P": P,
        "dt_mpc": dt_MPC,
        "simulation_time": simulation_time,
        "dt_sim": dt_sim,
        "Initial state x0": x0,
        "Static reference": x_ref_static
    }

    with open(filename, 'w') as file:
        for key, value in params.items():
            file.write(f"{key}: {value}\n")

def output_directory_creation():
    # Get the script's name without the extension
    scenario_name = os.path.splitext(os.path.basename(__file__))[0]

    # Create a folder name based on scenario name and current date-time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join('outputs', f"{scenario_name}_{current_time}")

    # Create the directory if it doesn't exist
    if plt_save:
        os.makedirs(output_folder, exist_ok=True)
    return output_folder
                    
def simulation_results_generation(output_folder):

    if plt_save:
        os.makedirs(output_folder, exist_ok=True)
        save_simulation_parameters(os.path.join(output_folder, "simulation_parameters.txt"))

    # Define time arrays for plotting
    time = np.linspace(0, simulation_time, num_steps + 1)
    time_inputs = np.linspace(0, simulation_time - dt_sim, num_steps)
    time_cost = np.linspace(0, simulation_time - dt_sim, len(cost_evolution))
    
    if plt_save:
        os.makedirs(output_folder, exist_ok=True)
        save_simulation_parameters(os.path.join(output_folder, "simulation_parameters.txt"))

    # Plot states
    fig, ax1= plt.subplots(figsize=(12, 8))
    ax1.plot(time, states[:, 0], label='x')
    ax1.plot(time, states[:, 1], label='y')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Position [m]')
    ax1.legend(loc='upper left')
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.plot(time, np.degrees(states_euler[:, 0]), label = 'yaw', color = 'red')
    ax2.set_ylabel('Attitude [º]')
    ax2.legend(loc='upper right')   
    # plt.figure(figsize=(12, 8))
    # plt.subplot(1, 1, 1)
    # plt.plot(time, states[:, 0], label='r_x')
    # plt.plot(time, states[:, 1], label='r_y')
    # plt.plot(time, states_euler[:, 0], label='yaw')
    # plt.xlabel('Time [s]')
    # plt.ylabel('States')
    # plt.legend()
    # plt.grid()

    if plt_save:
        plt.savefig(os.path.join(output_folder, 'state_plot.png'))

    # Plot inputs
    plt.figure(figsize=(12, 8))
    for i in range(1, 9):
        plt.subplot(5, 2, i)
        plt.step(time_inputs, inputs[:, i-1], label=f'u{i}')
        plt.xlabel('Time [s]')
        plt.ylabel('Inputs')
        plt.legend()
        plt.grid()
    u_cumsum = np.cumsum(np.sum(inputs, axis=1), axis=0)
    plt.subplot(5, 1, 5)
    plt.step(time_inputs, u_cumsum, label='Total input')
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

    if plt_save:
        plt.savefig(os.path.join(output_folder, 'inputs_plot.png'))

    # Plot cost history
    plt.figure(figsize=(12, 8))
    plt.plot(time_cost, cost_evolution)
    plt.xlabel('Time [s]')
    plt.ylabel('Cost')
    plt.title('Cost Evolution')
    plt.grid()

    if plt_save:
        plt.savefig(os.path.join(output_folder, 'cost_evolution_plot.png'))

    # Plot trajectory
    
    plt.figure(figsize=(8, 6))
    plt.plot(states[:, 0], states[:, 1], color = '#999999')
    x_ref_evolution = np.array([target_dynamics(t) for t in range(num_steps)])
    plt.plot(x_ref_evolution[:, 0], x_ref_evolution[:, 1],"--")
    chaser_body = plt.Circle((x0[0], x0[1]), r_chaser, color = '#333333', fill= True)
    target_point = plt.Circle((x_ref_static[0], x_ref_static[1]), r_target/5, color = '#ee3432', fill=True)
    plt.gca().add_patch(chaser_body)
    plt.gca().add_patch(target_point)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory')
    plt.grid()

    if plt_save:
        plt.savefig(os.path.join(output_folder, 'trajectory_plot.png'))

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

    if plt_save:
        plt.savefig(os.path.join(output_folder, 'quaternion_plot.png'))

    if plt_show:
        plt.show()

if __name__ == "__main__":
    simulation()
    print("Process finished --- %s seconds ---" % (time.time() - start_time))
    output_folder = output_directory_creation()
    simulation_results_generation(output_folder)

def run():
    simulation()
    print("Process finished --- %s seconds ---" % (time.time() - start_time))
    output_folder = output_directory_creation()
    simulation_results_generation(output_folder)
