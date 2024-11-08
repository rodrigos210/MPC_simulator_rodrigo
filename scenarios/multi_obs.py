## Obstacle Avoidance Scenario ##
# Tuned

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from src.controllers.mpc_multi_obs import MPCController
from src.dynamics.dynamics_3d import rk4_step
from matplotlib.animation import FuncAnimation
from src.util.quat2eul import np_quaternion_to_euler
from src.util.eul2quat import euler_to_quaternion
from src.util.quaternion_rotation import quaternion_to_rotation_matrix_numpy
import time

# Start Timer
start_time = time.time()

# Flags for plotting
plt_save = False # Save the plots
plt_show = True # Show the plots

# Constants
mass = 1 #[kg]
Ixx, Iyy, Izz = 1, 1, 1
I = np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
f_thruster = 1 # Maximum Thrust [N]
dx, dy = 0.5, 0.5 # Distance from the CoM to the Thrusters [m]
u_min = 0 # Lower Thrust Bound
u_max = 1 # Upper Thrust Bound
L = 1 # Length of the Robot (L x L)
r_spacecraft = 0.5 * (L * np.sqrt(2)) # For approximating the shape of the sc to a c
vertices = np.array([
    [-L/2, -L/2, 0],
    [L/2, -L/2, 0],
    [L/2, L/2, 0],
    [-L/2, L/2, 0]
])

# MPC Parameters
dt_MPC = 1
T_horizon = 12 # Prediction Horizon = T_horizon / dt_MPC #12
c_horizon = 3 # Control Horizon #3
Q = 1e2 * np.eye(13) # State Weighting Matrix # 1e3
Q[3,3] = 1e5 # 1e5
Q[4,4] = 1e5 # 1e5
Q[10:13,10:13] = 1e5 * np.eye(3) # 1e5
#Q[5,5] = 0
#Q[2,2] = 0
#Q[3:5,3:5] = 1
Q[6:10,:] = 1 # 1
R = 1e3 * np.eye(8)   # Control Weighting Matrix # 1e3
P = 1e4* np.eye(13) # Terminal Cost Weighting Matrix #1e4
P[6:10,:] = 0 #0
rho1 = 1e7 # Obstacle Marging Slack Variable Weight # 
rho2 = 1e7 # Obstacle Marging Slack Variable Weight # 
MPC_freq = 1 #1

# Simulation parameters
simulation_time = 150  # Total simulation time in seconds
dt_sim = 0.1  # Time step
num_steps = int(simulation_time / dt_sim) # Number of simulation steps
x0 = np.zeros(13) # Initial State Initialization
x0[0:2] = [0,0]
x0[6:10] = euler_to_quaternion(0,0,0)
x0[10:13] = [0.0, 1e-24, 0] # one of the terms of the angular velocity still has to be > 0 due to division by 0 errors

predicted_states = np.zeros((num_steps, c_horizon, 13)) # Initialization for predicted state and inputs evolution
predicted_inputs = np.zeros((num_steps, c_horizon, 8))

# Obstacles Parameters
x_obstacle1 = [3, 4, 0]
r_obstacle1 = 0.2

x_obstacle2 = [7, 6, 0]
r_obstacle2 = 0.3

# Static Reference Scenario Parameters
x_ref_static = np.zeros(13) 
x_ref_static[0:2] = [10, 10]
x_ref_static[6:10] = euler_to_quaternion(0, 0, 0) # Yaw, Pitch = 0, Roll = 0

# Dynamic Reference Trajectory / Path Following Parameters
x_ref_dyn_initial = np.zeros(13)
x_ref_dyn_initial[6:10] = euler_to_quaternion(0,0,0)

# Path Following Condition (True -> Static Reference Target, False -> Path Following Scenario)
static_reference = True

# Target x_ref definition
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

# Storage for states and inputs
states = np.zeros((num_steps + 1, 13))
inputs = np.zeros((num_steps, 8))
states_euler = np.zeros((num_steps + 1, 3))

# Set initial state
states[0, :] = x0
states_euler[0, :] = np_quaternion_to_euler(states[0, 6:10])
cost_evolution=[]
xi_evolution = [] # Obstacle Marging Slack Variable Evolution
eta_evolution = [] # Terminal Cost Slack Variable Evolution

def simulation():
    controller = MPCController(T_horizon, c_horizon, mass, I, dx, dy, dt_MPC, Q, R, P, u_min, u_max, x_obstacle1, r_obstacle1, x_obstacle2, r_obstacle2, rho1, rho2, r_spacecraft, vertices)
    u_guess = np.zeros((c_horizon * 8, 1))

    # Simulate the system
    for t in range(num_steps):
        x_ref = target_dynamics(t)
        if t % int(1/(MPC_freq*dt_sim)) == 0:
            # Get the optimal control input
            u, xi_optimal, eta_optimal, cost_iter = controller.get_optimal_input(states[t, :], x_ref, u_guess)
            predicted_inputs[t, :, :] = u
            
            # Use the model to predict the future trajectory
            X_pred = states[t, :].copy()
            for k in range(c_horizon):
                X_pred = rk4_step(X_pred, predicted_inputs[t, k, :], dt_sim)
                predicted_states[t, k, :] = X_pred
            cost_evolution.append(cost_iter[-1])
            xi_evolution.append(xi_optimal[-1])
            eta_evolution.append(eta_optimal[-1])
        # Apply the control input to get the next state
        x_next = states[t + 1, :] = rk4_step(states[t, :], u[0,:], dt_sim)
        # Store the input and the next state
        states[t + 1, :] = x_next
        # Store the vertices position 
        vertices_inertial = []
        for vertice in vertices:
            vertice_inertial = np.dot(quaternion_to_rotation_matrix_numpy(states[t+1, 6:10]), vertice) + states[t+1, 0:3]
            vertices_inertial.append(vertice_inertial)

        inputs[t, :] = u[0, :]
        u_guess = np.tile(u[0,:], (c_horizon, 1)).reshape(c_horizon * 8, 1)
        #u_guess = np.tile(np.zeros(8), (c_horizon, 1)).reshape(c_horizon * 8, 1)
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
        "rho1 (Obstacle 1 Marging Slack)": rho1,
        "rho2 (Obstacle 2 Marging Slack)": rho2,
        "dt_mpc": dt_MPC,
        "simulation_time": simulation_time,
        "dt_sim": dt_sim,
        "Initial state x0": x0,
        "Obstacle 1 position": x_obstacle1,
        "Obstacle 1 radius": r_obstacle1,
        "Obstacle 2 position": x_obstacle2,
        "Obstacle 2 radius": r_obstacle2,
        "Static reference": x_ref_static,
        "Spacecraft radius": r_spacecraft
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

    # Ensure the output folder exists
    
    # Save the parameters in a .txt file inside the folder
    if plt_save:
        os.makedirs(output_folder, exist_ok=True)
        save_simulation_parameters(os.path.join(output_folder, "simulation_parameters.txt"))

    # Define time arrays for plotting
    time = np.linspace(0, simulation_time, num_steps + 1)
    time_inputs = np.linspace(0, simulation_time - dt_sim, num_steps)
    time_cost = np.linspace(0, simulation_time - dt_sim, len(cost_evolution))
    time_xi = np.linspace(0, simulation_time - dt_sim, len(xi_evolution))
    time_eta = np.linspace(0, simulation_time - dt_sim, len(eta_evolution))

    # Plot states
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 1, 1)
    plt.plot(time, states[:, 0], label='r_x')
    plt.plot(time, states[:, 1], label='r_y')
    plt.plot(time, states_euler[:, 0], label='yaw')
    plt.xlabel('Time [s]')
    plt.ylabel('States')
    plt.legend()
    plt.grid()

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

    # Plot obstacle margin slack history
    plt.figure(figsize=(12, 8))
    plt.plot(time_xi, xi_evolution)
    plt.xlabel('Time [s]')
    plt.ylabel('Slack Variable Value')
    plt.title('Slack Variable Evolution')
    plt.grid()

    if plt_save:
        plt.savefig(os.path.join(output_folder, 'slack_variable_plot.png'))

    # # Plot terminal cost slack history
    # plt.figure(figsize=(12, 8))
    # plt.plot(time_eta, eta_evolution)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Slack Variable Value')
    # plt.title('Terminal Cost Slack Variable Evolution')
    # plt.grid()

    if plt_save:
        plt.savefig(os.path.join(output_folder, 'terminal_cost_slack_plot.png'))

    # Plot trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(states[:, 0], states[:, 1])
    x_ref_evolution = np.array([target_dynamics(t) for t in range(num_steps)])
    x_ref_evolution = np.array(x_ref_evolution)
    plt.plot(x_ref_evolution[:, 0], x_ref_evolution[:, 1],"--")
    body1 = plt.Circle((x_obstacle1[0], x_obstacle1[1]), r_obstacle1, color='#303030', fill=True)
    circle1= plt.Circle((x_obstacle1[0], x_obstacle1[1]), r_obstacle1 * 2, color='#B7B6B6', linestyle='dotted' , fill=True)
    circle_exterior1 = plt.Circle((x_obstacle1[0], x_obstacle1[1]), r_obstacle1 * 2 + r_spacecraft, color='#B7B6B6', linestyle='dotted' , fill=False)
    body2 = plt.Circle((x_obstacle2[0], x_obstacle2[1]), r_obstacle2, color='#303030', fill=True)
    circle2= plt.Circle((x_obstacle2[0], x_obstacle2[1]), r_obstacle2 * 2, color='#B7B6B6', linestyle='dotted' , fill=True)
    circle_exterior2 = plt.Circle((x_obstacle2[0], x_obstacle2[1]), r_obstacle2 * 2 + r_spacecraft, color='#B7B6B6', linestyle='dotted' , fill=False)
    dot = plt.Circle((x_ref_static[0],x_ref_static[1]), 0.01, color = 'r', fill = True)

    plt.gca().add_patch(dot)
    plt.gca().add_patch(circle_exterior1)
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(body1)
    plt.gca().add_patch(circle_exterior2)
    plt.gca().add_patch(circle2)
    plt.gca().add_patch(body2)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory')
    plt.grid()

    if plt_save:
        plt.savefig(os.path.join(output_folder, 'trajectory_plot.png'))

    # Plot quaternions
    plt.figure(figsize=(8, 6))
    plt.plot(time, states[:, 6], label='q0')
    plt.plot(time, states[:, 7], label='q1')
    plt.plot(time, states[:, 8], label='q2')
    plt.plot(time, states[:, 9], label='q3')
    plt.xlabel('time')
    plt.ylabel('quaternions')
    plt.legend()
    plt.grid()

    if plt_save:
        plt.savefig(os.path.join(output_folder, 'quaternion_plot.png'))

    # Plot yaw vs. x
    plt.figure(figsize=(8, 6))
    plt.plot(states[:, 0], states_euler[:, 0], label='Yaw vs. X')
    plt.xlabel('x')
    plt.ylabel('yaw')
    plt.grid()

    if plt_save:
        plt.savefig(os.path.join(output_folder, 'yaw_vs_x_plot.png'))
        plt.close()
    if plt_show:
        plt.show()
    
def animate_trajectory():
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the obstacle
    body1 = plt.Circle((x_obstacle1[0], x_obstacle1[1]), r_obstacle1, color='#303030', fill=True)
    circle1= plt.Circle((x_obstacle1[0], x_obstacle1[1]), r_obstacle1 * 2, color='#B7B6B6', linestyle='dotted' , fill=True)
    circle_exterior1 = plt.Circle((x_obstacle1[0], x_obstacle1[1]), r_obstacle1 * 2 + r_spacecraft, color='#B7B6B6', linestyle='dotted' , fill=False)
    body2 = plt.Circle((x_obstacle2[0], x_obstacle2[1]), r_obstacle2, color='#303030', fill=True)
    circle2= plt.Circle((x_obstacle2[0], x_obstacle2[1]), r_obstacle2 * 2, color='#B7B6B6', linestyle='dotted' , fill=True)
    circle_exterior2 = plt.Circle((x_obstacle2[0], x_obstacle2[1]), r_obstacle2 * 2 + r_spacecraft, color='#B7B6B6', linestyle='dotted' , fill=False)
    dot = plt.Circle((x_ref_static[0],x_ref_static[1]), 0.01, color = 'r', fill = True)

    plt.gca().add_patch(dot)
    plt.gca().add_patch(circle_exterior1)
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(body1)
    plt.gca().add_patch(circle_exterior2)
    plt.gca().add_patch(circle2)
    plt.gca().add_patch(body2)

    # Initialize the spacecraft's trajectory plot and the square body plot
    trajectory, = ax.plot([], [], 'b-', label='Trajectory')
    
    # Dummy vertices to initialize the Polygon
    dummy_vertices = np.zeros((len(vertices), 2))
    body = plt.Polygon(dummy_vertices, closed=True, color='r', alpha=0.5, label='Spacecraft Body')
    ax.add_patch(body)
    
    # Set plot limits
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 15)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Spacecraft Trajectory and Body Animation')
    ax.grid(True)
    ax.legend()

    def init():
        trajectory.set_data([], [])
        body.set_xy(dummy_vertices)
        return trajectory, body,

    def update(frame):
        # Update trajectory
        trajectory.set_data(states[:frame, 0], states[:frame, 1])
        
        # Compute the new vertices in the inertial frame
        vertices_inertial = []
        for vertice in vertices:
            vertice_inertial = np.dot(quaternion_to_rotation_matrix_numpy(states[frame, 6:10]), vertice) + states[frame, 0:3]
            vertices_inertial.append(vertice_inertial[:2])
        
        body.set_xy(vertices_inertial)
        return trajectory, body,

    ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, repeat=False)
    
    # Display the animation
    plt.show()

if __name__ == "__main__":
    simulation()
    print("Process finished --- %s seconds ---" % (time.time() - start_time))
    output_folder = output_directory_creation()
    simulation_results_generation(output_folder)

    #animate_trajectory()

def run():
    simulation()
    print("Process finished --- %s seconds ---" % (time.time() - start_time))
    output_folder = output_directory_creation()
    simulation_results_generation(output_folder)
    
    


        
