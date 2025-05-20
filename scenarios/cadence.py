## Entry Angle v2 scenario with obstacle avoidance ##

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import os
import time
from datetime import datetime
from src.controllers.mpc_cadence import MPCController
from src.dynamics.dynamics_3d import rk4_step
from matplotlib.animation import FuncAnimation
from src.util.quat2eul import np_quaternion_to_euler
from src.util.eul2quat import euler_to_quaternion
from src.util.quaternion_rotation import quaternion_to_rotation_matrix_numpy, pos_prime_rot_numpy
from src.util.quaternion_update import quaternion_update_np

# Start Timer
start_time = time.time()

# Flags for plotting
plt_save = False # Save the plots
plt_show = True # Show the plots

# Constants
mass = 15 #[kg]
f_thruster = 1 # Maximum Thrust [N]
dx, dy = 1, 1 # Distance from the CoM to the Thrusters [m]
u_min, u_max = 0, 1 # Thrust Bounds
L = 0.2828 # Length of the Robot (L x L) 
r_chaser = r_target = 0.5 * (L * np.sqrt(2)) # For approximating the shape of the sc to a circle
chaser_vector_C = np.array([1,0,0]) # Chaser Unity Direction Vector in the
target_vector_T = np.array([0,-1,0]) # Target Unity Direction Vector in the N
Ixx, Iyy = 1, 1
Izz = 0.5 * mass * (r_chaser**2) # Approximated to a cylinder shape
I = np.array([[Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz]])
theta1 = 135.0
theta2 = 45.0

# Test bed size coordinates
test_bed = np.array([
    [0, 0, 0],
    [4.10, 0, 0],
    [4.10, 3.32, 0],
    [0, 3.32, 0]
])

# Inner Part of the robot vertices coordinates
vertices = np.array([
    [-L/2, -L/2, 0],
    [L/2, -L/2, 0],
    [L/2, L/2, 0],
    [-L/2, L/2, 0]
])

# Probe (Male component of the dokcing) dimensions and coordinates
probe_l = 0.1
probe_w = 0.05
probe = np.array([
    [r_chaser, -probe_w, 0],
    [r_chaser, probe_w, 0],
    [r_chaser + probe_l, probe_w,0],
    [r_chaser + probe_l,-probe_w,0]
])
 
# Droge (Female component of the dokcking) dimensions and coordinates
drogue_w = 0.1
drogue = np.array([
    [-drogue_w, r_target, 0],
    [drogue_w, r_target, 0],
    [0, L/4, 0],
])

# MPC Parameters
dt_mpc = 1
t_horizon = 30 # Prediction Horizon = t_horizon / dt_mpc #13
c_horizon = 6 # Control Horizon #4
Q = 0 * np.eye(13) #wh State Weighting Matrix #1e1
Q[0,0] = 1 * 1e2
Q[1,1] = 1 * 1e2
Q[3,3] = 1 * 1e4 #1e4
Q[4,4] = 1 * 1e4 #1e4
#Q[10:13,10:13] = 0 * np.eye(3) #1e0
Q[6:10,6:10] = 0 * np.eye(4) #0
Q[10:13,:] = 0
R = 1*1e8 * np.eye(8)   # Control Weighting Matrix #1e0
P = 1 * 1e43 * np.eye(13) # Terminal Cost Weighting Matrix #1e1
P[1,1] = 1 * 1e3
P[0,0] = 1 * 1e3
#P[6:10,6:10] = 0 * np.eye(4) #0
#P[10:13,10:13] = 0 * np.eye(3)
sigma = 1 * 1e2
gamma = 1 * 1e0 * np.eye(3)
rho = 1e0
mpc_freq = 1

# Simulation parameters
simulation_time = 500  # Total simulation time in seconds
dt_sim = 0.1  # Time step
num_steps = int(simulation_time / dt_sim) # Number of simulation steps
x0 = np.zeros(13) # Initial State Initialization
x0[0:2] = [0.6, 0.6]
x0[6:10] = euler_to_quaternion(0,0,0)
x0[10:13] = [0.0, 1e-24, 0] # one of the terms of the angular velocity still has to be > 0 due to division by 0 errors

# Obstacles Parameters
x_obstacle = [1.5, 1.5, 0]
r_obstacle = r_chaser


predicted_states = np.zeros((num_steps, c_horizon, 13)) # Initialization for predicted state and inputs evolution
predicted_inputs = np.zeros((num_steps, c_horizon, 8))

# Static Reference Scenario Parameters
x_ref_static = np.zeros(13) 
x_ref_static[0:2] = [3, 2]
x_ref_static[6:10] = euler_to_quaternion(0, 0, 0) # Yaw, Pitch = 0, Roll = 0
x_ref_static[10:13] = [0, 1e-24, 0]

# Dynamic Reference Trajectory / Path Following Parameters

x_ref_dyn_initial = np.zeros(13)
x_ref_dyn_initial[0:3] = [3, 2, 0]
x_ref_dyn_initial[6:10] = euler_to_quaternion(0,0,0)
x_ref_dyn_initial[10:13] = [0, 1e-24, 0.000]
x_docking_initial = x_ref_dyn_initial.copy()
x_docking_initial[0:3] = [3, 2+r_target+r_chaser, 0]

# Path Following Condition (True -> Static Reference Target, False -> Path Following Scenario)
static_reference = False

# Target x_ref definition
def target_dynamics(t):

    if static_reference == True:
        x_ref = x_ref_static.copy()
        x_target = x_ref.copy()
        x_docking = x_docking_initial.copy()

    else:
        x_ref = x_ref_dyn_initial.copy()
        x_target = x_ref.copy()
        x_docking = x_docking_initial.copy()
        x_target[6:10] = quaternion_update_np(x_ref_dyn_initial[6:10], x_ref_dyn_initial[10:13], t)
        x_ref[10:13] = [0, 1e-24, 0] 
        x_docking[0:3] = pos_prime_rot_numpy(x_target[6:10], x_docking[0:3])
    return x_ref, x_target, x_docking

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
theta_evolution = []

def simulation():
    controller = MPCController(t_horizon, c_horizon, mass, I, dx, dy, dt_mpc, Q, R, P, u_min, u_max, sigma, r_chaser, r_target, chaser_vector_C, target_vector_T, gamma, x_obstacle, r_obstacle, rho)
    u_guess = np.zeros((c_horizon * 8, 1))

    # Simulate the system
    for t in range(num_steps):
        x_ref, x_target, x_docking = target_dynamics(t)
        #print(x_docking)
        #print(x_ref)
        #print(x_target)

        theta_evolution.append(np_quaternion_to_euler(x_ref[6:10])[2])
        if t % int(1/(mpc_freq*dt_sim)) == 0:
            # Get the optimal control input
            u, eta_optimal, cost_iter = controller.get_optimal_input(states[t, :], x_ref, x_target, x_docking, u_guess)
            predicted_inputs[t, :, :] = u
            
            # Use the model to predict the future trajectory
            X_pred = states[t, :].copy()
            for k in range(c_horizon):
                X_pred = rk4_step(X_pred, predicted_inputs[t, k, :], dt_sim)
                predicted_states[t, k, :] = X_pred
            cost_evolution.append(cost_iter[-1])
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

        inputs[t, :] = u[0,:]
        #u_guess = np.tile(u[0,:], (c_horizon, 1)).reshape(c_horizon * 8, 1)
        u_guess = np.array(u[0:c_horizon,:]).reshape(c_horizon * 8, 1)
        states_euler[t + 1, :] = np_quaternion_to_euler(x_next[6:10])

def save_simulation_parameters(filename):
    params = {
        "mass": mass,
        "Ixx": Ixx,
        "Iyy": Iyy,
        "Izz": Izz,
        "Thrust bounds": [u_min, u_max],
        "MPC Horizon": t_horizon,
        "Control Horizon": c_horizon,
        "State Weighting Matrix Q": Q,
        "Control Weighting Matrix R": R,
        "Terminal Cost Weighting Matrix P": P,
        "dt_mpc": dt_mpc,
        "simulation_time": simulation_time,
        "dt_sim": dt_sim,
        "Initial state x0": x0,
        "Static reference": x_ref_static,
        "Spacecraft radius": r_chaser
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

    # Plot vel
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 1, 1)
    plt.plot(time, states[:, 3], label='v_x')
    plt.plot(time, states[:, 4], label='v_y')
    plt.xlabel('Time [s]')
    plt.ylabel('Vel')
    plt.legend()
    plt.grid()

    if plt_save:
        plt.savefig(os.path.join(output_folder, 'vel_plot.png'))

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

    # # Plot obstacle margin slack history
    # plt.figure(figsize=(12, 8))
    # plt.plot(time_xi, xi_evolution)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Slack Variable Value')
    # plt.title('Slack Variable Evolution')
    # plt.grid()

    # if plt_save:
    #     plt.savefig(os.path.join(output_folder, 'slack_variable_plot.png'))

    # # Plot terminal cost slack history
    # plt.figure(figsize=(12, 8))
    # plt.plot(time_eta, eta_evolution)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Slack Variable Value')
    # plt.title('Terminal Cost Slack Variable Evolution')
    # plt.grid()

    # if plt_save:
    #     plt.savefig(os.path.join(output_folder, 'terminal_cost_slack_plot.png'))

    # Plot trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(states[:, 0], states[:, 1], color = "cornflowerblue",label='Trajectory')
    x_ref_evolution = np.array([target_dynamics(t) for t in range(num_steps)])
    plt.plot(x_ref_evolution[:, 0], x_ref_evolution[:, 1], "--", label='Reference Path')
    # plt.plot([x_ref[0], x_ref[0] + 1], [x_ref[1], x_ref[1] + 1], 'r--', label='Static Reference')
    # plt.plot([x_ref[0], x_ref[0] - 1], [x_ref[1], x_ref[1] + 1], 'r--')
    # dot = plt.Circle((x_ref[0], x_ref[1]), 0.01, color='r', fill=True)
    # plt.gca().add_patch(dot)
    body1 = plt.Circle((x_obstacle[0], x_obstacle[1]), r_obstacle, color='#303030', fill=True)
    circle1= plt.Circle((x_obstacle[0], x_obstacle[1]), r_obstacle * 2, color='#B7B6B6', linestyle='dotted' , fill=True)
    circle_exterior1 = plt.Circle((x_obstacle[0], x_obstacle[1]), r_obstacle * 2 + r_chaser, color='#B7B6B6', linestyle='dotted' , fill=False)
    plt.gca().add_patch(circle_exterior1)
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(body1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory')
    plt.grid()

    if plt_save:
        plt.savefig(os.path.join(output_folder, 'trajectory_plot.png'))

    # # Plot quaternions
    # plt.figure(figsize=(8, 6))
    # plt.plot(time, states[:, 6], label='q0')
    # plt.plot(time, states[:, 7], label='q1')
    # plt.plot(time, states[:, 8], label='q2')
    # plt.plot(time, states[:, 9], label='q3')
    # plt.xlabel('time')
    # plt.ylabel('quaternions')
    # plt.legend()
    # plt.grid()

    # if plt_save:
    #     plt.savefig(os.path.join(output_folder, 'quaternion_plot.png'))

    # # Plot yaw vs. x
    # plt.figure(figsize=(8, 6))
    # plt.plot(states[:, 0], states_euler[:, 0], label='Yaw vs. X')
    # plt.xlabel('x')
    # plt.ylabel('yaw')
    # plt.grid()

    # if plt_save:
    #     plt.savefig(os.path.join(output_folder, 'yaw_vs_x_plot.png'))
    #     plt.close()
    if plt_show:
        plt.show()
    
def animate_trajectory():
    fig, ax = plt.subplots(figsize=(8, 6))

    # Initialize the spacecraft's trajectory plot and the square body plot
    trajectory, = ax.plot([], [], color = 'k', label='Trajectory') #darkblue
    line1, = ax.plot([], [], color = "#D33E43" ,linestyle = 'dashed', label='Entry Angle Cone')
    line2, = ax.plot([], [], color = "#D33E43", linestyle = 'dashed')
    
    # Dummy vertices to initialize the Polygon
    dummy_vertices = np.zeros((len(vertices), 2))
    dummy_probe = np.zeros((len(probe),2))
    dummy_drogue = np.zeros((len(drogue),2))
    dummy_states = np.zeros(2)
    dummy_target_center = np.zeros(2)
    theta1 = 135.0
    theta2 = 45.0
    
    
    inner_body_chaser = plt.Polygon(dummy_vertices, closed=True, color='#28536B', alpha=1, label='Chaser Agent') #darkgray #alpha1
    outer_body_chaser = plt.Circle(dummy_states[0:2], radius= r_chaser, fill = True, color= '#7EA8BE', alpha = 0.3,label = 'Outer Cilinder') #k, 0.1
    target_body = plt.Circle(dummy_target_center[0:2], radius=r_target, fill = True, color= 'Grey', alpha = 0.3, label = 'Target Agent') #k
    #chaser_probe, = ax.plot([],[], 'k-', label="Probe")
    chaser_probe = plt.Polygon(dummy_probe, closed=True, color='k', alpha=0.5, label='Probe') #k 0.5
    target_drogue = plt.Polygon(dummy_drogue, closed=True, color='k', alpha=1, label='Drogue') #k 0.5
    test_bed_patch = plt.Rectangle(test_bed[0,:2], 4.10, 3.32, fill=False, edgecolor='k', label='Test Bed') #k 0.5
    #circle = plt.Circle(x_ref_dyn_initial[0:2], radius = 2 * (r_chaser + r_target), fill = False, color = 'k')
    #arc = Arc(x_ref_dyn_initial[0:2], 2 * (r_chaser + r_target), 2 * (r_chaser + r_target), theta1=135.0, theta2=45.0)
    arc = Arc(x_ref_dyn_initial[0:2], 2 * (r_chaser + r_target), 2 * (r_chaser + r_target), theta1=theta1, theta2=theta2)

    body1 = plt.Circle((x_obstacle[0], x_obstacle[1]), r_obstacle, color='#303030', fill=True)


    plt.gca().add_patch(body1)

    ax.add_patch(test_bed_patch)

    ax.add_patch(target_drogue)
    ax.add_patch(inner_body_chaser)
    ax.add_patch(outer_body_chaser)
    ax.add_patch(target_body)
    
    ax.add_patch(chaser_probe)
    #ax.add_patch(circle)
    ax.add_patch(arc)
    
    
    
    # Set plot limits
    ax.set_xlim(-1, 5.10)
    ax.set_ylim(-1, 4.32)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Spacecraft Trajectory and Body Animation')
    ax.grid(True, linewidth = 0.5)
    ax.legend()

    # Create two lines representing the entry angle cone
    

    def init():
        trajectory.set_data([], [])
        inner_body_chaser.set_xy(dummy_vertices)
        outer_body_chaser.set_center((dummy_states[0], dummy_states[1]))
        target_body.set_center((dummy_target_center[0], dummy_target_center[1]))
        chaser_probe.set_xy(dummy_probe)
        target_drogue.set_xy(dummy_drogue)
        arc.theta1 = theta1
        arc.theta2 = theta2
        
        #chaser_probe.set_data([], [])
        line1.set_data([], [])
        line2.set_data([], [])
    
        return trajectory, inner_body_chaser, outer_body_chaser, chaser_probe, line1, line2, target_body, target_drogue, arc

    def update(frame):
        global theta1, theta2
        # Update trajectory
        trajectory.set_data(states[:frame, 0], states[:frame, 1])
        # for t in range(num_steps):
        #     x_ref, x_target, x_docking = target_dynamics(t)
        
        x_ref, x_target, x_docking = target_dynamics(frame)  # Extract reference position
        #print(x_ref, x_target, x_docking)
        # Compute the new vertices in the inertial frame for spacecraft body
        vertices_inertial = []
        rotation_matrix = quaternion_to_rotation_matrix_numpy(states[frame, 6:10])  # Make sure this is 2D
        angular_rate_deg = np.rad2deg(x_target[12])
        theta1 += angular_rate_deg
        theta2 += angular_rate_deg

        # theta1 -= 0
        # theta2 -= 0



        # theta1 = theta1 % 360
        # theta2 = theta2 % 360

        for vertice in vertices:
            vertice_inertial = np.dot(quaternion_to_rotation_matrix_numpy(states[frame, 6:10]), vertice) + states[frame, 0:3]
            vertices_inertial.append(vertice_inertial[:2])

        probe_inertial = []
        for point in probe:
            probe_in = np.dot(quaternion_to_rotation_matrix_numpy(states[frame, 6:10]), point) + states[frame, 0:3]
            probe_inertial.append(probe_in[:2])

        drogue_inertial = []
        for point in drogue:
            drogue_in = np.dot(quaternion_to_rotation_matrix_numpy(x_target[6:10]), point) + x_target[0:3]
            drogue_inertial.append(drogue_in[:2])
        
        chaser_center = [states[frame,0], states[frame,1]]
        target_center = [x_target[0], x_target[1]]
        inner_body_chaser.set_xy(vertices_inertial)
        chaser_probe.set_xy(probe_inertial)
        outer_body_chaser.set_center(chaser_center)
        #outer_body_chaser.set_center([3,2])
        arc.theta1 = theta1
        arc.theta2 = theta2

        #chaser_probe.set_data([probe_inertial[0]], [probe_inertial[1]])

        # Entry angle cone lines update
        
        target_position = x_target[0:2]
        #target_body.set_center(target_center)
        target_body.set_center(target_center)
        target_drogue.set_xy(drogue_inertial)
        x_prime = states[frame, 0] - x_target[0]
        y_prime = states[frame, 1] - x_target[1]
        pos_prime = np.array([x_prime,y_prime,states[frame,2]])

        T_to_N_rotation_matrix = quaternion_to_rotation_matrix_numpy(x_target[6:10])
        x_prime_rotated, y_prime_rotated, z_rotated = np.dot(rotation_matrix,pos_prime)
        # Compute the two lines (cone boundaries) from x_ref
        length = 5  # Adjust length of the cone lines

        y_prime_rotated = x_prime_rotated

        x1 = np.linspace(x_ref[0], x_ref[0] + 1)
        x2 = np.linspace(x_ref[0] - 1, x_ref[0])
        #y1 = x1 - 0.5 * x_ref[1]
        #y1 = x1 - (0.5 * x_ref[1])
        #y1 = x1 - 1
        #y1 = ((2 * states[frame,6] * states[frame,9]) / (states[frame,6]**2 - states[frame,9]**2)) * x1 - x_ref[1]
        z = x1 * 0

        #y2 = -x2 + 2 * x_ref[0] - 0.5 * x_ref[1]
        #y2 = -x2 + 1*x_ref[0] - 0.5 * x_ref[1]
        #y2 = -((2 * states[frame,6] * states[frame,9]) / (states[frame,6]**2 - states[frame,9]**2)) * x1 - x_ref[1]

        # x1_shifted = x1 - x_target[0]
        # x2_shifted = x2 - x_target[0]
        # y1_shifted = y1 - x_target[1]
        # y2_shifted = y2 - x_target[1]

        # rotated_line1 = T_to_N_rotation_matrix @ np.vstack((x1_shifted, y1_shifted, z))
        # rotated_line2 = T_to_N_rotation_matrix @ np.vstack((x2_shifted, y2_shifted, z))
        # rotated_line1 = T_to_N_rotation_matrix @ np.vstack((x1, y1, z))
        # rotated_line2 = T_to_N_rotation_matrix @ np.vstack((x2, y2, z))

        #line1.set_data(rotated_line1[0] + x_ref[0], rotated_line1[1] + x_ref[1])
        #line2.set_data(rotated_line2[0] + x_ref[0], rotated_line2[1] + x_ref[1])
        # line1.set_data(x1, y_prime_rotated + x_ref[1])
        # line2.set_data(x1, y_prime_rotated + x_ref[1])
        
        return trajectory, inner_body_chaser, outer_body_chaser, chaser_probe, line1, line2, target_body, target_drogue, arc


    anim = FuncAnimation(fig, update, frames=np.arange(1, num_steps), init_func=init, blit=True, interval = 5)
    
    if plt_show:
        plt.show()

    if plt_save:
        output_folder = output_directory_creation()
        anim.save(os.path.join(output_folder, 'trajectory_animation.mp4'), writer='ffmpeg')



if __name__ == "__main__":

    simulation()
    print("Process finished --- %s seconds ---" % (time.time() - start_time))
    output_folder = output_directory_creation()
    simulation_results_generation(output_folder)
    animate_trajectory()

def run():
    simulation()
    print("Process finished --- %s seconds ---" % (time.time() - start_time))
    output_folder = output_directory_creation()
    simulation_results_generation(output_folder)
    animate_trajectory()