# 2D Translation and Path Following Scenario
# Set static_reference = True for Translation; static_reference = Flag for Path Following.

import numpy as np
import matplotlib.pyplot as plt
from mpc_controller import MPCController
from dynamics import rk4_step
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import time
import os
from datetime import datetime

# Start Timer
start_time = time.time()

# Flags for plotting
plt_save = True # Save the plots
plt_show = True # Show the plots

static_reference = False

# Constants
mass = 1
I = 1
f_thruster = 1
d = 1
u_min = 0
u_max = 1

# MPC Parameters
dt_MPC = 1
T_horizon = 12
c_horizon = 4
Q = np.diag([100, 100, 100, 100, 10, 100])   # State Weighting Matrix
R = 1e-2 * np.eye(8)                     # Control weighting matrix
P = np.diag([1000, 1000, 1000, 1000, 1000, 1000]) # Terminal Cost Weighting Matrix
MPC_freq = 1

# Simulation parameters
simulation_time = 150  # Total simulation time in seconds
dt_sim = 1  # Time step
num_steps = int(simulation_time / dt_sim) # Number of simulation steps
x0 = np.array([0.5, 0, 0.5, 0, 0, 0]) # Initial state

x_ref_static = np.array([4.0, 0, 2.0, 0, 0, 0]) # Reference state
x_ref_dyn_initial = np.array([0.5, 0, 0.5, 0, 0, 0]) # Reference Trajectory Intial State

# Storage for states and inputs
states = np.zeros((num_steps + 1, 6))
inputs = np.zeros((num_steps, 8))

# Set initial state
states[0, :] = x0
cost_evolution=[]

def target_dynamics(t):
    if static_reference == True:
        x_ref = x_ref_static
    else:
        x_ref = x_ref_dyn_initial.copy()
        # Simple Tranlastion
        # x_ref[0] += (0.1) * t 
        # x_ref[2] += (0.1) * t

        # Circular Motion
        # x_ref[0] += np.cos((np.pi/4)*t)
        x_ref[0] += 0.1 * t
        x_ref[2] += 0.1 * np.sin((np.pi/16)*t)

        # Decaying Eliptical Motion
        # x_ref[0] += 2 * np.exp(-0.01 * t) * np.cos(0.1*t)
        # x_ref[2] += 3 * np.exp(-0.01 * t) * np.sin(0.1*t) 
    return x_ref

def main():
    controller = MPCController(T_horizon, c_horizon, mass, d, dt_MPC, Q, R, P, u_min, u_max)
    u_guess = np.zeros((c_horizon * 8, 1))
    
    # Simulate the system
    for t in range(num_steps):
        x_ref = target_dynamics(t)
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

def compute_rmse(states, references):
    errors = states - references
    mse = np.mean(errors**2, axis=0)
    rmse = np.sqrt(mse)
    return rmse

def compute_rmse_over_time(states, references):
    errors = states - references
    mse_t = np.mean(errors**2, axis=1)
    rmse_t = np.sqrt(mse_t)
    return rmse_t


def plots(output_folder):
    # Plotting
    time = np.linspace(0, simulation_time, num_steps + 1)
    x_ref_evolution = []
    for t in range(num_steps):
        x_ref_evolution.append(target_dynamics(t))


    plt.figure(figsize=(12, 8))

    # Plot states
    plt.subplot(1, 1, 1)
    plt.plot(time, states[:, 0], label='x', color = "#00aff5")
    plt.plot(time, states[:, 2], label='y', color = "#E21612")
    plt.plot(time, states[:, 4], label='theta', color = "#23ce6b")
    plt.xlabel('Time [s]')
    plt.ylabel('States')
    plt.legend()
    plt.grid()

    if plt_save:
        plt.savefig(os.path.join(output_folder, 'state_plot.pdf'), format='pdf')

    # Create figure and axis
    fig, ax1 = plt.subplots()

    # Plot x and y on primary y-axis
    ax1.plot(time, states[:, 0], label='x', color="#00aff5")
    ax1.plot(time, states[:, 2], label='y', color="#E21612")
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('States (m)')
    #ax1.legend(loc='upper left')
    ax1.grid()

    # Add secondary y-axis for theta in degrees
    def rad_to_deg(x):
        return np.degrees(x)

    def deg_to_rad(x):
        return np.radians(x)

    ax2 = ax1.secondary_yaxis('right', functions=(rad_to_deg, deg_to_rad))
    ax2.set_ylabel('Theta [°]')

    # Plot theta on primary axis but with a different scale
    ax1.plot(time, states[:, 4], label='theta (radians)', color="#23ce6b")

    plt.title("State Evolution with Angle in Degrees")
    plt.show()

    # #Plot inputs
    plt.figure(figsize=(12, 8))
    time_inputs = np.linspace(0, simulation_time - dt_sim, num_steps)
    for i in range(1, 9): #change 5 to 9 when full thrusters
        plt.subplot(5, 2, i)
        plt.step(time_inputs, inputs[:, i-1], label=f'Thruster {i}', color='#5F758E')
        plt.xlabel('Time [s]')
        plt.ylabel('Magnitude [N]')
        plt.legend()
        plt.grid()

    u_cumsum = np.cumsum(np.sum(inputs, axis=1), axis=0)
    plt.subplot(5, 1, 5)
    plt.step(time_inputs, u_cumsum, color='#5F758E')
    plt.xlabel('Time [s]')
    plt.ylabel('Total input [N]')
    plt.legend()
    plt.grid()

    plt.tight_layout()

    # if plt_save:
    #     plt.savefig(os.path.join(output_folder, 'inputs_plot.pdf'), format='pdf')

    plt.figure(figsize=(12,6))
    plt.subplot(2, 1, 1)
    plt.step(time_inputs, inputs[:,5], color='#5F758E')
    plt.ylabel('Total input [N]')
    plt.grid()
    plt.subplot(2,1,2)
    plt.step(time_inputs, u_cumsum, color='#5F758E')
    plt.xlabel('Time [s]')
    plt.ylabel('Total input [N]')
    plt.grid()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    # Plot the first graph on the first subplot
    ax1.step(time_inputs, inputs[:, 2], color='#5F758E')
    ax1.set_ylabel('Total input [N]')
    ax1.grid()

    # Plot the second graph on the second subplot
    ax2.step(time_inputs, u_cumsum, color='#5F758E')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Cumulative input [N]')
    ax2.grid()

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # # Display the plot
    # plt.show()

    # if plt_save:
    #     plt.savefig(os.path.join(output_folder, 'inputs_total.pdf'), format='pdf')

    # Plot cost history
    # plt.figure(figsize=(12, 8))
    # time_cost = np.linspace(0, simulation_time - dt_sim, len(cost_evolution))
    # plt.plot(time_cost, cost_evolution, color='#5F758E')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Cost')
    # plt.title('Cost Evolution')
    # plt.grid()

    # if plt_save:
    #     plt.savefig(os.path.join(output_folder, 'cost_evolution_plot.pdf'), format='pdf')

    # Plot trajectory
    
    plt.figure(figsize=(8, 6))
    plt.plot(states[:, 0], states[:, 2], 'o', ls='-', ms=4, markevery=[0], color='k', label='Vehicle Trajectory')
    #plt.scatter(states[-1, 0],states[-1, 2], marker='>', facecolors = 'k', edgecolors='k', s=40)
    
    plt.xlabel('position (x) [m]')
    plt.ylabel('position (y) [m]')
    if static_reference == True:
        plt.title('Translation Scenario')
        plt.scatter(x_ref_static[0], x_ref_static[2],facecolors = '#e21612', edgecolors= '#e21612', s=80)
    else:
        plt.title('Path Following Scenario')
        x_ref_evolution = np.array([target_dynamics(t) for t in range(num_steps + 1)])
        plt.plot(x_ref_evolution[:, 0],x_ref_evolution[:, 2],'o', color= '#ee3432', label = 'Reference Trajectory', ls='--', ms=4,markevery=[0])
        #plt.scatter(x_ref_evolution[-1, 0],x_ref_evolution[-1, 2], marker='>', facecolors = '#ee3432', edgecolors='#ee3432', s=40)
    plt.legend(loc = "lower right")  # Add legend to the plot
    plt.grid()

    if plt_save:
        plt.savefig(os.path.join(output_folder, 'trajectory_plot.pdf'), format = 'pdf')

    if plt_show:
        plt.show()

    rmse = compute_rmse(states, x_ref_evolution)
    # Plot RMSE
    plt.figure(figsize=(8, 5))
    labels = ['x', 'vx', 'y', 'vy', 'theta', 'omega']
    colors = ['#00aff5', '#E21612', '#23ce6b', '#F5B700', '#F9D4BB', '#A52A2A']
    for i in range(6):
        plt.bar(i, rmse[i], color=colors[i])
    plt.xticks(range(6), labels)
    plt.ylabel("RMSE")
    plt.title("Root Mean Square Error (RMSE) per State Component")
    plt.grid(axis='y')

    if plt_save:
        plt.savefig(os.path.join(output_folder, 'rmse_plot.pdf'), format='pdf')

    plt.show()

    rmse_time = compute_rmse_over_time(states, x_ref_evolution)

    plt.figure(figsize=(8, 4))
    plt.plot(time, rmse_time, label="RMSE over time", color="#A52A2A")
    plt.xlabel("Time [s]")
    plt.ylabel("RMSE")
    plt.title("RMSE Evolution Over Time")
    plt.grid(True)
    plt.tight_layout()

    if plt_save:
        plt.savefig(os.path.join(output_folder, 'rmse_evolution_plot.pdf'), format='pdf')

    plt.show()

# def animate_trajectory():
#     fig, ax = plt.subplots(figsize=(8, 6))
#     trajectory, = ax.plot([], [], color='k', label='Trajectory')
#     start_point = ax.plot(states[0, 0], states[0, 2], 'ko', label='Start Point')[0]

#     if static_reference:
#         target_point = ax.plot(x_ref_static[0], x_ref_static[2], 'ro', label='Target Point')[0]
#     else:
#         target_point = ax.plot([], [], 'ro', label='Target Point')[0]

#     time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

#     ax.set_xlim(0, 4.5)
#     ax.set_ylim(0, 2.25)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_title('Example II - Translation with Rotation')
#     ax.grid(True, linewidth=0.5)
#     ax.legend()

#     def init():
#         trajectory.set_data([], [])
#         time_text.set_text('')
#         return trajectory, time_text, target_point

#     def update(frame):
#         trajectory.set_data(states[:frame, 0], states[:frame, 2])
#         time_text.set_text(f'Time: {frame * dt_sim:.1f} s')

#         if not static_reference:
#             x_ref = target_dynamics(frame)
#             target_point.set_data(x_ref[0], x_ref[2])

#         return trajectory, time_text, target_point

#     anim = FuncAnimation(fig, update, frames=np.arange(1, num_steps), init_func=init,
#                          blit=True, interval=50)

#     if plt_show:
#         plt.show()

#     if plt_save:
#         output_folder = output_directory_creation()
#         anim.save(os.path.join(output_folder, 'trajectory_animation.mp4'), writer='ffmpeg')

def animate_trajectory():
    fig, ax = plt.subplots(figsize=(8, 6))
    trajectory, = ax.plot([], [], color='k', label='Trajectory')
    start_point = ax.plot(states[0, 0], states[0, 2], 'ko', label='Start Point')[0]

    if static_reference:
        target_point = ax.plot(x_ref_static[0], x_ref_static[2], 'ro', label='Target Point')[0]
    else:
        target_point = ax.plot([], [], 'ro', label='Target Point')[0]

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    ax.set_xlim(0, 16)
    ax.set_ylim(0.35, 0.65)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Example II - Sinusoidal Trajectory')
    ax.grid(True, linewidth=0.5)
    ax.legend()

    # Orientation line and dot (initialized once)
    orientation_line, = ax.plot([], [], 'k-', lw=1)
    orientation_dot, = ax.plot([], [], 'ko', markersize=3)

    def init():
        trajectory.set_data([], [])
        orientation_line.set_data([], [])
        orientation_dot.set_data([], [])
        time_text.set_text('')
        return trajectory, time_text, target_point, orientation_line, orientation_dot

    def update(frame):
        x, y, theta = states[frame, 0], states[frame, 2], states[frame, 4]
        trajectory.set_data(states[:frame, 0], states[:frame, 2])
        time_text.set_text(f'Time: {frame * dt_sim:.1f} s')

        # Orientation line and dot
        length = 0.15
        x_end = x + length * np.cos(theta)
        y_end = y + length * np.sin(theta)

        orientation_line.set_data([x, x_end], [y, y_end])
        orientation_dot.set_data([x_end], [y_end])  # Wrap scalars in a list

        if not static_reference:
            x_ref = target_dynamics(frame)
            target_point.set_data([x_ref[0]], [x_ref[2]])  # Wrap scalars in a list

        return trajectory, time_text, target_point, orientation_line, orientation_dot

    anim = FuncAnimation(fig, update, frames=np.arange(1, num_steps), init_func=init,
                         blit=True, interval=70)

    if plt_show:
        plt.show()

    if plt_save:
        output_folder = output_directory_creation()
        anim.save(os.path.join(output_folder, 'trajectory_animation.mp4'), writer='ffmpeg')



if __name__ == "__main__":
    main()
    print("Process finished --- %s seconds ---" % (time.time() - start_time))
    output_folder = output_directory_creation()
    #plots(output_folder)
    animate_trajectory()

