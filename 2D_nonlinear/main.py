import numpy as np
import matplotlib.pyplot as plt
from mpc_controller import MPCController
from dynamics import rk4_step
from matplotlib.animation import FuncAnimation

# Constants
mass = 1
I = 1
f_thruster = 1
d = 1
u_min = 0
u_max = 1

# MPC Parameters
dt_MPC = 10
T_horizon = 120
c_horizon = 4
Q = np.diag([1, 1, 1, 1, 1, 1])   # State Weighting Matrix
R = 0.1 * np.eye(8)                     # Control weighting matrix
P = np.diag([10, 0, 10, 0, 0, 0]) # Terminal Cost Weighting Matrix
MPC_freq = 1

# Simulation parameters
simulation_time = 300  # Total simulation time in seconds
dt_sim = 1  # Time step
num_steps = int(simulation_time / dt_sim) # Number of simulation steps
x0 = np.array([0.5, 0, 0.5, 0, 0, 0]) # Initial state
static_reference = False
x_ref_static = np.array([3.5, 0, 3.5, 0, 0, 0]) # Reference state
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
        # x_ref[2] += np.sin((np.pi/4)*t)

        # Decaying Eliptical Motion
        x_ref[0] += 2 * np.exp(-0.01 * t) * np.cos(0.1*t)
        x_ref[2] += 3 * np.exp(-0.01 * t) * np.sin(0.1*t) 
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
        
    
    # Plotting
    time = np.linspace(0, simulation_time, num_steps + 1)
    x_ref_evolution = []
    for t in range(num_steps):
        x_ref_evolution.append(target_dynamics(t))


    plt.figure(figsize=(12, 8))

    # Plot states
    plt.subplot(1, 1, 1)
    plt.plot(time, states[:, 0], label='r_x')
    plt.plot(time, states[:, 2], label='r_y')
    plt.plot(time, states[:, 4], label='theta')
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
    plt.show()

    # Plot cost history
    plt.figure(figsize=(12, 8))
    time_cost = np.linspace(0, simulation_time - dt_sim, len(cost_evolution))
    plt.plot(time_cost, cost_evolution)
    plt.xlabel('Time [s]')
    plt.ylabel('Cost')
    plt.title('Cost Evolution')
    plt.grid()
    plt.show()

    # Plot trajectory
    
    plt.figure(figsize=(8, 6))
    plt.plot(states[:, 0], states[:, 2])
    x_ref_evolution = np.array(x_ref_evolution)
    plt.plot(x_ref_evolution[:, 0], x_ref_evolution[:, 2],"--")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory')
    plt.grid()
    plt.show()

# def animate_trajectory(states, simulation_time, dt_sim):
#     # Create a figure and axis
#     fig, ax = plt.subplots(figsize=(8, 6))
    
#     # Set up plot limits
#     ax.set_xlim(np.min(states[:, 0]) - 10, np.max(states[:, 0]) + 10)
#     ax.set_ylim(np.min(states[:, 2]) - 10, np.max(states[:, 2]) + 10)
    
#     # Plot the reference trajectory
#     ax.plot(states[:, 0], states[:, 2], 'r--', label='Reference Trajectory')
    
#     # Initialize a line for the trajectory and a point for the current position
#     line, = ax.plot([], [], 'b-', label='Trajectory')
#     point, = ax.plot([], [], 'bo', label='Current Position')
    
#     # Initialize an empty list for the arrows
#     arrows = []

#     def init():
#         line.set_data([], [])
#         point.set_data([], [])
#         return line, point
    
#     def update(frame):
#         nonlocal arrows
        
#         # Clear previous arrows
#         for arrow in arrows:
#             arrow.remove()
#         arrows = []
        
#         # Update the trajectory line
#         line.set_data(states[:frame, 0], states[:frame, 2])
        
#         # Update the current position point
#         point.set_data(states[frame, 0], states[frame, 2])
        
#         # Update the arrow to represent orientation
#         x, y = states[frame, 0], states[frame, 2]
#         theta = states[frame, 4]
#         dx = np.cos(theta)
#         dy = np.sin(theta)
        
#         # Add a new arrow to the plot
#         arrow = ax.arrow(x, y, dx * 5, dy * 5, head_width=2, head_length=2, fc='g', ec='g')
#         arrows.append(arrow)
        
#         return line, point, *arrows
    
#     # Time array
#     time = np.linspace(0, simulation_time, int(simulation_time/dt_sim) + 1)
    
#     # Create the animation
#     anim = FuncAnimation(fig, update, frames=len(time), init_func=init, blit=True, interval=dt_sim*1000, repeat=False)
    
#     # Show the animation
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend()
#     plt.grid()
#     plt.show()

if __name__ == "__main__":
    main()
    # animate_trajectory(states, simulation_time, dt_sim)
