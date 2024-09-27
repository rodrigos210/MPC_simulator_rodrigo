import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time
from src.controllers.mpc_controller_full import MPCController
from src.dynamics.dynamics_3d import rk4_step
from src.util.quat2eul import quaternion_to_euler
from src.util.eul2quat import euler_to_quaternion
from src.util.quaternion_rotation import quaternion_to_rotation_matrix_numpy

# Start timer
start_time = time.time()

# Flags for plotting
plt_save = True
plt_show = False

# Constants
MASS = 1  # [kg]
I = np.diag([1, 1, 1])  # Moment of inertia matrix
F_THRUSTER = 1  # Maximum Thrust [N]
DX, DY = 1, 1  # Distance from the CoM to the Thrusters [m]
U_MIN, U_MAX = 0, 1  # Thrust bounds
L = 1  # Length of the Robot
R_SPACECRAFT = 0.5 * (L * np.sqrt(2))  # Approximate shape of the spacecraft
OBSTACLE_PARAMS = {'position': [5, 5, 0], 'radius': 0.3}

# MPC Parameters
DT_MPC = 1
T_HORIZON = 12
C_HORIZON = 3
Q = np.eye(13) * 1e1
Q[3:5, 3:5] = 1e3
Q[10:13, 10:13] = 1e0
R = np.eye(8) * 1e-1
P = np.eye(13) * 1e1
rho = 1e4  # Obstacle Margin Slack Variable Weight
MPC_FREQ = 1

# Simulation parameters
SIMULATION_TIME = 300  # Total simulation time in seconds
DT_SIM = 0.1  # Time step
NUM_STEPS = int(SIMULATION_TIME / DT_SIM)  # Number of simulation steps
x0 = np.zeros(13)
x0[0:2] = [0, 10]
x0[6:10] = euler_to_quaternion(0, 0, 0)
x0[10:13] = [0.0, 1e-24, 0]

# Storage for states and inputs
states = np.zeros((NUM_STEPS + 1, 13))
inputs = np.zeros((NUM_STEPS, 8))
states_euler = np.zeros((NUM_STEPS + 1, 3))
states[0, :] = x0

# Cost evolution and slack variable evolution
cost_evolution = []
xi_evolution = []
eta_evolution = []

# Reference scenario parameters
static_reference = True  # Static or dynamic reference

def target_dynamics(t):
    """Define target dynamics based on the scenario type."""
    x_ref_static = np.zeros(13)
    x_ref_static[0:2] = [10, 0]
    x_ref_static[6:10] = euler_to_quaternion(0, 0, 0)
    x_ref_static[10:13] = [0, 1e-24, 0]
    
    if static_reference:
        return x_ref_static
    else:
        x_ref_dyn = np.zeros(13)
        if t <= NUM_STEPS / 2:
            x_ref_dyn[0] += 0.1 * t
            x_ref_dyn[1] += 0.2 * t
        else:
            x_ref_dyn[0] += 10
            x_ref_dyn[1] += 10
        x_ref_dyn[6:10] = euler_to_quaternion(0, 0, 0)
        return x_ref_dyn

def initialize_mpc_controller():
    """Initialize the MPC controller with given parameters."""
    return MPCController(T_HORIZON, C_HORIZON, MASS, I, DX, DY, DT_MPC, Q, R, P, U_MIN, U_MAX, 
                         OBSTACLE_PARAMS['position'], OBSTACLE_PARAMS['radius'], rho, R_SPACECRAFT)

def save_simulation_parameters(filename):
    """Save simulation parameters to a text file."""
    params = {
        "mass": MASS,
        "Ixx": 1, "Iyy": 1, "Izz": 1,
        "Thrust bounds": [U_MIN, U_MAX],
        "MPC Horizon": T_HORIZON,
        "Control Horizon": C_HORIZON,
        "State Weighting Matrix Q": Q.tolist(),
        "Control Weighting Matrix R": R.tolist(),
        "Terminal Cost Weighting Matrix P": P.tolist(),
        "rho (Obstacle Marging Slack)": rho,
        "dt_MPC": DT_MPC,
        "simulation_time": SIMULATION_TIME,
        "dt_sim": DT_SIM,
        "Initial state x0": x0.tolist(),
        "Obstacle position": OBSTACLE_PARAMS['position'],
        "Obstacle radius": OBSTACLE_PARAMS['radius'],
        "Static reference": static_reference,
        "Spacecraft radius": R_SPACECRAFT
    }

    with open(filename, 'w') as file:
        for key, value in params.items():
            file.write(f"{key}: {value}\n")

def output_directory_creation():
    """Create an output directory for saving simulation results."""
    scenario_name = os.path.splitext(os.path.basename(__file__))[0]
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join('outputs', f"{scenario_name}_{current_time}")
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

def run_simulation():
    """Run the MPC simulation."""
    controller = initialize_mpc_controller()
    u_guess = np.zeros((C_HORIZON * 8, 1))

    for t in range(NUM_STEPS):
        x_ref = target_dynamics(t)
        if t % int(1 / (MPC_FREQ * DT_SIM)) == 0:
            u, xi_optimal, eta_optimal, cost_iter = controller.get_optimal_input(states[t, :], x_ref, u_guess)
            predicted_inputs[t, :, :] = u
            
            # Predict future trajectory
            X_pred = states[t, :].copy()
            for k in range(C_HORIZON):
                X_pred = rk4_step(X_pred, predicted_inputs[t, k, :], DT_SIM)
            cost_evolution.append(cost_iter[-1])
            xi_evolution.append(xi_optimal[-1])
            eta_evolution.append(eta_optimal[-1])

        # Apply control input
        states[t + 1, :] = rk4_step(states[t, :], u[0, :], DT_SIM)
        states_euler[t + 1, :] = quaternion_to_euler(states[t + 1, 6:10])

def plot_results(output_folder):
    """Generate plots for the simulation results."""
    time = np.linspace(0, SIMULATION_TIME, NUM_STEPS + 1)
    time_inputs = np.linspace(0, SIMULATION_TIME - DT_SIM, NUM_STEPS)

    # Plot states
    plt.figure(figsize=(12, 8))
    plt.plot(time, states[:, 0], label='r_x')
    plt.plot(time, states[:, 1], label='r_y')
    plt.xlabel('Time [s]')
    plt.ylabel('States')
    plt.legend()
    plt.grid()
    if plt_save:
        plt.savefig(os.path.join(output_folder, 'state_plot.png'))

    # Plot inputs
    plt.figure(figsize=(12, 8))
    for i in range(8):
        plt.subplot(5, 2, i + 1)
        plt.step(time_inputs, inputs[:, i], label=f'u{i + 1}')
        plt.xlabel('Time [s]')
        plt.ylabel('Inputs')
        plt.legend()
        plt.grid()
    plt.tight_layout()
    if plt_save:
        plt.savefig(os.path.join(output_folder, 'inputs_plot.png'))

    # Plot cost evolution
    plt.figure(figsize=(12, 8))
    plt.plot(np.linspace(0, SIMULATION_TIME - DT_SIM, len(cost_evolution)), cost_evolution)
    plt.xlabel('Time [s]')
    plt.ylabel('Cost')
    plt.title('Cost Evolution')
    plt.grid()
    if plt_save:
        plt.savefig(os.path.join(output_folder, 'cost_evolution_plot.png'))

    # Plot trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(states[:, 0], states[:, 1], label='Trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory')
    plt.grid()
    if plt_save:
        plt.savefig(os.path.join(output_folder, 'trajectory_plot.png'))

def main():
    output_folder = output_directory_creation()
    run_simulation()
    plot_results(output_folder)
    if plt_show:
        plt.show()

if __name__ == "__main__":
    main()
