# MPC Rendezvous and Contact Simulator

**An entry-level MPC-based microgravity simulator**

This repository provides the simulation and control framework developed during the Master's thesis *"Model Predictive Control-based Rendezvous and Contact of Autonomous Space Vehicles"*, by Rodrigo Silva. It models various autonomous space vehicle scenarios in a planar or 3D microgravity environment, using Model Predictive Control (MPC) with CasADi.

Developed as part of a joint research effort between KTH Royal Institute of Technology (Sweden) and Instituto Superior Técnico, University of Lisbon (Portugal).

---

##  Features

- 2D and 3D simulation of spacecraft rendezvous and docking
- Entry angle constraints and obstacle avoidance
- Configurable test scenarios
- MPC controller using CasADi and IPOPT
- Plotting and animation tools (optional)

---

##  Repository Structure

```
MPC_simulator/
├── run_simulator.py           # Main entry point to run scenarios
├── requirements.txt           # Required Python packages
├── scenarios/                 # All scenario definitions
├── src/                       # Source code
│   ├── controllers/           # MPC controller definitions
│   ├── dynamics/              # System dynamics (2D and 3D)
│   └── utils/                 # Quaternion tools and helpers
└── results/                   # Optional folder to store simulation outputs
```

---

## Installation

Make sure you have **Python 3.12.6** installed.

Install the required packages using:

```bash
pip install -r requirements.txt
```

---

##  Running a Scenario

The scenarios can be launched in two different ways: either directly inside the scenario script (example 1) or by launching `run_simulator.py` (example 2). The interface will soon support command-line arguments, but for now, modify the script directly to choose:

example 1:
```bash
python scenarios/obs_avoidance.py
```
example 2:
```bash
python run_simulator.py mpc_rhapsody
```

- The simulation dimensionality (`2D` or `3D`)
- The desired scenario name

Example scenario files:
- `entry_angle_v2`
- `obs_avoidance`
- `entry_plus_obs`
- `dock_alignement`
- `fugue`
- `multistage`
- `ros_scenario`

To view animations and plots, make sure to set the `show = True` flag inside the scenario script.
To save, the same logic goes for the `save = True` flag inside the scenario script.

---

## Reproducing Thesis Results

A mapping between each thesis figure and its corresponding scenario configuration will be provided in [`results/reproducibility.md`](results/reproducibility.md). This will allow others to replicate the experimental results, both from simulations and laboratory data.

---

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT), a permissive license that allows reuse with attribution. It is suitable for academic and research purposes. If you're unsure or have a specific use case, feel free to reach out.

---

## Author

**Rodrigo Silva**  
This simulator was developed as part of the Master's thesis:  
*Model Predictive Control-based Rendezvous and Contact of Autonomous Space Vehicles*  
KTH Royal Institute of Technology & Instituto Superior Técnico (University of Lisbon)


