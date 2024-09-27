# Script adapted from Elias' Python Weightless Simulator
# https://github.com/DISCOWER/Python_Weightless_Simulator/blob/main/run_simulator.py

import sys
import argparse
import importlib

def main(scenario_name):
    try:
        # Dynamically import the scenario module
        scenario_path = f"scenarios.{scenario_name}"
        scenario = importlib.import_module(scenario_path)
    except ModuleNotFoundError:
        print(f"Scenario '{scenario_name}' not found.")
        sys.exit(1)

    # Call the run function from the imported module, passing the simulator instance
    scenario.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_name", help="Name of the scenario to run")
    args = parser.parse_args()

    main(args.scenario_name)