# DDOinVRP

#Introduction
This directory is dedicated to implementing the second project of my PhD research on DDO applications in VRP. Specifically, we aim to make use of ML and CG to improve optimization of VRP variants, namely CVRP and/or VRPTW.

#Scripts
To that end, we have the following modules:
- 
* `instance_generator.py`: generating random CVRPTW instance for given customer size.
* `column_generation.py`: implementing column generation procedure for the relaxed VRP instance.
* `branch_and_price.py`: implementing a branch and price algorithm for the VRP instnce.
* `DB_schernker_data_import.py`: imports DB Schenker data and accesses the google map API to retain relevant travel information
* `example_environments.py`: implements ESPRCTW environment and an RL agent which trains on the environment to solve it. The agent is a maskable PPO implemented with stable-baselines3 contrib.


#Literature