# DDOinVRP

#Introduction
This directory is dedicated to implementing the second project of my PhD research on DDO applications in VRP. Specifically, we aim to make use of ML and CG to improve optimization of VRP variants, namely CVRP and/or VRPTW.

#Scripts
To that end, we have the following modules
* `instance_generator.py`: generating random CVRPTW instance for given customer size.
* `column_generation.py`: implementing column generation procedure for the relaxed VRP instance.
* `branch_and_price.py`: implementing a branch and price algorithm for the VRP instnce.

#Literature