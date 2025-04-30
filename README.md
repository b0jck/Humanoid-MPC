# Humanoid-MPC
Model Predictive Control Algorithms using three different models of increasing complexity:
- **Centroidal Model**
- **Kinodynamics Model**
- **Full-Dynamics Model**

**NOTICE:** This is nothing but a reimplementation on the work done by the authors of this [paper](https://inria.hal.science/hal-04647996v1/document). All software was originally published by the authors and only a few modifications were made in order to make the scripts work.
## Repo Structure:
For each dynamical model, 2 files are available, one for "straight walk" and one for "stairs walks". Run any of these and a PyBullet simulation will be launched. Some parameters (for instance step lenght and step apex height) can be modified by the user, as well as the MPC's predictive horizon lenght. Notice that any changes may lead to instability of the Robot, i.e. its fall.
All other files are better not to be modified and are necessary to run the PyBullet simulation.
## Warning
For running the simulation, a series of libraries and dependecies need to be installed/satisfied. For non-Linux users, this might require a large dose of patience. For more info about all the dipendencies, please refer to the authors' original repo.
