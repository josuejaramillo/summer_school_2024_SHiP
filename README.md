# LLPsim

## Overview

`LLPsim` is a Python script designed for simulating and analyzing particle physics data. It leverages numerical and visualization techniques to analyze distributions related to LLP (Long-Lived Particles) scenarios. The script performs tasks such as kinematic sampling, distribution analysis, and 3D visualization of particle data.

## Features

1. **Kinematic Sampling**: Generates kinematic data samples based on input distributions.
2. **Distribution Analysis**: Analyzes and plots angular and energy distributions of particles.
3. **3D Visualization**: Creates 3D plots of particle data and visualizes the geometry of a truncated pyramid, which represents the detector or experimental setup.
4. **Decay Product Analysis**: Simulates and analyzes the decay products of long-lived particles, saving the results for further study.

## Components

1. **`LLPsim.py`**: The main script that orchestrates the entire simulation process. It initializes the LLP scenario, performs kinematic sampling, analyzes the distributions, and generates visualizations.

2. **`kinematics.py`**: Contains functions for sampling and interpolating kinematic data. This includes grid-based sampling and resampling techniques.

3. **`decays.py`**: Simulates the decay of long-lived particles into various final states, providing detailed information about the decay poducts.

4. **`crosscheck.py`**: Provides the `DistributionAnalyzer` class for analyzing and plotting angular and energy distributions based on the sampled kinematic data.

5. **`vertex_graph.py`**: Contains code for visualizing a truncated pyramid and scatter plotting particle data in a 3D space.



## How to Use

1. **Set Up Environment**:
   - Ensure you have Python 3.8.0 installed.
   - Install required packages using:
     ```bash
     pip install -r requirements.txt
     ```

2. **Prepare Input Data**:
   - Input files should be prepared and located in the appropriate directories as specified in the script (./Distributions/model).
      - **LLP Distribution File**: Contains the LLP particle distribution data.
      - **Maximum Energy Distribution File**: Provides the maximum energy distribution of the particles.
      - **Branching Ratios File**: Includes data on the branching ratios for the LLP decays.
      - **Decay Channels File**: Specifies the decay channels and associated parameters.

3. **Run the Main Script**:
   - Execute `LLPsim.py` to run the simulation and analysis:
     ```bash
     python LLPsim.py
     ```

   - The script will:
     - Initialize the LLP particle scenario.
     - Generate kinematic samples.
     - Analyze the angular and energy distributions.
     - Create and save plots for both distributions and 3D visualizations.

4. **Simulate Decay Products**:
   - Use `decays.py` to simulate decay products for long-lived particles:

   - This script will:
     - Initialize the decay simulation with given parameters.
     - Compute the decay products.
     - Save the decay product information to a CSV file.

5. **View Results**:
   - Distribution plots will be saved in the specified directory.
   - 3D visualization will also be saved and can be viewed using image viewers.
   - LLP decays vertices and decay product details will be saved in a CSV file within the specified directory.

## Example Usage

Here is an example snippet of how the `LLPsim.py` script might be used:

```python
from src import init
from src import kinematics
from src import crosscheck
from src import vertex_graph
from src import decays

# ........................Sampling........................


# Initialize LLP selection and set mass and lifetime (c*tau)
LLP = init.LLP()

# Input parameters
nPoints = 1000000  # Number of random points for interpolation
mass = LLP.mass      # Mass of the selected LLP
c_tau = LLP.c_tau    # Lifetime c*tau of the LLP
resampleSize = 100000  # Size of the resampled subset
timing = True        # Flag to enable execution time measurement

# Initialize the `grids` class with distributions and parameters
kinematics_samples = kinematics.Grids(LLP.Distr, LLP.Energy_distr, nPoints, mass, c_tau)

# Perform interpolation on the data
# This step calculates interpolated values for distribution and energy
kinematics_samples.interpolate(timing)

# Perform resampling based on the interpolated values
# This step selects a subset of points based on the interpolated distribution
kinematics_samples.resample(resampleSize, timing)

# Compute true kinematic samples
# This step calculates kinematic properties and decay probabilities for the samples
kinematics_samples.true_samples(timing)

# Save the kinematic properties to a file
# This step writes the calculated kinematic properties to a CSV file
kinematics_samples.save_kinematics(LLP.particle_path)

# Retrieve the momentum data from the kinematics_samples object
momentum = kinematics_samples.get_momentum()

# Create an instance of the Decays class with the specified parameters
# This step initializes the Decays object with the mass, momentum, LLP decay channels, Branching ratio distribution, 
# and optionally times the computation of decay products
decays_products = decays.Decays(LLP.mass, momentum, LLP.decay_channels, LLP.BrRatios_distr, True)

# Save the computed decay products to a file
# This step writes the decay product information to a CSV file in the specified directory
decays_products.save_decay_products(LLP.particle_path)

# ........................Crosscheck........................

samples_analysis = crosscheck.DistributionAnalyzer(
    LLP.Distr,  # Distribution data for 3D interpolation
    LLP.Energy_distr,  # Energy distribution data for 2D interpolation
    LLP.mass,  # Mass value used in analysis
    kinematics_samples.get_energy(),  # Array of energy values from kinematics_samples
    kinematics_samples.get_theta(),  # Array of theta (angle) values from kinematics_samples
    LLP.LLP_name,  # Name of the LLP particle (used in plot titles)
    LLP.particle_path  # Path where output files will be saved
)

# Perform analysis and plotting of angular distribution
# This function calculates the normalized angular distribution and plots it.

samples_analysis.crosscheck("angle")

# Perform analysis and plotting of energy distribution
# This function calculates the normalized energy distribution and plots it.

samples_analysis.crosscheck("energy")

# ........................Vertex graph........................

vertex_graph.plot3D(LLP.particle_path)

