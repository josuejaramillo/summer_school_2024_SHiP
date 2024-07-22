# LLPsim

## Overview

`LLPsim` is a Python script designed for simulating and analyzing particle physics data. It leverages numerical and visualization techniques to analyze distributions related to LLP (Light Long-Lived Particles) scenarios. The script performs tasks such as kinematic sampling, distribution analysis, and 3D visualization of particle data.

## Features

1. **Kinematic Sampling**: Generates kinematic data samples based on input distributions.
2. **Distribution Analysis**: Analyzes and plots angular and energy distributions of particles.
3. **3D Visualization**: Creates 3D plots of particle data and visualizes the geometry of a truncated pyramid, which represents the detector or experimental setup.

## Components

1. **`kinematics.py`**: Contains functions for sampling and interpolating kinematic data. This includes grid-based sampling and resampling techniques.

2. **`crosscheck.py`**: Provides the `DistributionAnalyzer` class for analyzing and plotting angular and energy distributions based on the sampled kinematic data.

3. **`LLPsim.py`**: The main script that orchestrates the entire simulation process. It initializes the LLP scenario, performs kinematic sampling, analyzes the distributions, and generates visualizations.

4. **`plot_truncated_pyramid.py`**: Contains code for visualizing a truncated pyramid and scatter plotting particle data in a 3D space.

## How to Use

1. **Set Up Environment**:
   - Ensure you have Python 3.x installed.
   - Install required packages using:
     ```bash
     pip install numpy pandas matplotlib scipy
     ```

2. **Prepare Input Data**:
   - Input files should be prepared and located in the appropriate directories as specified in the script.

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

4. **View Results**:
   - Distribution plots will be saved in the specified directory.
   - 3D visualization will also be saved and can be viewed using image viewers.

## Example Usage

Here is an example snippet of how the `LLPsim.py` script might be used:

```python
from src import init
from src import kinematics
from src import crosscheck

# Initialize LLP scenario
LLP = init.LLP()

# Generate kinematic samples
nPoints = 1000000
mass = LLP.mass
c_tau = LLP.c_tau
resampleSize = 10**5
timing = "True"
kinematics_samples = kinematics.grids(LLP.Distr, LLP.Energy_distr, nPoints, mass, c_tau)
kinematics_samples.interpolate(timing)
kinematics_samples.resample(resampleSize, timing)
kinematics_samples.true_samples(timing)
kinematics_samples.save_kinematics(LLP.particle_path)

# Analyze distributions
samples_analysis = crosscheck.DistributionAnalyzer(
    LLP.Distr, LLP.Energy_distr, LLP.mass, 
    kinematics_samples.get_energy(), 
    kinematics_samples.get_theta(), 
    LLP.LLP_name, LLP.particle_path
)
samples_analysis.crosscheck("angle")
samples_analysis.crosscheck("energy")
