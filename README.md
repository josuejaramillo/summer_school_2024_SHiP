# Decay Simulation Code

## Overview

This repository contains code for simulating particle decays of a Lightest Long-Lived Particle (LLP). The simulation includes the generation of decay products in the rest frame of the LLP, calculation of kinematic properties, and optimization for performance. The code utilizes Numba for Just-In-Time (JIT) compilation to enhance speed and efficiency.

## Installation

Ensure you have the required packages installed. You can use the following command to install the dependencies:

```bash
pip install numpy sympy numba scipy
```

## File Structure

- `funcs/`:
  - `initLLP.py`: Contains the `LLP` class which initializes the LLP object with attributes like mass, PDGs (Particle Data Group identifiers), and branching ratios.
  - `decayProducts.py`: Contains functions for simulating decays and computing decay products.
  - `HNLmerging.py`: Contains functions for handling HNL merging processes.
  - `PDG.py`: Contains functions or data related to Particle Data Group identifiers.
  - `rotateVectors.py`: Contains functions for rotating vectors.
  - `ThreeBodyDecay.py`: Contains functions for simulating three-body decays.
  - `TwoBodyDecay.py`: Contains functions for simulating two-body decays.
  - `boost.py`: Contains functions for boosting decay products to the lab frame.
  - `kinematics.py`: Contains functions for handling kinematic distributions and interpolations.

- Main code:
  - `simulate.py`: Main script to run the decay simulation.

## Usage

### Initialization

Initialize LLP:

The `initLLP.LLP()` function initializes an LLP object with properties needed for the simulation.

### Simulation

Simulate Decays:

The `decayProducts.simulateDecays_rest_frame` function performs the decay simulation. It uses parameters from the LLP object and generates decay products.

### Example

Here's an example of how to run the simulation, including initialization, interpolation, resampling, and timing:

```python
# Import necessary functions and modules
from funcs import initLLP, decayProducts, boost, kinematics
import time
import numpy as np

# Initialize the LLP (Lightest Long-Lived Particle) object
# This object encapsulates properties like mass, PDG (Particle Data Group) identifiers, branching ratios, etc.
LLP = initLLP.LLP()

# Define the number of events to simulate
nEvents = 1000000
resampleSize = 100000
timing = False  # Set to True if you want to measure execution time for various steps


# Simulate decays in the rest frame of the LLP
# This generates decay products based on LLP properties and the specified number of events
decayProducts.simulateDecays_rest_frame(LLP.mass, LLP.PDGs, LLP.BrRatios_distr, nEvents, LLP.Matrix_elements)



# Measure the time taken for the second decay simulation run
t = time.time()  # Start the timer

# Initialize the `Grids` class for kinematic distributions
# This setup includes distributions and parameters for the simulation
kinematics_samples = kinematics.Grids(LLP.Distr, LLP.Energy_distr, nEvents, LLP.mass, LLP.c_tau)

# Interpolate the kinematic distributions and energy data
# This step generates interpolated values based on the provided distributions
kinematics_samples.interpolate(timing)

# Resample the data based on the interpolated distributions
# This step selects a subset of the interpolated points for further analysis
kinematics_samples.resample(resampleSize, timing)

# Compute true kinematic samples for the simulation
# This step calculates kinematic properties and decay probabilities for the samples
kinematics_samples.true_samples(timing)

# Retrieve the momentum data from the kinematics_samples object
momentum = kinematics_samples.get_momentum()

# Determine the number of final decay products
finalEvents = len(momentum)

# Simulate decays again to observe performance improvements (if any) due to caching or other factors
unBoostedProducts = decayProducts.simulateDecays_rest_frame(LLP.mass, LLP.PDGs, LLP.BrRatios_distr, finalEvents, LLP.Matrix_elements)

# Apply boosts to the decay products based on the momentum data
boostedProducts = boost.tab_boosted_decay_products(LLP.mass, momentum, unBoostedProducts)

# Print the elapsed time for the second run
print("total time second time ", time.time()-t)

#Save results

# Save the kinematic sample data to an output file
kinematics_samples.save_kinematics("./outputs", LLP.LLP_name)

# Save the boosted decay products to an output file
np.savetxt('./outputs/' + LLP.LLP_name + '_decayProducts.dat', boostedProducts)


```

### Explanation

- **Initialization**: Create an LLP object using `initLLP.LLP()`, which sets up the LLP properties such as mass, PDGs, and branching ratios.
- **Interpolation and Resampling**: Use `kinematics.Grids` to handle distributions, interpolate values, resample the data, and compute kinematic properties.
- **Simulation**: Use `simulateDecays_rest_frame` to simulate decay processes based on the LLP properties.
- **Boosting**: Convert decay products to the lab frame using `boost.tab_boosted_decay_products`.
- **Timing**: Measure and print execution times to assess performance.

## Functions

- **initLLP.LLP()**  
  Initializes an LLP object with parameters including mass, PDGs, branching ratios, and matrix elements.

- **decayProducts.simulateDecays_rest_frame**  
  Simulates decay processes for a given LLP. Takes parameters such as mass, PDGs, branching ratios, number of events, and matrix elements to compute decay products in the rest frame.

- **boost.tab_boosted_decay_products**  
  Transforms decay products from the rest frame to the lab frame.

- **kinematics.Grids**  
  Handles distribution data, performs interpolation, resampling, and calculates kinematic properties.

## Performance Optimization

The code uses Numba's JIT compilation to improve performance, particularly for the numerical functions involved in the simulations. Running the simulation multiple times may show improved performance due to optimizations and caching effects.



## Results

**Kinematic Sampling Results**

The results for the kinematic sampling are saved in a .dat file called LLP_name_kinematic_sampling.dat. Each column in the file represents the following kinematic quantities:

- **theta**: The polar angle (θ) of the particle.
- **energy**: The energy of the particle.
- **px**: The x-component of the particle's momentum.
- **py**: The y-component of the particle's momentum.
- **pz**: The z-component of the particle's momentum.
- **P**: The magnitude of the particle's momentum.
- **x**: The x-coordinate of the particle's position.
- **y**: The y-coordinate of the particle's position.
- **z**: The z-coordinate of the particle's position.
- **r**: The radial distance from the origin to the particle's position, calculated as √(x² + y² + z²).
- **P_decay**: The decay momentum of the particle.

**Decay Products Results**

The decay products are saved in a .dat file with the name format:

LLP_name\_str(mass)\_MixingPatternArray[0]\_MixingPatternArray[1]\_MixingPatternArray[2]\_c_tau\_decayProducts.dat

The file is separated by process and sample points, indicated as:

#<process={channel}; sample_points={channel_size}

Columns
- **px1, py1, pz1, e1, MASS1, pdg1, charge1, stability1**: Properties of the first decay product.
- **px2, py2, pz2, e2, MASS2, pdg2, charge2, stability2**: Properties of the second decay product.
- **px3, py3, pz3, e3, MASS3, pdg3, charge3, stability3**: Properties of the third decay product.

The 