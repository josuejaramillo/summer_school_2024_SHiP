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
  - `rotateVectors.py`: Contains functions for rotating vectors.
  - `main.py`: Script to run the decay simulation.

## Usage

### Initialization

Initialize LLP:

The `initLLP.LLP()` function initializes an LLP object with properties needed for the simulation.

### Simulation

Simulate Decays:

The `decayProducts.simulateDecays_rest_frame` function performs the decay simulation. It uses parameters from the LLP object and generates decay products.

### Example

Here's an example of how to run the simulation and measure the execution time:

```python
from funcs import initLLP
from funcs import decayProducts
import time

# Initialize LLP
LLP = initLLP.LLP()

# Number of events to simulate
nEvents = 100000

# Measure the time taken to run the decay simulation for the first time
t = time.time()
decayProducts.simulateDecays_rest_frame(LLP.mass, LLP.PDGs, LLP.BrRatios_distr, nEvents, LLP.Matrix_elements)
print("Total time for first run:", time.time() - t)

# Measure the time taken to run the decay simulation for the second time
t = time.time()
unBoostedProducts = decayProducts.simulateDecays_rest_frame(LLP.mass, LLP.PDGs, LLP.BrRatios_distr, nEvents, LLP.Matrix_elements)
print("Total time for second run:", time.time() - t)
```

### Explanation

- **Initialization**: Create an LLP object using `initLLP.LLP()`, which sets up the LLP properties such as mass, PDGs, and branching ratios.
- **Simulation**: Use `simulateDecays_rest_frame` to simulate the decay processes. This function generates the decay products and calculates their kinematic properties.
- **Timing**: Measure the execution time for running the simulation. This can help in assessing performance and optimization.

## Functions

- **initLLP.LLP()**  
  Initializes an LLP object with parameters including mass, PDGs, branching ratios, and matrix elements.

- **decayProducts.simulateDecays_rest_frame**  
  Simulates decay processes for a given LLP. Takes parameters such as mass, PDGs, branching ratios, number of events, and matrix elements to compute decay products in the rest frame.

## Performance Optimization

The code uses Numba's JIT compilation to improve performance, particularly for the numerical functions involved in the simulations. Running the simulation multiple times may show improved performance due to optimizations and caching effects.
```