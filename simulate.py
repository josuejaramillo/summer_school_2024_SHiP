# Import necessary functions and modules
from funcs import initLLP, decayProducts, boost, kinematics, mergeResults
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
compile = decayProducts.simulateDecays_rest_frame(LLP.mass, LLP.PDGs, LLP.BrRatios_distr, nEvents, LLP.Matrix_elements)

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

# Simulate decays again to observe performance improvements
unBoostedProducts, size_per_channel = decayProducts.simulateDecays_rest_frame(LLP.mass, LLP.PDGs, LLP.BrRatios_distr, finalEvents, LLP.Matrix_elements)

# Apply boosts to the decay products based on the momentum data
boostedProducts = boost.tab_boosted_decay_products(LLP.mass, momentum, unBoostedProducts)

# Print the elapsed time for the second run
print("total time second time ", time.time()-t)

#Save results

motherParticleResults = kinematics_samples.get_kinematics() #Get kinematics as an array
decayProductsResults = boostedProducts

# Save the kinematic sample data to an output file
# kinematics_samples.save_kinematics("./outputs", LLP.LLP_name)

# Save the boosted decay products to an output file
# boost.saveProducts(boostedProducts, LLP.LLP_name, LLP.mass, LLP.MixingPatternArray, LLP.c_tau, LLP.decayChannels, size_per_channel)

# Merge results
mergeResults.save(motherParticleResults, decayProductsResults, LLP.LLP_name, LLP.mass, LLP.MixingPatternArray, LLP.c_tau, LLP.decayChannels, size_per_channel)

