# Import necessary functions and modules
from funcs import initLLP
from funcs import decayProducts
import time

# Initialize the LLP (Lightest Long-Lived Particle) object
# This object will likely contain properties such as mass, PDGs (Particle Data Group identifiers), branching ratios, etc.
LLP = initLLP.LLP()

# Number of events to simulate
nEvents = 100000

# Measure the time taken to run the decay simulation for the first time
t = time.time()  # Start time measurement
# Simulate decays in the rest frame for the LLP using the parameters from the LLP object
# `simulateDecays_rest_frame` function generates decay products based on the LLP properties and specified number of events
decayProducts.simulateDecays_rest_frame(LLP.mass, LLP.PDGs, LLP.BrRatios_distr, nEvents, LLP.Matrix_elements)
print("total time  ", time.time()-t)  # Print the elapsed time

# Measure the time taken to run the decay simulation for the second time
t = time.time()  # Start time measurement
# Run the simulation again to see if performance improves due to caching or other factors
unBoostedProducts = decayProducts.simulateDecays_rest_frame(LLP.mass, LLP.PDGs, LLP.BrRatios_distr, nEvents, LLP.Matrix_elements)
print("total time second time ", time.time()-t)  # Print the elapsed time for the second run
