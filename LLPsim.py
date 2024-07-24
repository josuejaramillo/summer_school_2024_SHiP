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
# This step initializes the Decays object with the mass, momentum, LLP model name, decay channel, 
# and optionally times the computation of decay products
decays_products = decays.Decays(LLP.mass, momentum, LLP.LLP_name, "e+e-", True)

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

# samples_analysis.crosscheck("angle")

# Perform analysis and plotting of energy distribution
# This function calculates the normalized energy distribution and plots it.

# samples_analysis.crosscheck("energy")

# ........................Vertex graph........................

# vertex_graph.plot3D(LLP.particle_path)
