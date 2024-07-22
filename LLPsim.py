from src import init
from src import kinematics

#Select LLP, add mass and c*tau
LLP = init.LLP()

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



# theta, energy, max_energy, interpolated_distr = points_interpolation.interpolate()
# # Initial parameters
# rsample_size = 10**5
# mass = m * np.ones(nPoints)
# emin = m
# emax = Distr[2].max()
# thetamin = Distr[1].min()
# thetamax = 0.043 #Distr[1].max()
