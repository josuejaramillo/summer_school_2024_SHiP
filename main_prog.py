import numba as nb
import matplotlib.pyplot as plt
from scipy.integrate import simps, nquad
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import numpy as np
import time
import os
import mplhep as hep
hep.style.use("CMS")

@nb.njit('(float64[::1], float64)', inline='always')
def searchsorted_opt(arr, val):
    i = 0
    while i < arr.size and val > arr[i]:
        i += 1
    return i

@nb.njit('(float64[:,::1], float64[::1], float64[::1], float64[:,::1])', parallel=True)
def bilinear_interpolation(rand_points, grid_x, grid_y, distr):
    results = np.empty(len(rand_points))
    len_y = grid_y.shape[0]
    
    for i in nb.prange(len(rand_points)):
        x, y = rand_points[i]

        # Find the indices of the grid points surrounding the point
        idx_x1 = searchsorted_opt(grid_x, x) - 1
        idx_x2 = idx_x1 + 1
        idx_y1 = searchsorted_opt(grid_y, y) - 1
        idx_y2 = idx_y1 + 1

        # Ensure the indices are within the bounds of the grid
        idx_x1 = max(0, min(idx_x1, len(grid_x) - 2))
        idx_x2 = max(1, min(idx_x2, len(grid_x) - 1))
        idx_y1 = max(0, min(idx_y1, len_y - 2))
        idx_y2 = max(1, min(idx_y2, len_y - 1))

        # Get the coordinates of the grid points
        x1, x2 = grid_x[idx_x1], grid_x[idx_x2]
        y1, y2 = grid_y[idx_y1], grid_y[idx_y2]

        # Get the values at the corners of the cell
        z11 = distr[idx_x1, idx_y1]
        z21 = distr[idx_x2, idx_y1]
        z12 = distr[idx_x1, idx_y2]
        z22 = distr[idx_x2, idx_y2]

        # Calculate the interpolation weights
        xd = (x - x1) / (x2 - x1)
        yd = (y - y1) / (y2 - y1)

        # Perform the interpolation
        c0 = z11 * (1 - xd) + z21 * xd
        c1 = z12 * (1 - xd) + z22 * xd

        result = c0 * (1 - yd) + c1 * yd

        results[i] = result

    return results


@nb.njit('(float64[:,::1], float64[::1], float64[::1], float64[::1], float64[:,:,::1], float64[::1])', parallel=True)
def trilinear_interpolation(rand_points, grid_x, grid_y, grid_z, distr, max_energy):
    results = np.empty(len(rand_points))
    len_y, len_z = grid_y.shape[0], grid_z.shape[0]

    for i in nb.prange(len(rand_points)):
        x, y, z = rand_points[i]

        #Conditional to consider angle dependance on energy
        if z > max_energy[i]:
            continue

        idx_x1 = searchsorted_opt(grid_x, x) - 1
        idx_x2 = idx_x1 + 1
        idx_y1 = searchsorted_opt(grid_y, y) - 1
        idx_y2 = idx_y1 + 1
        idx_z1 = searchsorted_opt(grid_z, z) - 1
        idx_z2 = idx_z1 + 1

        idx_x1 = max(0, min(idx_x1, len(grid_x) - 2))
        idx_x2 = max(1, min(idx_x2, len(grid_x) - 1))
        idx_y1 = max(0, min(idx_y1, len_y - 2))
        idx_y2 = max(1, min(idx_y2, len_y - 1))
        idx_z1 = max(0, min(idx_z1, len_z - 2))
        idx_z2 = max(1, min(idx_z2, len_z - 1))

        x1, x2 = grid_x[idx_x1], grid_x[idx_x2]
        y1, y2 = grid_y[idx_y1], grid_y[idx_y2]
        z1, z2 = grid_z[idx_z1], grid_z[idx_z2]

        z111 = distr[idx_x1, idx_y1, idx_z1]
        z211 = distr[idx_x2, idx_y1, idx_z1]
        z121 = distr[idx_x1, idx_y2, idx_z1]
        z221 = distr[idx_x2, idx_y2, idx_z1]
        z112 = distr[idx_x1, idx_y1, idx_z2]
        z212 = distr[idx_x2, idx_y1, idx_z2]
        z122 = distr[idx_x1, idx_y2, idx_z2]
        z222 = distr[idx_x2, idx_y2, idx_z2]

        xd = (x - x1) / (x2 - x1)
        yd = (y - y1) / (y2 - y1)
        zd = (z - z1) / (z2 - z1)

        c00 = z111 * (1 - xd) + z211 * xd
        c01 = z112 * (1 - xd) + z212 * xd
        c10 = z121 * (1 - xd) + z221 * xd
        c11 = z122 * (1 - xd) + z222 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        result = c0 * (1 - zd) + c1 * zd

        results[i] = result

    return results

def fill_distr_3D(distr_grid, Distr):
    # Fill the distribution array with corresponding 'f' values
    mass__, angle__, energy__, value__ = np.asarray(Distr[0]), np.asarray(Distr[1]), np.asarray(Distr[2]), np.asarray(Distr[3])
    Distr_size = len(Distr)
    @nb.njit('(float64[:,:,::1],)', parallel=True)
    def filling(distr_grid):
        for i in nb.prange(Distr_size):
            ix = searchsorted_opt(grid_x, mass__[i])
            iy = searchsorted_opt(grid_y, angle__[i])
            iz = searchsorted_opt(grid_z, energy__[i])
            distr_grid[ix, iy, iz] = value__[i]
        return distr_grid
    return filling(distr_grid)

#*********************************************************************

#Particle selection
main_folder = "./Distributions"
folders = np.array(os.listdir(main_folder))

print("\n Particle selector \n")
for i in range(len(folders)):
    print(str(i+1) + ". " + folders[i])
selected_particle = int(input("Select particle: ")) - 1
LLP = folders[selected_particle].replace("_", " ")

try:
    particle_distr_folder = folders[selected_particle]
except:
    raise ValueError("Error during particle selection")

# Read data with the correct delimiter
files = os.listdir(main_folder+"/"+particle_distr_folder)

distribution_file = [f for f in files if f.startswith('D')]
energy_file = [f for f in files if f.startswith('E')]

distribution_file_path = main_folder+"/"+particle_distr_folder+"/"+distribution_file[0]
energy_file_path = main_folder+"/"+particle_distr_folder+"/"+energy_file[0]

Distr = pd.read_csv(distribution_file_path, header=None, sep="\t")
Energy_distr = pd.read_csv(energy_file_path, header=None, sep="\t")


# Random dataset
nPoints = 1000000
mass = 0.212 * np.ones(nPoints)
emin = mass[0]
emax = Distr[2].max()
thetamin = Distr[1].min()
thetamax = Distr[1].max()
theta = np.random.uniform(thetamin, thetamax, nPoints)

#***************************Bilinearinterpolation*********************************

# Define set of points
point_bilinear_interpolation = np.column_stack((mass, theta))

# Prepare the grid and distribution values
grid_m = np.unique(Energy_distr.iloc[:, 0])
grid_a = np.unique(Energy_distr.iloc[:, 1])
energy_distr_grid = np.zeros((len(grid_m), len(grid_a)))

# Fill the distribution array with corresponding 'f' values
for i in range(len(Energy_distr)):
    ix = searchsorted_opt(grid_m, Energy_distr.iloc[i, 0])
    iy = searchsorted_opt(grid_a, Energy_distr.iloc[i, 1])
    energy_distr_grid[ix, iy] = Energy_distr.iloc[i, 2]

#Interpolated values for max energy
max_energy = bilinear_interpolation(point_bilinear_interpolation, grid_m, grid_a, energy_distr_grid)

#***************************Trilinearinterpolation*********************************

# Prepare the grid and distribution values
grid_x = np.unique(Distr.iloc[:, 0])
grid_y = np.unique(Distr.iloc[:, 1])
grid_z = np.unique(Distr.iloc[:, 2])
distr_grid = np.zeros((len(grid_x), len(grid_y), len(grid_z)))

# energy = np.random.uniform(emin, emax, nPoints)
energy = np.ones_like(theta)

#Filling 3D grid with values of the original distribution `Distr`
distr = fill_distr_3D(distr_grid, Distr)

# Define set of points
points_to_interpolate = np.column_stack((mass, theta, energy))
energy = np.random.uniform(emin, max_energy)
points_to_interpolate[:, 2] = energy

interpolated_values = trilinear_interpolation(points_to_interpolate, grid_x, grid_y, grid_z, distr, max_energy)

#***************************Cross-check*********************************

def crosscheck(var):
    # intTime = time.time()
    weights = interpolated_values * (max_energy - mass)
    true_points_indices = np.random.choice(nPoints, size=10**5, p=weights/weights.sum())

    # Define interpolation function using scipy.interpolate.RegularGridInterpolator
    interpolating_function = RegularGridInterpolator((grid_x, grid_y, grid_z), distr)

    fig = plt.figure(figsize=(10,8))

    if var == "angle":
        # Calculate histogram with normalized probability density
        counts, bin_edges = np.histogram(theta[true_points_indices], bins=25, density=False)
        bin_width = np.diff(bin_edges)
        probability_density = counts / (counts.sum() * bin_width)

        y_values = np.arange(thetamin, thetamax, 0.0003)

        #Find emax for each angle in the array to integrate
        point_bilinear_interpolation = np.column_stack((emin*np.ones_like(y_values), y_values))
        emax_integration = bilinear_interpolation(point_bilinear_interpolation, grid_m, grid_a, energy_distr_grid)
        
        # Function for integration
        def integrand(y):
            def inner_integrand(z):
                return interpolating_function((0.212, y, z))
            emax = emax_integration[searchsorted_opt(y_values, y)] #Finds max energy for the given angle y
            return nquad(inner_integrand, [[emin, emax]], opts={"epsabs": 1e-2, "epsrel": 1e-2})[0]
        
        # Calculate the angular distribution
        angular_distribution = np.asarray([integrand(y) for y in y_values])
        # Normalize the angular distribution by integrating
        total_integral = simps(angular_distribution, y_values)
        distribution_normalized = angular_distribution/total_integral

        plt.plot(y_values, distribution_normalized, label="Normalized " + var +" distr.", color = "black")
        plt.xlabel("θ [Rad]")
        plt.ylabel("$f_{θ}$")
        plt.title(LLP + " angular distribution")

    if var == "energy":
        # Calculate histogram with normalized probability density
        counts, bin_edges = np.histogram(energy[true_points_indices], bins=25, density=False)
        bin_width = np.diff(bin_edges)
        probability_density = counts / (counts.sum() * bin_width)

        # Function for integration
        def integrand(z):
            def inner_integrand(y):
                e_max = bilinear_interpolation(np.asarray([[emin, y]]), grid_m, grid_a, energy_distr_grid)
                if z > e_max:
                    return 0
                else:
                    return interpolating_function((0.212, y, z))
            return nquad(inner_integrand, [[thetamin, thetamax]], opts={"epsabs": 1e-2, "epsrel": 1e-2})[0]

        # Calculate the energy distribution
        z_values = np.arange(emin, emax, 3)
        energy_distribution = np.asarray([integrand(z) for z in z_values])

        # Normalize the angular distribution by integrating
        total_integral = simps(energy_distribution, z_values)
        distribution_normalized = energy_distribution/total_integral

        plt.plot(z_values, distribution_normalized, label="Normalized " + var +" distr.", color = "black")
        plt.xlabel("E [GeV]")
        plt.ylabel("$f_{E}$")
        plt.title(LLP + " energy distribution")

    # Plot the probability density histogram
    plt.bar(bin_edges[:-1], probability_density, width=bin_width, alpha=0.7, label= var.capitalize() +" points Probability Density")

    # plt.xlim(0.0005, 0.01)
    plt.legend()
    fig.savefig(main_folder+"/"+particle_distr_folder+"/"+var+".png")
    
    # intExcTime = time.time() - intTime
    # print(f"Time crosscheck {intExcTime} s")
    # plt.show()

crosscheck("angle")
crosscheck("energy")


