import numba as nb
import matplotlib.pyplot as plt
from scipy.integrate import simps, nquad
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import numpy as np
import time
import os
import glob


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
        # z11 = distr[idx_x1, idx_y1]
        # z21 = distr[idx_x2, idx_y1]
        # z12 = distr[idx_x1, idx_y2]
        # z22 = distr[idx_x2, idx_y2]

        z11 = distr[idx_x1, idx_y1] * (y - y1) / (y2 - y1)
        z21 = distr[idx_x2, idx_y1] * (y - y1) / (y2 - y1)
        z12 = distr[idx_x1, idx_y2] * (y2 - y) / (y2 - y1)
        z22 = distr[idx_x2, idx_y2] * (y2 - y) / (y2 - y1)

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

        if z > max_energy[i]:
            z = max_energy[i]
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


#Particle selection
main_folder = "./Distributions"
folders = np.array(os.listdir(main_folder))

print("\n Particle selector \n")
for i in range(len(folders)):
    print(str(i+1) + ". " + folders[i])
selected_particle = int(input("Select particle: ")) - 1

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
mass_, angle_, energy_value_ = Energy_distr[0], Energy_distr[1], Energy_distr[2]
for i in range(len(Energy_distr)):
    ix = searchsorted_opt(grid_m, mass_[i])
    iy = searchsorted_opt(grid_a, angle_[i])
    energy_distr_grid[ix, iy] = energy_value_[i]

#Interpolated values for max energy
max_energy = bilinear_interpolation(point_bilinear_interpolation, grid_m, grid_a, energy_distr_grid)

#***************************Trilinearinterpolation*********************************

# Prepare the grid and distribution values
grid_x = np.unique(Distr.iloc[:, 0])
grid_y = np.unique(Distr.iloc[:, 1])
grid_z = np.unique(Distr.iloc[:, 2])
distr = np.zeros((len(grid_x), len(grid_y), len(grid_z)))

energy = np.random.uniform(emin, emax, nPoints)

# Define set of points
points_to_interpolate = np.column_stack((mass, theta, energy))

t = time.time()
interpolated_values = trilinear_interpolation(points_to_interpolate, grid_x, grid_y, grid_z, distr, max_energy)
print(time.time()-t)




