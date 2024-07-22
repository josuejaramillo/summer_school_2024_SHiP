import numba as nb
import numpy as np
import time
import pandas as pd

@nb.njit('(float64[::1], float64)', inline='always')
def _searchsorted_opt(arr, val):
    i = 0
    while i < arr.size and val > arr[i]:
        i += 1
    return i

@nb.njit('(float64[:,::1], float64[::1], float64[::1], float64[:,::1])', parallel=True)
def _bilinear_interpolation(rand_points, grid_x, grid_y, distr):
    results = np.empty(len(rand_points))
    len_y = grid_y.shape[0]
    
    for i in nb.prange(len(rand_points)):
        x, y = rand_points[i]

        # Find the indices of the grid points surrounding the point
        idx_x1 = _searchsorted_opt(grid_x, x) - 1
        idx_x2 = idx_x1 + 1
        idx_y1 = _searchsorted_opt(grid_y, y) - 1
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
def _trilinear_interpolation(rand_points, grid_x, grid_y, grid_z, distr, max_energy):
    results = np.empty(len(rand_points))
    len_y, len_z = grid_y.shape[0], grid_z.shape[0]

    for i in nb.prange(len(rand_points)):
        x, y, z = rand_points[i]

        #Conditional to consider angle dependance on energy
        if z > max_energy[i]:
            continue

        idx_x1 = _searchsorted_opt(grid_x, x) - 1
        idx_x2 = idx_x1 + 1
        idx_y1 = _searchsorted_opt(grid_y, y) - 1
        idx_y2 = idx_y1 + 1
        idx_z1 = _searchsorted_opt(grid_z, z) - 1
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

#Decay volume geometry
def x_max(z):
    # return 0.5 + z*np.tan(1.5/50)
    return (0.02*(82 - z) + (2/25)*(-32 + z))/2

def y_max(z):
    # return 1.35 + z*np.tan(1.75/50)
    return (0.054*(82 - z) + 0.124*(-32 + z))/2

class grids:

    def __init__(self, Distr, Energy_distr, nPoints, mass, c_tau):
        
        self.Distr = Distr
        self.Energy_distr = Energy_distr
        self.nPoints = nPoints
        self.m = mass
        self.c_tau = c_tau

        #Random angles
        self.thetamin = self. Distr[1].min()
        self.thetamax = 0.043

        # Grids

        # Prepare the grid and distribution values for max energy
        self.grid_m = np.unique(self.Energy_distr.iloc[:, 0])
        self.grid_a = np.unique(self.Energy_distr.iloc[:, 1])

        #Filling 2D grid with values of the original distribution `Energy_distr`
        self.energy_distr = self._fill_distr_2D(self.grid_m, self.grid_a)

        # Prepare the grid and distribution values for distribution
        self.grid_x = np.unique(self.Distr.iloc[:, 0])
        self.grid_y = np.unique(self.Distr.iloc[:, 1])
        self.grid_z = np.unique(self.Distr.iloc[:, 2])

        #Filling 3D grid with values of the original distribution `Distr`
        self.distr = self._fill_distr_3D(self.grid_x, self.grid_y, self.grid_z)
        
    def _fill_distr_3D(self, grid_x, grid_y, grid_z):
        # Fill the distribution array with corresponding 'f' values
        distr_grid = np.zeros((len(grid_x), len(grid_y), len(grid_z)))
        mass__, angle__, energy__, value__ = np.asarray(self.Distr[0]), np.asarray(self.Distr[1]), np.asarray(self.Distr[2]), np.asarray(self.Distr[3])
        Distr_size = len(self.Distr)
        @nb.njit('(float64[:,:,::1],)', parallel=True)
        def filling(grid):
            for i in nb.prange(Distr_size):
                ix = _searchsorted_opt(grid_x, mass__[i])
                iy = _searchsorted_opt(grid_y, angle__[i])
                iz = _searchsorted_opt(grid_z, energy__[i])
                grid[ix, iy, iz] = value__[i]
            return grid
        return filling(distr_grid)
        
    def _fill_distr_2D(self, grid_m, grid_a):
        # Fill the distribution array with corresponding 'f' values
        distr_grid = np.zeros((len(self.grid_m), len(self.grid_a)))
        mass__, angle__, energy__ = np.asarray(self.Energy_distr[0]), np.asarray(self.Energy_distr[1]), np.asarray(self.Energy_distr[2])
        Distr_size = len(self.Energy_distr)
        @nb.njit('(float64[:,::1],)', parallel=True)
        def filling(grid):
            for i in nb.prange(Distr_size):
                ix = _searchsorted_opt(grid_m, mass__[i])
                iy = _searchsorted_opt(grid_a, angle__[i])
                grid[ix, iy] = energy__[i]
            return grid
        return filling(distr_grid)
    
    def interpolate(self, timing="False"):
        
        if timing:
            t = time.time()

        #Sample angle points
        self.theta = np.random.uniform(self.thetamin, self.thetamax, self.nPoints)
        self.mass = self.m*np.ones(self.nPoints)
        #Maximun energy
        #Interpolated values for max energy
        point_bilinear_interpolation = np.column_stack((self.mass, self.theta))
        self.max_energy = _bilinear_interpolation(point_bilinear_interpolation, self.grid_m, self.grid_a, self.energy_distr)

        # Sample energy and define interpolation points
        energy = np.ones_like(self.theta)
        points_to_interpolate = np.column_stack((self.mass, self.theta, energy))
        self.energy = np.random.uniform(self.m, self.max_energy)

        points_to_interpolate[:, 2] = energy

        #Interpolated values for particle distribution
        self.interpolated_values = _trilinear_interpolation(points_to_interpolate, self.grid_x, self.grid_y, self.grid_z, self.distr, self.max_energy)
        
        if timing:
            print(f"Interpolation time t = {time.time() - t} s")
            
        # return [theta, energy, max_energy, interpolated_values]
    
    def resample(self, rsample_size, timing="False"):

        if timing:
            t = time.time()

        self.rsample_size = rsample_size
        weights = self.interpolated_values * (self.max_energy - self.mass)
        self.true_points_indices = np.random.choice(self.nPoints, size=self.rsample_size, p=weights/weights.sum())

        #Resampled angle and energies
        self.r_theta = self.theta[self.true_points_indices]
        self.r_energy = self.energy[self.true_points_indices]

        if timing:
            print(f"Resample angle and energy  t = {time.time() - t} s")

    def true_samples(self, timing = "False"):

        if timing:
            t = time.time()

        #Angle phi sampling
        self.phi = np.random.uniform(-np.pi,np.pi, len(self.true_points_indices))

        #Momentum calculation
        momentum = np.sqrt(self.r_energy**2 - (self.m*np.ones_like(self.true_points_indices))**2)
        px = momentum*np.cos(self.phi)*np.sin(self.r_theta)
        py = momentum*np.sin(self.phi)*np.sin(self.r_theta)
        pz = momentum*np.cos(self.r_theta)

        #Z values
        cmin = 1 - np.exp(-32*self.m / (np.cos(self.r_theta) * self.c_tau * momentum))
        cmax = 1 - np.exp(-82*self.m / (np.cos(self.r_theta) * self.c_tau * momentum))
        c = np.random.uniform(cmin, cmax, size = self.rsample_size)
        c[c==1] = 0.9999999999999999

        z = np.cos(self.r_theta)*self.c_tau*(momentum/self.m)*np.log(1/(1-c))

        #X and Y values
        x = z*np.cos(self.phi)*np.tan(self.r_theta)
        y = z*np.sin(self.phi)*np.tan(self.r_theta)

        #Decay probability
        P_decay = np.exp(-32*self.m / (np.cos(self.r_theta) * self.c_tau * momentum)) - np.exp(-82*self.m / (np.cos(self.r_theta) * self.c_tau * momentum))


        #Mask for particles decaying inside the volume
        mask = (-x_max(z) < x) & (x < x_max(z)) & (-y_max(z) < y) & (y < y_max(z)) & (32 < z) & (z < 82)

        self.kinematics_dic = {
            "theta": self.r_theta[mask],
            "energy": self.r_energy[mask],
            "px": px[mask],
            "py": py[mask],
            "pz": pz[mask],
            "P": momentum[mask],
            "x": x[mask],
            "y": y[mask],
            "z": z[mask],
            "r": np.sqrt(x[mask]**2 + y[mask]**2 + z[mask]**2),
            "P_decay" : P_decay[mask]
        }

        if timing:
            print(f"Samplig vertices t = {time.time() - t} s")

    def save_kinematics(self, path):
        kinetics_df = pd.DataFrame(self.kinematics_dic)
        kinetics_df.to_csv(path+"/"+"kinetic_sampling.dat", sep = "\t", index=False)

    

