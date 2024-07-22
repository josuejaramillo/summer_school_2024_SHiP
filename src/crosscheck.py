import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import nquad, simps
from src.functions import _searchsorted_opt, _bilinear_interpolation, _trilinear_interpolation, _fill_distr_3D, _fill_distr_2D

class DistributionAnalyzer:
    def __init__(self, Distr, Energy_distr, m, energy, theta, LLP, path):
        """
        Initialize the DistributionAnalyzer with required parameters.
        
        Parameters
        ----------
        Distr : pd.DataFrame
            DataFrame containing the distribution values for interpolation with columns representing the grid.
        Energy_distr : pd.DataFrame
            DataFrame containing the energy distribution values for interpolation.
        emin : float
            Minimum energy value.
        energy : np.ndarray
            Array of energy values for analysis.
        theta : np.ndarray
            Array of angle values for analysis.
        LLP : str
            Name of the LLP particle.
        path : str
            Directory path for saving the plots.
        """
        # Prepare the grid and distribution values for 3D interpolation
        self.grid_x = np.unique(Distr.iloc[:, 0])
        self.grid_y = np.unique(Distr.iloc[:, 1])
        self.grid_z = np.unique(Distr.iloc[:, 2])

        # Prepare the grid and distribution values for 2D energy interpolation
        self.grid_m = np.unique(Energy_distr.iloc[:, 0])
        self.grid_a = np.unique(Energy_distr.iloc[:, 1])

        self.distr = _fill_distr_3D(self.grid_x, self.grid_y, self.grid_z, Distr)
        self.energy_distr_grid = _fill_distr_2D(self.grid_m, self.grid_a, Energy_distr)

        self.emin = m
        self.emax = Distr.iloc[:, 2].max()
        self.thetamin = Distr.iloc[:, 1].min()
        self.thetamax = 0.043
        self.energy = energy
        self.theta = theta

        self.LLP = LLP
        self.path = path
        self.interpolating_function = RegularGridInterpolator((self.grid_x, self.grid_y, self.grid_z), self.distr)

    def crosscheck(self, var):
        """
        Analyze and plot the distribution of angles or energies.
        
        Parameters
        ----------
        var : str
            Either "angle" or "energy" to determine the type of distribution to analyze.
        """
        fig = plt.figure(figsize=(10, 8))

        if var == "angle":
            # Calculate histogram with normalized probability density
            counts, bin_edges = np.histogram(self.theta, bins=25, density=False)
            bin_width = np.diff(bin_edges)
            probability_density = counts / (counts.sum() * bin_width)
            y_values = np.arange(self.thetamin, self.thetamax, 0.0003)

            # Find emax for each angle in the array to integrate
            point_bilinear_interpolation = np.column_stack((self.emin * np.ones_like(y_values), y_values))
            emax_integration = _bilinear_interpolation(point_bilinear_interpolation, self.grid_m, self.grid_a, self.energy_distr_grid)
            
            # Function for integration
            def integrand(y):
                def inner_integrand(z):
                    return self.interpolating_function((self.emin, y, z))
                emax = emax_integration[_searchsorted_opt(y_values, y)]  # Finds max energy for the given angle y
                return nquad(inner_integrand, [[self.emin, emax]], opts={"epsabs": 1e-2, "epsrel": 1e-2})[0]
            
            # Calculate the angular distribution
            angular_distribution = np.asarray([integrand(y) for y in y_values])
            # Normalize the angular distribution by integrating
            total_integral = simps(angular_distribution, y_values)
            distribution_normalized = angular_distribution / total_integral

            plt.plot(y_values, distribution_normalized, label="Normalized " + var + " Distribution", color="black")
            plt.xlabel("θ [Rad]")
            plt.ylabel("$f_{θ}$")
            plt.title(self.LLP + " Angular Distribution")

        elif var == "energy":
            # Calculate histogram with normalized probability density
            counts, bin_edges = np.histogram(self.energy, bins=25, density=False)
            bin_width = np.diff(bin_edges)
            probability_density = counts / (counts.sum() * bin_width)

            # Function for integration
            def integrand(z):
                def inner_integrand(y):
                    e_max = _bilinear_interpolation(np.asarray([[self.emin, y]]), self.grid_m, self.grid_a, self.energy_distr_grid)
                    if z > e_max:
                        return 0
                    else:
                        return self.interpolating_function((self.emin, y, z))
                return nquad(inner_integrand, [[self.thetamin, self.thetamax]], opts={"epsabs": 1e-2, "epsrel": 1e-2})[0]

            # Calculate the energy distribution
            z_values = np.arange(self.emin, self.emax, 1)
            energy_distribution = np.asarray([integrand(z) for z in z_values])

            # Normalize the energy distribution by integrating
            total_integral = simps(energy_distribution, z_values)
            distribution_normalized = energy_distribution / total_integral

            plt.plot(z_values, distribution_normalized, label="Normalized " + var + " Distribution", color="black")
            plt.xlabel("E [GeV]")
            plt.ylabel("$f_{E}$")
            plt.title(self.LLP + " Energy Distribution")

        else:
            raise ValueError("Invalid variable. Must be 'angle' or 'energy'.")

        # Plot the probability density histogram
        plt.bar(bin_edges[:-1], probability_density, width=bin_width, alpha=0.7, label=var.capitalize() + " Points Probability Density")

        plt.legend()
        fig.savefig(self.path + "/" + var + ".png")
        plt.close(fig)  # Close the figure to free memory
