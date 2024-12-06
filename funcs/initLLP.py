# initLLP.py
import os
import numpy as np
import pandas as pd
from funcs import HNLmerging
from scipy.interpolate import RegularGridInterpolator
import sympy as sp  # Import sympy for matrix element compilation

class LLP:
    """
    A class used to represent a Long-Lived Particle (LLP) with properties and methods 
    for particle selection, data import, and parameter input.
    """

    def __init__(self, mass, particle_selection, mixing_pattern=None, uncertainty=None):
        """
        Initializes the LLP instance by setting up the main folder path, selecting the particle, 
        and importing distribution data.
        """
        self.main_folder = "./Distributions"  # Directory containing particle distribution data
        self.LLP_name = particle_selection['LLP_name']
        self.Matrix_elements = None  # Placeholder for matrix elements

        # Set mass
        self.mass = mass

        # Set particle selection
        self.particle_path = particle_selection['particle_path']

        # For HNL, set the mixing pattern
        if self.LLP_name == "HNL":
            if mixing_pattern is None:
                raise ValueError("Mixing pattern must be provided for HNL.")
            else:
                self.MixingPatternArray = mixing_pattern
        else:
            self.MixingPatternArray = None  # Not needed for other particles

        # For Dark-photons, set the uncertainty
        if self.LLP_name == "Dark-photons":
            if uncertainty is None:
                raise ValueError("Uncertainty must be provided for Dark-photons.")
            else:
                self.uncertainty = uncertainty
        else:
            self.uncertainty = None  # Not needed for other particles

        # Import particle-specific data (mass-independent)
        self.import_particle()

    def compute_mass_dependent_properties(self):
        """
        Computes properties that depend on mass.
        """
        if self.mass is None:
            raise ValueError("Mass must be set before computing mass-dependent properties.")
        if "Scalar" in self.LLP_name:
            self.compute_mass_dependent_properties_scalars()
        elif self.LLP_name == "HNL":
            self.compute_mass_dependent_properties_HNL()
        elif self.LLP_name == "ALP-photon":
            self.compute_mass_dependent_properties_ALP_photon()
        elif self.LLP_name == "Dark-photons":
            self.compute_mass_dependent_properties_dark_photons()
        else:
            raise ValueError("Unknown LLP name.")

    def set_c_tau(self, c_tau_input):
        """
        Sets the user-specified lifetime (c_tau_input) for the LLP.
        """
        self.c_tau_input = c_tau_input

    def import_particle(self):
        """
        Imports particle-specific data depending on the selected LLP type.
        Calls respective import functions for scalars or Heavy Neutral Leptons (HNLs).
        """
        if "Scalar" in self.LLP_name:
            self.import_scalars()  # Import data for scalars
        elif self.LLP_name == "HNL":
            self.import_HNL()  # Import data for Heavy Neutral Leptons (HNLs)
        elif self.LLP_name == "ALP-photon":
            self.import_ALP_photon()  # Import data for ALP-photon
        elif self.LLP_name == "Dark-photons":
            self.import_dark_photons()  # Import data for Dark-photons
        else:
            raise ValueError("Unknown LLP name.")

    def import_scalars(self):
        """
        Imports scalar particle data and distributions (mass-independent).
        """
        if "mixing" in self.LLP_name:
            suffix = "mixing"
        elif "quartic" in self.LLP_name:
            suffix = "quartic"
        else:
            raise ValueError("Unknown Scalar type in LLP_name. Must include 'mixing' or 'quartic'.")

        # Define file paths based on the suffix
        distribution_file_path = os.path.join(self.particle_path, f"DoubleDistr-Scalar-{suffix}.txt")
        energy_file_path = os.path.join(self.particle_path, f"Emax-Scalar-{suffix}.txt")
        yield_path = os.path.join(self.particle_path, f"Total-yield-Scalar-{suffix}.txt")
        ctau_path = os.path.join(self.particle_path, "ctau-Scalar.txt")
        decay_json_path = os.path.join(self.particle_path, "HLS-decay.json")

        # Load data into pandas DataFrames
        self.Distr = pd.read_csv(distribution_file_path, header=None, sep="\t")
        self.Energy_distr = pd.read_csv(energy_file_path, header=None, sep="\t")

        # Load yield data
        self.Yield_data = pd.read_csv(yield_path, header=None, sep="\t")

        # Load ctau data
        self.ctau_data = pd.read_csv(ctau_path, header=None, sep="\t")

        # Create interpolators for ctau and yield
        mass_ctau = self.ctau_data.iloc[:, 0].to_numpy()
        ctau_values = self.ctau_data.iloc[:, 1].to_numpy()
        self.ctau_interpolator = RegularGridInterpolator(
            (mass_ctau,), ctau_values, bounds_error=False, fill_value=None
        )

        mass_yield = self.Yield_data.iloc[:, 0].to_numpy()
        yield_values = self.Yield_data.iloc[:, 1].to_numpy()
        self.yield_interpolator = RegularGridInterpolator(
            (mass_yield,), yield_values, bounds_error=False, fill_value=None
        )

        # Load decay data from JSON file
        HLS_decay = pd.read_json(decay_json_path)

        # Extract decay channels and particle PDGs
        self.decayChannels = HLS_decay.iloc[:, 0].to_numpy()
        self.PDGs = HLS_decay.iloc[:, 1].apply(np.array).to_numpy()

        # Store branching ratios for later interpolation
        self.BrRatios = np.array(HLS_decay.iloc[:, 2])

        # For scalars, Matrix_elements may not be required
        self.Matrix_elements = None  # Or set appropriately if needed

    def compute_mass_dependent_properties_scalars(self):
        """
        Computes mass-dependent properties for scalars.
        """
        # Interpolate branching ratios at the given mass
        self.BrRatios_distr = self.LLP_BrRatios(self.mass, self.BrRatios)

        # Interpolate intrinsic ctau at the given mass
        self.c_tau_int = self.ctau_interpolator([self.mass])[0]

        # Interpolate yield at the given mass
        self.Yield = self.yield_interpolator([self.mass])[0]

    def import_ALP_photon(self):
        """
        Imports ALP-photon particle data and distributions (mass-independent).
        """
        # Define file paths
        distribution_file_path = os.path.join(self.particle_path, "DoubleDistr-ALP-photon.txt")
        energy_file_path = os.path.join(self.particle_path, "Emax-ALP-photon.txt")
        yield_path = os.path.join(self.particle_path, "Total-yield-ALP-photon.txt")
        ctau_path = os.path.join(self.particle_path, "ctau-ALP.txt")
        decay_json_path = os.path.join(self.particle_path, "ALP-photon-decay.json")

        # Load data into pandas DataFrames
        self.Distr = pd.read_csv(distribution_file_path, header=None, sep="\t")
        self.Energy_distr = pd.read_csv(energy_file_path, header=None, sep="\t")

        # Load yield data
        self.Yield_data = pd.read_csv(yield_path, header=None, sep="\t")

        # Load ctau data
        self.ctau_data = pd.read_csv(ctau_path, header=None, sep="\t")

        # Create interpolators for ctau and yield
        mass_ctau = self.ctau_data.iloc[:, 0].to_numpy()
        ctau_values = self.ctau_data.iloc[:, 1].to_numpy()
        self.ctau_interpolator = RegularGridInterpolator(
            (mass_ctau,), ctau_values, bounds_error=False, fill_value=None
        )

        mass_yield = self.Yield_data.iloc[:, 0].to_numpy()
        yield_values = self.Yield_data.iloc[:, 1].to_numpy()
        self.yield_interpolator = RegularGridInterpolator(
            (mass_yield,), yield_values, bounds_error=False, fill_value=None
        )

        # Load decay data from JSON file
        ALP_decay = pd.read_json(decay_json_path)

        # Extract decay channels, particle PDGs, branching ratios, and matrix elements
        self.decayChannels = ALP_decay.iloc[:, 0].to_numpy()
        self.PDGs = ALP_decay.iloc[:, 1].apply(np.array).to_numpy()
        self.BrRatios = np.array(ALP_decay.iloc[:, 2])

        # Extract matrix elements from the last column
        self.Matrix_elements_raw = ALP_decay.iloc[:, -1].to_numpy()

        # Compile matrix elements
        self.compile_matrix_elements()

    def compute_mass_dependent_properties_ALP_photon(self):
        """
        Computes mass-dependent properties for ALP-photon.
        """
        # Interpolate branching ratios at the given mass
        self.BrRatios_distr = self.LLP_BrRatios(self.mass, self.BrRatios)

        # Interpolate intrinsic ctau at the given mass
        self.c_tau_int = self.ctau_interpolator([self.mass])[0]

        # Interpolate yield at the given mass
        self.Yield = self.yield_interpolator([self.mass])[0]

    def import_dark_photons(self):
        """
        Imports Dark-photon particle data and distributions (mass-independent).
        """
        # Ensure that uncertainty is set
        if self.uncertainty is None:
            raise ValueError("Uncertainty must be provided for Dark-photons.")

        # Define file paths using the uncertainty
        distribution_file_path = os.path.join(self.particle_path, f"DoubleDistr-DP-{self.uncertainty}.txt")
        energy_file_path = os.path.join(self.particle_path, f"Emax-DP-{self.uncertainty}.txt")
        yield_path = os.path.join(self.particle_path, f"Total-yield-DP-{self.uncertainty}.txt")
        ctau_path = os.path.join(self.particle_path, "ctau-DP.txt")
        decay_json_path = os.path.join(self.particle_path, "DP-decay.json")

        # Load data into pandas DataFrames
        self.Distr = pd.read_csv(distribution_file_path, header=None, sep="\t")
        self.Energy_distr = pd.read_csv(energy_file_path, header=None, sep="\t")

        # Load yield data
        self.Yield_data = pd.read_csv(yield_path, header=None, sep="\t")

        # Load ctau data
        self.ctau_data = pd.read_csv(ctau_path, header=None, sep="\t")

        # Create interpolators for ctau and yield
        mass_ctau = self.ctau_data.iloc[:, 0].to_numpy()
        ctau_values = self.ctau_data.iloc[:, 1].to_numpy()
        self.ctau_interpolator = RegularGridInterpolator(
            (mass_ctau,), ctau_values, bounds_error=False, fill_value=None
        )

        mass_yield = self.Yield_data.iloc[:, 0].to_numpy()
        yield_values = self.Yield_data.iloc[:, 1].to_numpy()
        self.yield_interpolator = RegularGridInterpolator(
            (mass_yield,), yield_values, bounds_error=False, fill_value=None
        )

        # Load decay data from JSON file
        DP_decay = pd.read_json(decay_json_path)

        # Extract decay channels, particle PDGs, branching ratios, and matrix elements
        self.decayChannels = DP_decay.iloc[:, 0].to_numpy()
        self.PDGs = DP_decay.iloc[:, 1].apply(np.array).to_numpy()
        self.BrRatios = np.array(DP_decay.iloc[:, 2])

        # Extract matrix elements from the last column
        self.Matrix_elements_raw = DP_decay.iloc[:, -1].to_numpy()

        # Compile matrix elements
        self.compile_matrix_elements()

    def compute_mass_dependent_properties_dark_photons(self):
        """
        Computes mass-dependent properties for Dark-photons.
        """
        # Interpolate branching ratios at the given mass
        self.BrRatios_distr = self.LLP_BrRatios(self.mass, self.BrRatios)

        # Interpolate intrinsic ctau at the given mass
        self.c_tau_int = self.ctau_interpolator([self.mass])[0]

        # Interpolate yield at the given mass
        self.Yield = self.yield_interpolator([self.mass])[0]

    def compile_matrix_elements(self):
        """
        Compiles matrix element expressions into callable functions.
        """
        # Define symbols for sympy
        mLLP, E1, E3 = sp.symbols('mLLP E1 E3')

        # Define locals for sympify
        local_dict = {'E1': E1, 'E3': E3, 'mLLP': mLLP}

        compiled_expressions = []
        for expr_str in self.Matrix_elements_raw:
            if expr_str not in [None, "", "-"]:
                # Replace '***' with 'e' to handle scientific notation
                expr_str_corrected = expr_str.replace('***', 'e')

                # Parse the corrected expression with local variables
                expr = sp.sympify(expr_str_corrected, locals=local_dict)
                func = sp.lambdify((mLLP, E1, E3), expr, 'numpy')
                compiled_expressions.append(func)
            else:
                compiled_expressions.append(None)
        self.Matrix_elements = compiled_expressions

    def LLP_BrRatios(self, m, LLP_BrRatios):
        """
        Interpolates branching ratios for a given mass using the loaded branching ratio data.
        """
        mass_axis = np.array(LLP_BrRatios[0])[:, 0]

        # Create interpolators for each channel
        interpolators = np.asarray([
            RegularGridInterpolator((mass_axis,), np.array(LLP_BrRatios[i])[:, 1], bounds_error=False, fill_value=0.0)
            for i in range(len(LLP_BrRatios))
        ])
        return np.array([interpolator([m])[0] for interpolator in interpolators])

    def import_HNL(self):
        """
        Imports Heavy Neutral Lepton (HNL) data including decay channels, decay widths, 
        yield data, and distributions from various files (mass-independent).
        """
        # Define paths to various files required for HNL data import
        decay_json_path = os.path.join(self.particle_path, "HNL-decay.json")
        decay_width_path = os.path.join(self.particle_path, "HNLdecayWidth.dat")

        yield_e_path = os.path.join(self.particle_path, "Total-yield-HNL-e.txt")
        yield_mu_path = os.path.join(self.particle_path, "Total-yield-HNL-mu.txt")
        yield_tau_path = os.path.join(self.particle_path, "Total-yield-HNL-tau.txt")

        distrHNL_e_path = os.path.join(self.particle_path, "DoubleDistr-HNL-mixing-e.txt")
        distrHNL_mu_path = os.path.join(self.particle_path, "DoubleDistr-HNL-mixing-mu.txt")
        distrHNL_tau_path = os.path.join(self.particle_path, "DoubleDistr-HNL-mixing-tau.txt")

        energy_file_path = os.path.join(self.particle_path, "Emax-HNL.txt")
        self.Energy_distr = pd.read_csv(energy_file_path, header=None, sep="\t")

        # Tuple containing all the required paths
        paths = (
            decay_json_path,
            decay_width_path,
            yield_e_path,
            yield_mu_path,
            yield_tau_path,
            distrHNL_e_path,
            distrHNL_mu_path,
            distrHNL_tau_path
        )

        # Load HNL data from specified paths
        (
            self.decayChannels,
            self.PDGs,
            self.BrRatios,
            self.Matrix_elements_raw,
            self.decayWidthData,
            self.yieldData,
            self.massDistrData,
            self.DistrDataFrames
        ) = HNLmerging.load_data(paths)

    def compute_mass_dependent_properties_HNL(self):
        """
        Computes mass-dependent properties for HNLs.
        """
        # Compute decay widths for the given mass
        decay_widths = HNLmerging.compute_decay_widths(self.mass, self.decayWidthData)

        # Compute and merge branching ratios using mixing pattern and decay widths
        self.BrRatios_distr = HNLmerging.compute_BrMerged(
            self.mass, self.BrRatios, self.MixingPatternArray, decay_widths
        )

        # Merge matrix elements using mixing pattern and decay widths
        self.Matrix_elements = HNLmerging.MatrixElements(
            self.Matrix_elements_raw, decay_widths, self.MixingPatternArray
        )

        # Merge distributions using mixing pattern, yield data, and distribution data
        self.Distr = HNLmerging.merge_distributions(
            self.massDistrData, self.MixingPatternArray, self.yieldData, self.DistrDataFrames
        )

        # Compute intrinsic c_tau and Yield
        self.c_tau_int = HNLmerging.compute_ctau(self.mass, self.decayWidthData, self.MixingPatternArray)
        self.Yield = HNLmerging.compute_total_yield(self.mass, self.yieldData, self.MixingPatternArray)

