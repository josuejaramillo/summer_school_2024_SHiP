import os
import numpy as np
import pandas as pd
from funcs import HNLmerging
from scipy.interpolate import RegularGridInterpolator
from funcs import rotateVectors

class LLP:
    """
    A class used to represent a Long-Lived Particle (LLP) with properties and methods 
    for particle selection, data import, and parameter input.

    Attributes
    ----------
    main_folder : str
        The main folder containing the particle distribution subfolders.
    particle_path : str
        The path to the selected particle's folder.
    Distr : pd.DataFrame
        The dataframe containing the distribution data of the selected particle.
    Energy_distr : pd.DataFrame
        The dataframe containing the energy distribution data of the selected particle.
    LLP_name : str
        The name of the selected LLP.
    mass : float
        The mass of the selected LLP.
    c_tau : float
        The lifetime (c*tau) of the selected LLP.
    BrRatios_distr : np.ndarray
        The branching ratios distribution interpolated at the given mass.
    Matrix_elements : Any
        Matrix elements merged with decay width data.
    MixingPatternArray : np.ndarray
        An array containing the mixing pattern (Ue2, Umu2, Utau2) for the LLP.

    Methods
    -------
    __init__()
        Initializes the LLP instance and prompts the user for necessary inputs.
    select_particle()
        Prompts the user to select a particle from available folders.
    import_particle()
        Imports the data based on the particle type (Higgs-like scalar or HNL).
    import_scalars()
        Imports scalar particle data and distributions.
    import_HNL()
        Imports HNL (Heavy Neutral Lepton) data and distributions.
    LLP_BrRatios(m, LLP_BrRatios)
        Interpolates branching ratios for a given mass.
    mergeHNL(BrRatios, Matrix_elements, decayWidthData, yieldData, massDistrData, DistrDataFrames)
        Merges HNL data including branching ratios, decay widths, and distributions.
    prompt_mixing_pattern()
        Prompts the user for mixing pattern (Ue2, Umu2, Utau2) values and normalizes them.
    prompt_mass_and_ctau()
        Prompts the user to input the mass and c*tau values for the LLP.
    import_decay_channels()
        Imports decay channels data for the selected particle.
    """

    def __init__(self):
        """
        Initializes the LLP instance by setting up the main folder path, selecting the particle, 
        importing distribution data, and prompting for mass and c*tau inputs.
        """
        self.main_folder = "./Distributions"  # Directory containing particle distribution data
        self.LLP_name = ""  # Placeholder for LLP name
        self.MixingPatternArray = None
        self.Matrix_elements = None  # Placeholder for matrix elements

        # Initialize particle selection and mass/lifetime prompts
        self.select_particle()
        self.prompt_mass_and_ctau()

        # Import particle-specific data
        self.import_particle()
        # self.import_distributions()  # Commented out as per the original code

        # Uncomment the following block if you want to precompile vector rotation functions
        # self.precompileRotateVectors()

    # def precompileRotateVectors(self):
    #     """
    #     Pre-compiles rotation functions for performance optimization.
    #     Calls rotation functions once with dummy data to optimize subsequent calls.
    #     """
    #     rotateVectors.p1rotatedX_jit(1, 1, 1, 1)
    #     rotateVectors.p1rotatedY_jit(1, 1, 1, 1)
    #     rotateVectors.p1rotatedZ_jit(1, 1, 1, 1)

    #     rotateVectors.p2rotatedX_jit(10, 10, 10, 1, 1, 1, 1, 1, 1)
    #     rotateVectors.p2rotatedY_jit(10, 10, 10, 1, 1, 1, 1, 1, 1)
    #     rotateVectors.p2rotatedZ_jit(10, 10, 10, 1, 1, 1, 1, 1, 1)

    #     rotateVectors.p3rotatedX_jit(10, 10, 10, 1, 1, 1, 1, 1, 1)
    #     rotateVectors.p3rotatedY_jit(10, 10, 10, 1, 1, 1, 1, 1, 1)
    #     rotateVectors.p3rotatedZ_jit(10, 10, 10, 1, 1, 1, 1, 1, 1)

    def select_particle(self):
        """
        Prompts the user to select a particle from the available subfolders in the main folder.
        
        Raises
        ------
        ValueError
            If the selected index is invalid or out of range.
        """
        folders = np.array(os.listdir(self.main_folder))  # List all subfolders in the main folder
        
        print("\n Particle selector \n")
        for i, folder in enumerate(folders):
            print(f"{i + 1}. {folder}")  # Display available folders for selection

        try:
            # User selects the particle by inputting the corresponding number
            selected_particle = int(input("Select particle: ")) - 1
            particle_distr_folder = folders[selected_particle]
        except (IndexError, ValueError):
            raise ValueError("Invalid selection. Please select a valid particle.")  # Raise error for invalid selection

        # Set the particle path and name based on user selection
        self.particle_path = os.path.join(self.main_folder, particle_distr_folder)
        self.LLP_name = particle_distr_folder.replace("_", " ")

    def import_particle(self):
        """
        Imports particle-specific data depending on the selected LLP type.
        Calls respective import functions for Higgs-like scalars or Heavy Neutral Leptons (HNLs).
        """
        if self.LLP_name == "Higgs like scalars":
            self.import_scalars()  # Import data for Higgs-like scalars
        elif self.LLP_name == "HNL":
            self.import_HNL()  # Import data for Heavy Neutral Leptons (HNLs)

    def import_scalars(self):
        """
        Imports scalar particle data and distributions.
        Loads distribution, energy, and branching ratio data from respective files.
        """
        # # List all files in the particle folder
        # files = os.listdir(self.particle_path)
        
        # # Identify relevant files by their prefixes (D for distribution, E for energy, BrR for branching ratios)
        # distribution_file = next(f for f in files if f.startswith('D'))
        # energy_file = next(f for f in files if f.startswith('E'))
        # BrRatios_file = next(f for f in files if f.startswith('BrR'))

        # # Construct full paths to the identified files
        # distribution_file_path = os.path.join(self.particle_path, distribution_file)
        # energy_file_path = os.path.join(self.particle_path, energy_file)
        # BrRatios_file_path = os.path.join(self.particle_path, BrRatios_file)

        # # Load data into pandas DataFrames
        # self.Distr = pd.read_csv(distribution_file_path, header=None, sep="\t")
        # self.Energy_distr = pd.read_csv(energy_file_path, header=None, sep="\t")
        
        # # Interpolate branching ratios at the given mass
        # self.BrRatios_distr = self.LLP_BrRatios(self.mass, pd.read_csv(BrRatios_file_path, header=None, sep="\t"))

        # self.import_decay_channels()  # Import decay channel data

        distribution_file_path = os.path.join(self.particle_path, "Double-Distr-BC4.dat")
        decay_json_path = os.path.join(self.particle_path, "HLS-decay.json")
        energy_file_path = os.path.join(self.particle_path, "Emax-BC4.dat")

        # Load data into pandas DataFrames
        self.Distr = pd.read_csv(distribution_file_path, header=None, sep="\t")
        self.Energy_distr = pd.read_csv(energy_file_path, header=None, sep="\t")

        # Load decay data from JSON file
        HLS_decay = pd.read_json(decay_json_path)

        # Extract decay channels and particle PDGs (Particle Data Group IDs)
        self.decayChannels = HLS_decay.iloc[:, 0].to_numpy()
        self.PDGs = HLS_decay.iloc[:, 1].apply(np.array).to_numpy()

        # Interpolate branching ratios at the given mass
        BrRatios = np.array(HLS_decay.iloc[:, 2])

        self.BrRatios_distr = self.LLP_BrRatios(self.mass, BrRatios)


    def LLP_BrRatios(self, m, LLP_BrRatios):
        """
        Interpolates branching ratios for a given mass using the loaded branching ratio data.

        Parameters
        ----------
        m : float
            Mass of the LLP.
        LLP_BrRatios : pd.DataFrame
            DataFrame containing branching ratios for different masses.

        Returns
        -------
        np.ndarray
            Interpolated branching ratios at the specified mass.
        """
        # mass_axis = LLP_BrRatios[0]  # Extract mass axis from the first column
        # channels = LLP_BrRatios.columns[1:]  # Get the branching ratio channels (columns)


        # Create interpolators for each channel
        # interpolators = np.asarray([RegularGridInterpolator((mass_axis,), LLP_BrRatios[channel].values) for channel in channels])
        
        # Evaluate interpolators at the specified mass
        # return np.array([interpolator([m])[0] for interpolator in interpolators])


        mass_axis = np.array(LLP_BrRatios[0])[:,0]

        # Create interpolators for each channel
        interpolators = np.asarray([RegularGridInterpolator((mass_axis,), np.array(LLP_BrRatios[i])[:,1]) for i in range(len(LLP_BrRatios))])        
        return np.array([interpolator([m])[0] for interpolator in interpolators])
        

    def import_HNL(self):
        """
        Imports Heavy Neutral Lepton (HNL) data including decay channels, decay widths, 
        yield data, and distributions from various files.
        """
        # Define paths to various files required for HNL data import
        decay_json_path = os.path.join(self.particle_path, "HNL-decay.json")
        decay_width_path = os.path.join(self.particle_path, "HNLdecayWidth.dat")

        yield_e_path = os.path.join(self.particle_path, "Total-yield-HNL-e.txt")
        yield_mu_path = os.path.join(self.particle_path, "Total-yield-HNL-mu.txt")
        yield_tau_path = os.path.join(self.particle_path, "Total-yield-HNL-tau.txt")

        distrHNL_e_path = os.path.join(self.particle_path, "DoubleDistrHNL-Mixing-e.txt")
        distrHNL_mu_path = os.path.join(self.particle_path, "DoubleDistrHNL-Mixing-mu.txt")
        distrHNL_tau_path = os.path.join(self.particle_path, "DoubleDistrHNL-Mixing-tau.txt")


        energy_file_path = os.path.join(self.particle_path, "Emax_HNL.txt")
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

        self.prompt_mixing_pattern()  # Prompt the user for mixing pattern inputs

        # Load HNL data from specified paths
        self.decayChannels, self.PDGs, BrRatios, Matrix_elements, decayWidthData, yieldData, massDistrData, DistrDataFrames = HNLmerging.load_data(paths)
        
        # Merge HNL data (branching ratios, matrix elements, distributions)
        self.mergeHNL(BrRatios, Matrix_elements, decayWidthData, yieldData, massDistrData, DistrDataFrames)

    def mergeHNL(self, BrRatios, Matrix_elements, decayWidthData, yieldData, massDistrData, DistrDataFrames):
        """
        Merges HNL data including branching ratios, matrix elements, decay widths, and distributions.

        Parameters
        ----------
        BrRatios : pd.DataFrame
            DataFrame containing branching ratios.
        Matrix_elements : pd.DataFrame
            DataFrame containing matrix elements.
        decayWidthData : pd.DataFrame
            DataFrame containing decay width data.
        yieldData : list
            List of yield data files.
        massDistrData : pd.DataFrame
            DataFrame containing mass distribution data.
        DistrDataFrames : pd.DataFrame
            DataFrame containing distribution data.
        """
        # Compute decay widths for the given mass
        decay_widths = HNLmerging.compute_decay_widths(self.mass, decayWidthData)
        
        # Compute and merge branching ratios using mixing pattern and decay widths
        self.BrRatios_distr = HNLmerging.compute_BrMerged(self.mass, BrRatios, self.MixingPatternArray, decay_widths)
        
        # Merge matrix elements using mixing pattern and decay widths
        # self.Matrix_elementsMerged = HNLmerging.mergeMatrixElements(self.mass, Matrix_elements, decay_widths, self.MixingPatternArray)
        self.Matrix_elements = HNLmerging.MatrixElements(Matrix_elements, decay_widths, self.MixingPatternArray)
        
        # Merge distributions using mixing pattern, yield data, and distribution data
        self.Distr = HNLmerging.merge_distributions(massDistrData, self.MixingPatternArray, yieldData, DistrDataFrames)

    def prompt_mixing_pattern(self):
        """
        Prompts the user for mixing pattern values (Ue2, Umu2, Utau2) and normalizes them to ensure
        that the sum equals 1.

        Raises
        ------
        ValueError
            If the input values are not valid floats.
        """
        try:
            # Get mixing pattern values from user input
            self.Ue2 = float(input("\nUe2: "))
            self.Umu2 = float(input("\nUmu2: "))
            self.Utau2 = float(input("\nUtau2: "))

            # Create mixing pattern array
            self.MixingPatternArray = np.array([self.Ue2, self.Umu2, self.Utau2])

            # Normalize mixing pattern if the sum is not equal to 1
            sumMixingPattern = self.Ue2 + self.Umu2 + self.Utau2
            if sumMixingPattern != 1:
                # Normalize each component
                self.Ue2 = self.Ue2 / sumMixingPattern
                self.Umu2 = self.Umu2 / sumMixingPattern
                self.Utau2 = self.Utau2 / sumMixingPattern
                self.MixingPatternArray = np.array([self.Ue2, self.Umu2, self.Utau2])

        except ValueError:
            raise ValueError("Invalid input. Please enter numerical values.")  # Raise error for invalid input

    def prompt_mass_and_ctau(self):
        """
        Prompts the user to input the mass and c*tau (lifetime) values for the LLP.
        
        Raises
        ------
        ValueError
            If the input values for mass or c*tau are not valid floating point numbers.
        """
        try:
            # Prompt for LLP mass and lifetime (c*tau)
            self.mass = float(input("\nLLP mass: "))
            self.c_tau = float(input("\nLife time c*tau: "))
        except ValueError:
            raise ValueError("Invalid input for mass or c*tau. Please enter numerical values.")  # Raise error for invalid input

    def import_decay_channels(self):
        """
        Imports decay channels data from a text file and stores it as a NumPy array.

        The function reads a text file containing decay channel data, processes each line to extract relevant
        values, and converts them into a NumPy array. The file is expected to be located in the directory
        specified by `self.particle_path` with the filename 'decay_channels.txt'.

        The text file should have rows with values separated by spaces or tabs. Each line should contain:
        - Mass (twice, for two identical values)
        - Particle IDs (two entries for particle and antiparticle)
        - Charges of the particles (two entries)
        - Additional information (four entries for specific cases)

        Raises:
            FileNotFoundError: If the decay channels file cannot be found in the specified directory.
            Exception: For any other errors encountered during file processing.

        Attributes:
            self.decay_channels (np.ndarray): A NumPy array containing the processed decay channel data.
        """

        def parse_value(val):
            """
            Converts a string value to a float or fraction.

            Tries to evaluate the value as a Python expression (e.g., fractions like '1/3') or converts it
            directly to a float if evaluation fails.

            Args:
                val (str): The string representation of the value to convert.

            Returns:
                float: The converted float or fractional value.
            """
            try:
                return eval(val)  # Convert to float or fraction
            except:
                return float(val)  # In case of direct float representation

        def create_array_from_file(filename):
            """
            Reads a text file and creates a NumPy array from its contents.

            Processes each line of the file, extracting and converting values to create a NumPy array.

            Args:
                filename (str): The path to the text file containing decay channel data.

            Returns:
                np.ndarray: A NumPy array with the processed decay channel data.
            """
            with open(filename, 'r') as file:
                lines = file.readlines()
            
            data = [
                [parse_value(p) for p in line.strip().split()[:8]]  # Extract the first 8 elements of each line
                for line in lines
                if line.strip() and not line.startswith('#')  # Skip empty lines and comments
            ]
            
            return np.array(data)

        try:
            # Construct the full path to the decay channels file
            decay_channels_path = os.path.join(self.particle_path, 'decay_channels_4struct.txt')
            
            # Create an array from the file contents
            decay_channels = create_array_from_file(decay_channels_path)
            
            # Extract particle IDs (PDGs) from the decay channels
            self.PDGs = decay_channels[:, 4:8]  # Columns 4 to 8 correspond to particle IDs
                
        except FileNotFoundError:
            raise FileNotFoundError("Decay channels file not found in the selected particle folder.")  # Raise error if file is not found
        except Exception as e:
            raise Exception(f"Error importing decay channels: {e}")  # Raise general error for any other issues
