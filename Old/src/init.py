import os
import numpy as np
import pandas as pd

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

    Methods
    -------
    __init__()
        Initializes the LLP instance and prompts the user for necessary inputs.
    select_particle()
        Prompts the user to select a particle from available folders.
    import_distributions()
        Imports distribution and energy files for the selected particle.
    import_decay_channels()
        Imports decay channels for the selected particle.    
    prompt_mass_and_ctau()
        Prompts the user to input the mass and c*tau values.
    """

    def __init__(self):
        """
        Initializes the LLP instance by setting up the main folder path, selecting the particle, 
        importing distribution data, and prompting for mass and c*tau inputs.
        """
        self.main_folder = "./Distributions"
        self.particle_path = ""
        self.Distr = None
        self.Energy_distr = None
        self.LLP_name = ""
        self.mass = 0.0
        self.c_tau = 0.0
        self.decay_channels = None

        self.select_particle()
        self.import_distributions()
        self.prompt_mass_and_ctau()
        self.import_decay_channels()

    def select_particle(self):
        """
        Prompts the user to select a particle from the available subfolders in the main folder.
        
        Raises
        ------
        ValueError
            If the selected index is invalid or out of range.
        """
        folders = np.array(os.listdir(self.main_folder))
        
        print("\n Particle selector \n")
        for i, folder in enumerate(folders):
            print(f"{i + 1}. {folder}")

        try:
            selected_particle = int(input("Select particle: ")) - 1
            particle_distr_folder = folders[selected_particle]
        except (IndexError, ValueError):
            raise ValueError("Invalid selection. Please select a valid particle.")

        self.particle_path = os.path.join(self.main_folder, particle_distr_folder)
        self.LLP_name = particle_distr_folder.replace("_", " ")

    def import_distributions(self):
        """
        Imports the distribution and energy files for the selected particle and stores them in dataframes.
        
        Raises
        ------
        FileNotFoundError
            If the distribution or energy files are not found in the selected particle folder.
        Exception
            For any other errors that occur during file import.
        """
        try:
            files = os.listdir(self.particle_path)
            distribution_file = next(f for f in files if f.startswith('D'))
            energy_file = next(f for f in files if f.startswith('E'))
            BrRatios_file = next(f for f in files if f.startswith('BrR'))

            distribution_file_path = os.path.join(self.particle_path, distribution_file)
            energy_file_path = os.path.join(self.particle_path, energy_file)
            BrRatios_file_path = os.path.join(self.particle_path, BrRatios_file)

            self.Distr = pd.read_csv(distribution_file_path, header=None, sep="\t")
            self.Energy_distr = pd.read_csv(energy_file_path, header=None, sep="\t")
            self.BrRatios_distr = pd.read_csv(BrRatios_file_path, header=None, sep="\t")
            
        except StopIteration:
            raise FileNotFoundError("Distribution file not found in the selected particle folder.")
        except Exception as e:
            raise Exception(f"Error importing distributions: {e}")

    def prompt_mass_and_ctau(self):
        """
        Prompts the user to input the mass and c*tau (lifetime) values for the LLP.
        
        Raises
        ------
        ValueError
            If the input values for mass or c*tau are not valid floating point numbers.
        """
        try:
            self.mass = float(input("\nLLP mass: "))
            self.c_tau = float(input("\nLife time c*tau: "))
        except ValueError:
            raise ValueError("Invalid input for mass or c*tau. Please enter numerical values.")


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
            [parse_value(p) for p in line.strip().split()[:8]]
            for line in lines
            if line.strip() and not line.startswith('#')
            ]
            
            return np.array(data)

        try:
            decay_channels_path = os.path.join(self.particle_path, 'decay_channels.txt')
            self.decay_channels = create_array_from_file(decay_channels_path)
                
        except FileNotFoundError:
            raise FileNotFoundError("Decay channels file not found in the selected particle folder.")
        except Exception as e:
            raise Exception(f"Error importing decay channels: {e}")