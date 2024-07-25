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

        self.select_particle()
        self.import_distributions()
        self.prompt_mass_and_ctau()

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
            raise FileNotFoundError("Distribution or Energy file not found in the selected particle folder.")
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
