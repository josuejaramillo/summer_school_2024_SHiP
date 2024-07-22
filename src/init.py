import os
import numpy as np
import pandas as pd


class LLP:

    def __init__(self):
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
        self.particle_path = main_folder+"/"+particle_distr_folder

        #Distribution import
        self.Distr = pd.read_csv(distribution_file_path, header=None, sep="\t")
        self.Energy_distr = pd.read_csv(energy_file_path, header=None, sep="\t")

        #LLP name
        self.LLP_name = folders[selected_particle].replace("_", " ")

        #mass selection
        self.mass = float(input("\nLLP mass: "))

        #c*tau selection
        self.c_tau = float(input("\nLife time c*tau: "))

        