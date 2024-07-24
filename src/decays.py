import numpy as np
import numba as nb
from src.decays_functions import decay_products
import time
import pandas as pd
import os

"""
This module defines decay properties for various particles and organizes them into models of Long-Lived Particles (LLPs).

scalar_decays:
    A dictionary containing decay properties for different decay channels.
    Keys:
        - "e+e-"    : Electron-positron decay channel.
        - "mu+mu-"  : Muon-antimuon decay channel.
        - "pi+pi-"  : Pion decay channel.
        - "pi0pi0"  : Neutral pion decay channel.
        - "k+k-"    : Kaon decay channel.
        - "klkl"    : Long-lived kaon decay channel.
        - "ksks"    : Short-lived kaon decay channel.
        - "klks"    : Mixed kaon decay channel.
        - "4pi"     : Four pion decay channel.
        - "gg"      : Gluon decay channel.
        - "tau+tau-": Tau-antitau decay channel.
        - "s/s"     : Strange quark-antiquark decay channel.
        - "c/c"     : Charm quark-antiquark decay channel.
        - "b/b"     : Bottom quark-antiquark decay channel.
    Values:
        Each value is a list containing properties of the decay channel in the following order:
        - Mass of particle 1 (GeV)
        - Mass of particle 2 (GeV)
        - PDG code of particle 1
        - PDG code of particle 2
        - Charge of particle 1
        - Charge of particle 2
        - Stability of particle 1 (1 indicates stable)
        - Stability of particle 2 (1 indicates stable)
        Note: The "4pi" channel contains properties for two sets of pion pairs.

LLP_models:
    A dictionary containing different models of Long-Lived Particles (LLPs).
    Keys:
        - "Higgs like scalars": Represents a model of Higgs-like scalar particles.
    Values:
        Each value is a dictionary containing decay channels relevant to the model.
        Currently, it includes:
        - scalar_decays: The dictionary defined above with various decay properties.
"""

scalar_decays = {
    "e+e-" : [0.51099e-3, 0.51099e-3, 11, -11, -1, 1, 1, 1],
    "mu+mu-" : [105.66e-3, 105.66e-3, 13, -13, -1, 1, 1, 1],
    "pi+pi-" : [139.57e-3, 139.57e-3, 211, -211, 1, -1, 1, 1],
    "pi0pi0" : [134.97e-3, 134.97e-3, 111, -111, 0, 0, 1, 1],
    "k+k-" : [493.7e-3, 493.7e-3, 321, -321, 1, -1, 1, 1],
    "klkl" : [497.7e-3, 497.7e-3, 130, -130, 0, 0, 1, 1],
    "ksks" : [497.7e-3, 497.7e-3, 310, -310, 0, 0, 1, 1],
    "klks" : [497.7e-3, 497.7e-3, 130, 310, 0, 0, 1, 1],
    "4pi" : [139.57e-3, 139.57e-3, 211, -211, 1, -1, 0, 0, 134.97e-3, 134.97e-3, 111, -111, 0, 0, 0, 0],
    "gg" : [0, 0, 21, -21, 0, 0, 1, 1],
    "tau+tau-" : [1.77686, 1.77686, 15, -15, -1, 1, 1, 1],
    "s/s" : [0.093, 0.093, 3, -3, -1/3, 1/3, 1, 1],
    "c/c" : [1.29, 1.29, 4, -4, 2/3, -2/3, 1, 1],
    "b/b" : [4.7, 4.7, 5, -5, -1/3, 1/3, 1, 1]
}

LLP_models = {
    "Higgs like scalars" : scalar_decays
}

class Decays:
    def __init__(self, m, momentum, LLP, channel, timing=False):
        """
        Initialize an instance of the class.

        This constructor initializes the object with mass, momentum, and decay product 
        properties. It extracts the relevant decay model parameters based on the given 
        LLP and channel, computes the decay products, and optionally times the operation.

        Parameters:
        m (float): The mass of the primary particle.
        momentum (array-like): The momentum of the primary particle, typically in 3D.
        LLP (str): The type of long-lived particle model to use.
        channel (str): The specific decay channel within the LLP model.
        timing (bool, optional): Whether to time the execution of the decay product computation. Defaults to False.

        Attributes:
        m (float): The mass of the primary particle.
        momentum (array-like): The momentum of the primary particle.
        products (numpy.ndarray): The computed decay products based on the given parameters.

        Prints:
        If `timing` is True, prints the time taken to compute the decay products.
        """

        self.m = m
        self.momentum = momentum

        # Extract decay model parameters based on LLP and channel
        m1, m2, pdg1, pdg2, charge1, charge2, stability1, stability2 = LLP_models[LLP][channel]

        if timing:
            t = time.time()

        # Compute decay products
        self.products = decay_products(self.m, self.momentum, m1, m2, pdg1, pdg2, charge1, charge2, stability1, stability2)

        if timing:
            print(f"Decays products t = {time.time() - t} s")

    def save_decay_products(self, path):
        """
        Save the decay product information to a CSV file.

        This method extracts the decay product data from the `self.products` array
        and saves it into a tab-separated file. The data for each product includes
        momentum components (px, py, pz), energy (E), mass (m), PDG ID (pdg),
        charge, and stability.

        The resulting CSV file will have columns for two products, with each column
        named according to the type of data and product number (e.g., 'px1', 'py1',
        'pz1', 'E1', 'm1', 'pdg1', 'charge1', 'stability1', 'px2', 'py2', etc.).

        Parameters:
        path (str): The directory path where the CSV file will be saved.

        Returns:
        None
        """
        # Define column names for each product type
        columns = ['px', 'py', 'pz', 'E', 'm', 'pdg', 'charge', 'stability']
        num_products = 2  # Number of products to save

        # Initialize a dictionary to hold the data
        decay_dic = {}

        for i in range(1, num_products + 1):
            # Extract data for each product
            start_idx = (i - 1) * len(columns)
            end_idx = i * len(columns)
            product_data = self.products[:, start_idx:end_idx]

            # Assign data to dictionary
            for j, col in enumerate(columns):
                decay_dic[f'{col}{i}'] = product_data[:, j]

        # Create DataFrame and save to CSV
        decay_df = pd.DataFrame(decay_dic)
        file_path = os.path.join(path, "decay_products.dat")
        decay_df.to_csv(file_path, sep="\t", index=False)
