import numpy as np
import numba as nb
from src.decays_functions import decay_products, LLP_BrRatios
import time
import pandas as pd
import os

class Decays:
    def __init__(self, m, momentum, decay_channels, BrRatios, timing=False):
        """
        Initialize an instance of the class.

        This constructor initializes the object with mass, momentum, and decay product 
        properties. It extracts the relevant decay model parameters based on the given 
        LLP and channel, computes the decay products, and optionally times the operation.

        Parameters:
        m (float): The mass of the primary particle.
        momentum (array-like): The momentum of the primary particle, typically in 3D.
        LLP (str): The type of long-lived particle model to use.
        decay_channels (array): decay channels for the selected LLP. 
        BrRatios (array): Branching ratios for the specific LLP mass.
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
        if timing:
            t = time.time()

        # Compute BrRatio for specific mass
        BrRatio = LLP_BrRatios(self.m, BrRatios)
        BrRatio[-6] = 0

        # Compute decay products
        self.products = decay_products(self.m, self.momentum, BrRatio, decay_channels)

        #Check momentum conservation
        # momentum_3_mother = self.momentum[:,0:3]
        # momentum_3_daugthers = self.products[:,0:3] + self.products[:,8:11]
        # array_bool = abs(momentum_3_mother - momentum_3_daugthers) < 1e-13
        # print(np.any(array_bool == False))

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
