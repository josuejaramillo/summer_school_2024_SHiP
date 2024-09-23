import pandas as pd
import numpy as np

def save(motherParticleResults, decayProductsResults, LLP_name, mass, MixingPatternArray, c_tau, decayChannels, size_per_channel):
    # Format od resutls
    # px_mother, py_mother, pz_mother, energy_mother, m_mother, PDG_mother, P_decay_mother, x_mother, y_mother, z_mother, px1, py1, pz1, e1, MASS1, pdg1, charge1, stability1, px2, py2, pz2, e2, MASS2, pdg2, charge2, stability2, px3, py3, pz3, e3, MASS3, pdg3, charge3, stability3
    results = np.concatenate((motherParticleResults, decayProductsResults), axis=1)

    if any(MixingPatternArray) == None:
        outputfileName = f'./outputs/{LLP_name}_{str(mass )}_{c_tau}_decayProducts.dat'
    else:
        outputfileName = f'./outputs/{LLP_name}_{str(mass )}_{round(MixingPatternArray[0], 2)}_{round(MixingPatternArray[1],2)}_{round(MixingPatternArray[2],2)}_{c_tau}_decayProducts.dat'

    with open(outputfileName, 'w') as f:
        start_row = 0  # Initialize the starting row index
        for i, channel in enumerate(decayChannels):
            channel_size = size_per_channel[i]

            # Skip channels with size 0
            if channel_size == 0:
                continue

            end_row = start_row + channel_size
            data = results[start_row:end_row, :]  # Extract the relevant rows
            
            # Write the header
            f.write(f"#<process={channel}; sample_points={channel_size}>\n\n")
            
            # Write the data
            for row in data:
                row_str = ' '.join(map(str, row))
                f.write(f"{row_str}\n")
            
            # Add a blank line between channels
            f.write("\n")
            
            # Update the starting row index for the next channel
            start_row = end_row
