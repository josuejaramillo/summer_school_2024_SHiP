from . import TwoBodyDecay, ThreeBodyDecay, FourBodyDecay
from . import PDG
import numpy as np
import time

def distribute_events(total_events, branching_ratios):
    """
    Distributes the total number of events among different decay channels according to their branching ratios.

    Parameters:
    -----------
    total_events : int
        The total number of events to distribute.
    branching_ratios : list or np.ndarray
        The branching ratios for each decay channel.

    Returns:
    --------
    np.ndarray
        An array containing the number of events allocated to each decay channel.
    """
    # Calculate the initial number of events for each channel based on branching ratios
    events = np.array([total_events * ratio for ratio in branching_ratios])
    
    # Round the event counts to the nearest integer
    rounded_events = np.round(events).astype(int)
    
    # Set the event count to 0 where the branching ratio is 0
    rounded_events = np.where(np.array(branching_ratios) == 0, 0, rounded_events)
    
    # Calculate the difference between the rounded total and the original total event count
    difference = total_events - np.sum(rounded_events)
    
    # Adjust event counts to ensure the total equals total_events
    while difference != 0:
        for i in range(len(branching_ratios)):
            if difference == 0:
                break
            # Adjust only if branching ratio is non-zero
            if branching_ratios[i] > 0:
                # Increase the event count if the current count is below the original calculation
                if difference > 0 and (rounded_events[i] < events[i] or np.sum(rounded_events) < total_events):
                    rounded_events[i] += 1
                    difference -= 1
                # Decrease the event count if the current count is too high
                elif difference < 0 and rounded_events[i] > 0:
                    rounded_events[i] -= 1
                    difference += 1

    return rounded_events

def simulateDecays_rest_frame(mass, PDGdecay, BrRatio, size, Msquared3BodyLLP):
    """
    Simulates the decays of a particle in its rest frame, distributing events among two-body and three-body decays.

    Parameters:
    -----------
    mass : float
        The mass of the decaying particle.
    PDGdecay : list of np.ndarrays
        A list of PDG codes representing the decay products for each decay channel.
    BrRatio : list or np.ndarray
        The branching ratios for each decay channel.
    size : int
        The total number of decay events to simulate.
    Msquared3BodyLLP : list or np.ndarray
        Squared matrix elements for three-body decays.

    Returns:
    --------
    np.ndarray
        A 2D array containing the simulated decay products for all events.
        A 1D array containing the size per channel
    """
    # Helper function to retrieve the mass, charge, and stability of particles based on their PDG codes
    def get_particle_properties(pdg_list):
        masses = [PDG.get_mass(pdg) for pdg in pdg_list]
        charges = [PDG.get_charge(pdg) for pdg in pdg_list]
        stabilities = [PDG.get_stability(pdg) for pdg in pdg_list]
        return masses, charges, stabilities

    # Distribute events among the decay channels according to the branching ratios
    size_per_channel = distribute_events(size, BrRatio)
    results = [None] * len(PDGdecay)

    # Loop through each decay channel and simulate the decays
    for i, decay_modes in enumerate(PDGdecay):
        # Remove any invalid PDG codes (-999 represents an invalid code)
        pdg_list = decay_modes[decay_modes != -999]

        if len(pdg_list) == 2:
            # Two-body decay case
            pdg1, pdg2 = pdg_list
            masses, charges, stabilities = get_particle_properties([pdg1, pdg2])

            
            
            # Simulate decays only if there are events assigned to this channel
            if size_per_channel[i] > 0:
                decay_results = TwoBodyDecay.decay_products(
                    mass, size_per_channel[i], masses[0], masses[1], pdg1, pdg2, charges[0], charges[1], stabilities[0], stabilities[1]
                )
                zeros_array = np.zeros((len(decay_results), 16))
                zeros_array[:, 5] = -999 
                zeros_array[:, 13] = -999 
                # Concatenate the results with an additional array of zeros to match the expected output shape
                results[i] = np.concatenate((decay_results, zeros_array), axis=1)

        elif len(pdg_list) == 3:
            # Three-body decay case
            pdg1, pdg2, pdg3 = pdg_list
            masses, charges, stabilities = get_particle_properties([pdg1, pdg2, pdg3])

            # Parameters needed for three-body decay simulation
            specific_decay_params = (
                pdg1, pdg2, pdg3, masses[0], masses[1], masses[2], 
                charges[0], charges[1], charges[2], 
                stabilities[0], stabilities[1], stabilities[2], 
                Msquared3BodyLLP[i]
            )

            
            # Simulate decays only if there are events assigned to this channel
            if size_per_channel[i] > 0:
                decay_results = ThreeBodyDecay.decay_products(mass, size_per_channel[i], specific_decay_params)
                # Concatenate the results with an additional array of zeros to match the expected output shape
                zeros_array = np.zeros((len(decay_results), 8))
                zeros_array[:, 5] = -999 
                results[i] = np.concatenate((decay_results, zeros_array), axis=1)

        elif len(pdg_list) == 4:
            # Four-body decay case
            pdg1, pdg2, pdg3, pdg4 = pdg_list
            masses, charges, stabilities = get_particle_properties([pdg1, pdg2, pdg3, pdg4])

            # Parameters needed for four-body decay simulation
            specific_decay_params = (
                pdg1, pdg2, pdg3, pdg4, 
                masses[0], masses[1], masses[2], masses[3], 
                charges[0], charges[1], charges[2], charges[3], 
                stabilities[0], stabilities[1], stabilities[2], stabilities[3]
            )

            if size_per_channel[i] > 0:
                results[i] = FourBodyDecay.decay_products(mass, specific_decay_params, size_per_channel[i])


    # Convert results to a numpy array of objects
    results = np.array(results, dtype=object)

    # Combine the results from all decay channels into a single array
    final_array = np.vstack([
        sub_item 
        for item in results if item is not None 
        for sub_item in (item if isinstance(item, list) else [item])
    ])
    return (final_array, size_per_channel)
