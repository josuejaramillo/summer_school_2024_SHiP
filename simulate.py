# main.py
import matplotlib.pyplot as plt
import sys
import os
from funcs import initLLP, decayProducts, boost, kinematics, mergeResults
import time
import numpy as np

# Define the number of events to simulate
try:
    resampleSize = int(input("\nEnter the number of events to simulate: "))
    if resampleSize <= 0:
        raise ValueError("The number of events must be a positive integer.")
except ValueError as e:
    raise ValueError(f"Invalid input for the number of events: {e}")
nEvents = resampleSize * 10  # Integer multiplication for clarity

# Define N_pot (number of protons on target)
N_pot = 6e20

def select_particle():
    main_folder = "./Distributions"
    folders = np.array(os.listdir(main_folder))
                
    print("\nParticle Selector\n")
    for i, folder in enumerate(folders):
        print(f"{i + 1}. {folder}")

    try:
        selected_particle = int(input("Select particle: ")) - 1
        particle_distr_folder = folders[selected_particle]
    except (IndexError, ValueError):
        raise ValueError("Invalid selection. Please select a valid particle.")

    particle_path = os.path.join(main_folder, particle_distr_folder)
    LLP_name = particle_distr_folder.replace("_", " ")

    return {'particle_path': particle_path, 'LLP_name': LLP_name}

def prompt_uncertainty():
    print("\nWhich variation of the dark photon flux within the uncertainty to select?")
    print("1. lower")
    print("2. central")
    print("3. upper")

    try:
        selected_uncertainty = int(input("Select uncertainty level (1-3): "))
        if selected_uncertainty == 1:
            uncertainty = "lower"
        elif selected_uncertainty == 2:
            uncertainty = "central"
        elif selected_uncertainty == 3:
            uncertainty = "upper"
        else:
            raise ValueError("Invalid selection.")
    except ValueError as e:
        raise ValueError(f"Invalid input for uncertainty level: {e}")
    return uncertainty

def prompt_mixing_pattern():
    try:
        mixing_input = input("\nEnter xi_e, xi_mu, xi_tau: (Ue2, Umu2, Utau2) = U2(xi_e,xi_mu,xi_tau), summing to 1, separated by spaces: ").strip().split()
        if len(mixing_input) != 3:
            raise ValueError("Please enter exactly three numerical values separated by spaces.")
        
        Ue2, Umu2, Utau2 = map(float, mixing_input)

        MixingPatternArray = np.array([Ue2, Umu2, Utau2])
        sumMixingPattern = Ue2 + Umu2 + Utau2
        if sumMixingPattern != 1:
            Ue2 /= sumMixingPattern
            Umu2 /= sumMixingPattern
            Utau2 /= sumMixingPattern
            MixingPatternArray = np.array([Ue2, Umu2, Utau2])

        return MixingPatternArray

    except ValueError as e:
        raise ValueError(f"Invalid input. Please enter three numerical values separated by spaces: {e}")

def prompt_masses_and_c_taus():
    """
    Prompts the user for masses of LLPs and their lifetimes (c*tau).
    If `ifSameLifetimes` is True, the same list of lifetimes will be used for all masses.
    Otherwise, lifetimes are input separately for each mass.
    """
    try:
        masses_input = input("\nLLP masses in GeV (separated by spaces): ").split()
        # Automatically remove trailing dots from each mass
        masses = [float(m.rstrip('.')) for m in masses_input]

        # Hardcoded definition for using the same lifetimes for all masses
        ifSameLifetimes = True

        c_taus_list = []
        if ifSameLifetimes:
            # Single lifetime list for all masses
            c_taus_input = input("Enter lifetimes c*tau in m for all masses (separated by spaces or commas): ")
            c_taus = [float(tau) for tau in c_taus_input.replace(',', ' ').split()]
            c_taus_list = [c_taus] * len(masses)
        else:
            # Separate lifetime lists for each mass
            for mass in masses:
                c_taus_input = input(f"Life times c*tau for mass {mass} (separated by spaces or commas): ")
                c_taus = [float(tau) for tau in c_taus_input.replace(',', ' ').split()]
                c_taus_list.append(c_taus)
        
        return masses, c_taus_list
    except ValueError:
        raise ValueError("Invalid input for masses or c*taus. Please enter numerical values.")

def prompt_decay_channels(decayChannels):
    print("\nSelect the decay modes:")
    print("0. All")
    for i, channel in enumerate(decayChannels):
        print(f"{i + 1}. {channel}")
    
    user_input = input("Enter the numbers of the decay channels to select (separated by spaces): ")
    try:
        selected_indices = [int(x) for x in user_input.strip().split()]
        if not selected_indices:
            raise ValueError("No selection made.")
        if 0 in selected_indices:
            return list(range(len(decayChannels)))
        else:
            selected_indices = [x - 1 for x in selected_indices]
            for idx in selected_indices:
                if idx < 0 or idx >= len(decayChannels):
                    raise ValueError(f"Invalid index {idx + 1}.")
            return selected_indices
    except ValueError as e:
        raise ValueError(f"Invalid input for decay channel selection: {e}")

# Initialize the LLP parameters
particle_selection = select_particle()

# Initialize uncertainty to None
uncertainty = None

# Only prompt for mixing pattern if the selected particle is HNL
if particle_selection['LLP_name'] == "HNL":
    mixing_pattern = prompt_mixing_pattern()
else:
    mixing_pattern = None

# If the selected particle is "Dark-photons", prompt for uncertainty
if particle_selection['LLP_name'] == "Dark-photons":
    uncertainty = prompt_uncertainty()

# Create LLP instance (mass-independent)
LLP = initLLP.LLP(mass=None, particle_selection=particle_selection, mixing_pattern=mixing_pattern, uncertainty=uncertainty)

# Prompt for decay channels
selected_decay_indices = prompt_decay_channels(LLP.decayChannels)

# Prompt for masses and c_taus
masses, c_taus_list = prompt_masses_and_c_taus()

timing = False

for mass, c_taus in zip(masses, c_taus_list):
    print(f"\nProcessing mass {mass}")

    LLP.mass = mass
    LLP.compute_mass_dependent_properties()

    br_visible_val = sum(LLP.BrRatios_distr[idx] for idx in selected_decay_indices)
    
    if br_visible_val == 0:
        print("No decay events for the given selected decay modes for the given mass. Skipping...")
        continue

    for c_tau in c_taus:
        print(f"  Processing c_tau {c_tau}")

        coupling_squared = LLP.c_tau_int / c_tau

        if particle_selection['LLP_name'] != "Scalar-quartic":
            N_LLP_tot = N_pot * LLP.Yield * coupling_squared
        else:
            # Br ratio of h -> SS in BC5 model
            Br_h_SS = 0.01
            N_LLP_tot = N_pot * LLP.Yield * Br_h_SS

        print(f"    Coupling squared: {coupling_squared}")
        print(f"    Total number of LLPs produced: {N_LLP_tot}")

        LLP.set_c_tau(c_tau)

        t = time.time()

        kinematics_samples = kinematics.Grids(
            LLP.Distr, LLP.Energy_distr, nEvents, LLP.mass, LLP.c_tau_input
        )

        kinematics_samples.interpolate(timing)
        kinematics_samples.resample(resampleSize, timing)
        epsilon_polar = kinematics_samples.epsilon_polar
        kinematics_samples.true_samples(timing)
        momentum = kinematics_samples.get_momentum()

        finalEvents = len(momentum)
        epsilon_azimuthal = finalEvents / resampleSize

        unBoostedProducts, size_per_channel = decayProducts.simulateDecays_rest_frame(
            LLP.mass, LLP.PDGs, LLP.BrRatios_distr, finalEvents, LLP.Matrix_elements,
            selected_decay_indices, br_visible_val
        )
        boostedProducts = boost.tab_boosted_decay_products(
            LLP.mass, momentum, unBoostedProducts
        )

        print("    Total time for this iteration: ", time.time() - t)

        motherParticleResults = kinematics_samples.get_kinematics()
        decayProductsResults = boostedProducts

        P_decay_data = motherParticleResults[:, 6]
        P_decay_averaged = np.mean(P_decay_data)

        N_ev_tot = N_LLP_tot * epsilon_polar * epsilon_azimuthal * P_decay_averaged * br_visible_val
        
        print(f"    Total number of decay events in SHiP: {N_ev_tot}")
        
        print(f"    Exporting results...")
        
        t = time.time()

        mergeResults.save(
            motherParticleResults, decayProductsResults, LLP.LLP_name, LLP.mass,
            LLP.MixingPatternArray, LLP.c_tau_input, LLP.decayChannels, size_per_channel,
            finalEvents, epsilon_polar, epsilon_azimuthal, N_LLP_tot, coupling_squared,
            P_decay_averaged, N_ev_tot, br_visible_val, selected_decay_indices
        )
        
        print("    Total time spent on exporting: ", time.time() - t)
        
        print(f"    Done")

