# decayProducts.py
import sys
import os
import numpy as np
import time
import multiprocessing

# Ensure Pythia8 is in the Python path
sys.path.insert(0, '/home/name/Downloads/pythia8312/lib')
import pythia8

from . import TwoBodyDecay, ThreeBodyDecay, FourBodyDecay
from . import PDG

def distribute_events(total_events, branching_ratios):
    """
    Distributes the total number of events among different decay channels according to their branching ratios.
    """
    events = np.array([total_events * ratio for ratio in branching_ratios])
    rounded_events = np.round(events).astype(int)
    rounded_events = np.where(np.array(branching_ratios) == 0, 0, rounded_events)
    difference = total_events - np.sum(rounded_events)
    while difference != 0:
        for i in range(len(branching_ratios)):
            if difference == 0:
                break
            if branching_ratios[i] > 0:
                if difference > 0 and (rounded_events[i] < events[i] or np.sum(rounded_events) < total_events):
                    rounded_events[i] += 1
                    difference -= 1
                elif difference < 0 and rounded_events[i] > 0:
                    rounded_events[i] -= 1
                    difference += 1
    return rounded_events

def simulateDecays_rest_frame(mass, PDGdecay, BrRatio, size, Msquared3BodyLLP):
    """
    Simulates the decays of a particle in its rest frame, distributing events among different decay channels.
    """
    def get_particle_properties(pdg_list):
        masses = [PDG.get_mass(pdg) for pdg in pdg_list]
        charges = [PDG.get_charge(pdg) for pdg in pdg_list]
        stabilities = [PDG.get_stability(pdg) for pdg in pdg_list]
        return masses, charges, stabilities

    # Distribute events among the decay channels
    size_per_channel = distribute_events(size, BrRatio)
    all_decay_events = []

    # Loop through each decay channel and simulate the decays
    for i, decay_modes in enumerate(PDGdecay):
        pdg_list = decay_modes[decay_modes != -999]

        if size_per_channel[i] == 0:
            continue  # Skip if no events are assigned to this decay channel

        if len(pdg_list) == 2:
            # Two-body decay
            pdg1, pdg2 = pdg_list
            masses, charges, stabilities = get_particle_properties([pdg1, pdg2])
            decay_results = TwoBodyDecay.decay_products(
                mass, size_per_channel[i], masses[0], masses[1], pdg1, pdg2,
                charges[0], charges[1], stabilities[0], stabilities[1]
            )
        elif len(pdg_list) == 3:
            # Three-body decay
            pdg1, pdg2, pdg3 = pdg_list
            masses, charges, stabilities = get_particle_properties([pdg1, pdg2, pdg3])
            specific_decay_params = (
                pdg1, pdg2, pdg3, masses[0], masses[1], masses[2],
                charges[0], charges[1], charges[2],
                stabilities[0], stabilities[1], stabilities[2],
                Msquared3BodyLLP[i]
            )
            decay_results = ThreeBodyDecay.decay_products(
                mass, size_per_channel[i], specific_decay_params
            )
        elif len(pdg_list) == 4:
            # Four-body decay
            pdg1, pdg2, pdg3, pdg4 = pdg_list
            masses, charges, stabilities = get_particle_properties([pdg1, pdg2, pdg3, pdg4])
            specific_decay_params = (
                pdg1, pdg2, pdg3, pdg4, masses[0], masses[1], masses[2], masses[3],
                charges[0], charges[1], charges[2], charges[3],
                stabilities[0], stabilities[1], stabilities[2], stabilities[3]
            )
            decay_results = FourBodyDecay.decay_products(
                mass, specific_decay_params, size_per_channel[i]
            )
        else:
            continue  # Skip invalid decay channels

        # Append decay results to the list of all decay events
        all_decay_events.extend(decay_results)

    # Process all decay events through Pythia in parallel
    if all_decay_events:
        processed_results = process_with_pythia_parallel(all_decay_events, mass)
    else:
        processed_results = np.array([])

    return (processed_results, size_per_channel)

def process_with_pythia_parallel(decay_events, mass):
    """
    Processes decay events through Pythia in parallel using multiprocessing.
    """
    import multiprocessing
    num_processes = multiprocessing.cpu_count()
    chunks = np.array_split(decay_events, num_processes)

    # Prepare arguments as a list of tuples
    args = [(chunk, mass) for chunk in chunks]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_events_with_pythia, args)

    # Flatten the list of lists into a single list
    processed_events = [event for sublist in results for event in sublist]

    # Compute the global maximum length for padding
    if processed_events:
        max_length = max(len(evt) for evt in processed_events)
        # Pad all events to the global maximum length
        processed_events_padded = np.array([
            evt + [0]*(max_length - len(evt)) for evt in processed_events
        ])
    else:
        processed_events_padded = np.array([])

    return processed_events_padded


def process_events_with_pythia(decay_events_chunk, mass):
    """
    Processes a chunk of decay events through Pythia.
    """
    import sys
    sys.path.insert(0, '/home/name/Downloads/pythia8312/lib')
    import pythia8

    # Initialize Pythia
    pythia = pythia8.Pythia()
    # Suppress initialization output
    pythia.readString("Print:quiet = on")
    pythia.readString("Print:banner = off")
    # Disable process-level generation since we're providing events
    pythia.readString("ProcessLevel:all = off")
    # Enable hadronization and decay processes
    pythia.readString("HadronLevel:all = on")
    # Keep certain particles stable
    stable_particles = [13, -13, 211, -211, 321, -321, 130]
    for pid in stable_particles:
        pythia.readString(f"{pid}:mayDecay = off")
    mother_id = 25  # PDG ID for the Higgs boson
    pythia.init()

    processed_events = []

    # Initialize color tag counter
    color_tag = 501  # Starting value for color tags

    for decay in decay_events_chunk:
        pythia.event.reset()
        num_particles = len(decay) // 8

        # Set mother particle at rest
        mother_px, mother_py, mother_pz = 0.0, 0.0, 0.0
        mother_e = mass
        mother_m = mass

        # Append the mother particle (color singlet)
        pythia.event.append(
            mother_id, 23, 0, 0, 2, num_particles + 1,
            0, 0,  # col, acol
            mother_px, mother_py, mother_pz, mother_e, mother_m
        )

        # Check decay products PDG IDs
        decay_pdgs = [int(decay[i * 8 + 5]) for i in range(num_particles) if int(decay[i * 8 + 5]) != -999]

        # Handle special cases (e.g., gg decay)
        if decay_pdgs == [21, 21]:
            # Assign colors to form a color singlet
            col1 = color_tag
            acol1 = color_tag + 1
            color_tag += 2
            # First gluon
            idx = 0
            px, py, pz, e, m, pdgId = decay[idx:idx+6]
            pdgId = int(pdgId)
            pythia.event.append(
                pdgId, 1, 1, 1, 0, 0,
                col1, acol1,
                px, py, pz, e, m
            )
            # Second gluon
            idx = 8
            px, py, pz, e, m, pdgId = decay[idx:idx+6]
            pdgId = int(pdgId)
            pythia.event.append(
                pdgId, 1, 1, 1, 0, 0,
                acol1, col1,  # Swap colors
                px, py, pz, e, m
            )
        else:
            for i_particle in range(num_particles):
                idx = i_particle * 8
                px = decay[idx]
                py = decay[idx + 1]
                pz = decay[idx + 2]
                e = decay[idx + 3]
                m = decay[idx + 4]
                pdgId = int(decay[idx + 5])

                if pdgId == -999:
                    continue  # Skip placeholder entries

                # Initialize color and anticolor
                col = 0
                acol = 0

                # Assign colors based on PDG ID
                if abs(pdgId) in [1, 2, 3, 4, 5, 6]:  # Quarks
                    if pdgId > 0:
                        col = color_tag
                        acol = 0
                    else:
                        col = 0
                        acol = color_tag
                    color_tag += 1  # Increment color tag for uniqueness
                elif abs(pdgId) == 21:  # Gluon
                    # Handle gluons appropriately
                    col = color_tag
                    acol = color_tag + 1
                    color_tag += 2
                else:
                    col = 0
                    acol = 0

                # Append particle to Pythia event
                pythia.event.append(
                    pdgId, 1, 1, 1, 0, 0,
                    col, acol,
                    px, py, pz, e, m
                )

        # Verify momentum conservation
        total_px = sum([decay[i * 8] for i in range(num_particles)])
        total_py = sum([decay[i * 8 + 1] for i in range(num_particles)])
        total_pz = sum([decay[i * 8 + 2] for i in range(num_particles)])

        if not np.allclose([total_px, total_py, total_pz], [0.0, 0.0, 0.0], atol=1e-6):
            print("Decay products do not conserve momentum.")
            continue  # Skip this event

        # Process the event through Pythia
        if not pythia.forceHadronLevel():
            print("Error processing event")
            pythia.event.list()
            continue  # Skip this event

        # Extract final state particles
        final_state_particles = []
        for p in pythia.event:
            if p.isFinal():
                particle_data = [
                    p.px(), p.py(), p.pz(), p.e(), p.m(), p.id()
                ]
                final_state_particles.extend(particle_data)

        processed_events.append(final_state_particles)

    return processed_events

