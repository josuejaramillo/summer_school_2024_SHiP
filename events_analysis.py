import os
import numpy as np
import re
import sys
import matplotlib.pyplot as plt
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def parse_filenames(directory):
    """
    Parses filenames in the given directory and its subdirectories to extract LLP names, masses, lifetimes, and mixing patterns.
    Returns a dictionary llp_dict[llp_name][(mass, lifetime)][mixing_patterns] = filepath
    """
    llp_dict = {}  # LLP_name: { (mass, lifetime): { mixing_patterns: filepath } }

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('_decayProducts.dat'):
                filepath = os.path.relpath(os.path.join(root, filename), directory)
                # Extract LLP_name from the parent directory of 'eventData'
                rel_path = os.path.relpath(root, directory)
                llp_name = os.path.basename(os.path.dirname(rel_path))
                # Parse filename to extract mass, lifetime, and mixing patterns
                base_name = filename[:-len('_decayProducts.dat')]
                tokens = base_name.split('_')
                # Identify indices of tokens that can be converted to float
                float_token_indices = []
                for i, token in enumerate(tokens):
                    try:
                        float(token)
                        float_token_indices.append(i)
                    except ValueError:
                        continue
                if len(float_token_indices) >= 2:
                    mass_index = float_token_indices[0]
                    # mass and lifetime
                    mass = tokens[mass_index]
                    lifetime = tokens[mass_index + 1]
                    # Mixing patterns, if any
                    mixing_pattern_tokens = tokens[mass_index + 2:]
                    mixing_patterns = []
                    for token in mixing_pattern_tokens:
                        try:
                            mixing_patterns.append(float(token))
                        except ValueError:
                            print(f"Invalid mixing pattern token: {token} in filename {filename}")
                    # Convert mass and lifetime to float
                    mass = float(mass)
                    lifetime = float(lifetime)
                    # Store in llp_dict
                    if llp_name not in llp_dict:
                        llp_dict[llp_name] = {}
                    mass_lifetime = (mass, lifetime)
                    if mass_lifetime not in llp_dict[llp_name]:
                        llp_dict[llp_name][mass_lifetime] = {}
                    mixing_patterns_tuple = tuple(mixing_patterns) if mixing_patterns else None
                    llp_dict[llp_name][mass_lifetime][mixing_patterns_tuple] = filepath
                else:
                    print(f"Filename {filename} does not have enough float tokens to extract mass and lifetime.")
            else:
                continue  # Skip files not ending with '_decayProducts.dat'
    return llp_dict

def user_selection(llp_dict):
    """
    Allows the user to select an LLP, mass-lifetime combination, and mixing patterns.
    Returns the selected filepath and the selected LLP name.
    """
    print("Available LLPs:")
    llp_names_list = sorted(llp_dict.keys())
    for i, llp_name in enumerate(llp_names_list):
        print(f"{i+1}. {llp_name}")

    # Ask user to choose an LLP
    while True:
        try:
            choice = int(input("Choose an LLP by typing the number: "))
            if 1 <= choice <= len(llp_names_list):
                break
            else:
                print(f"Please enter a number between 1 and {len(llp_names_list)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    selected_llp = llp_names_list[choice - 1]
    print(f"Selected LLP: {selected_llp}")

    # Get available mass-lifetime combinations
    mass_lifetime_list = sorted(llp_dict[selected_llp].keys())
    print(f"Available mass-lifetime combinations for {selected_llp}:")
    for i, (mass, lifetime) in enumerate(mass_lifetime_list):
        print(f"{i+1}. mass={mass:.6e} GeV, lifetime={lifetime:.6e}")

    # Ask user to choose a mass-lifetime
    while True:
        try:
            mass_lifetime_choice = int(input("Choose a mass-lifetime combination by typing the number: "))
            if 1 <= mass_lifetime_choice <= len(mass_lifetime_list):
                break
            else:
                print(f"Please enter a number between 1 and {len(mass_lifetime_list)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    selected_mass_lifetime = mass_lifetime_list[mass_lifetime_choice - 1]
    selected_mass, selected_lifetime = selected_mass_lifetime
    print(f"Selected mass: {selected_mass:.6e} GeV, lifetime: {selected_lifetime:.6e}")

    # Get mixing patterns
    mixing_patterns_dict = llp_dict[selected_llp][selected_mass_lifetime]
    mixing_patterns_list = sorted(mixing_patterns_dict.keys())
    if any(mixing_patterns_list):
        print(f"Available mixing patterns for {selected_llp} with mass {selected_mass:.6e} GeV and lifetime {selected_lifetime:.6e}:")
        for i, mixing_patterns in enumerate(mixing_patterns_list):
            print(f"{i+1}. {mixing_patterns}")
        # Ask user to choose a mixing pattern
        while True:
            try:
                mixing_choice = int(input("Choose a mixing pattern by typing the number: "))
                if 1 <= mixing_choice <= len(mixing_patterns_list):
                    break
                else:
                    print(f"Please enter a number between 1 and {len(mixing_patterns_list)}.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        selected_mixing_patterns = mixing_patterns_list[mixing_choice - 1]
        print(f"Selected mixing pattern: {selected_mixing_patterns}")
    else:
        selected_mixing_patterns = None

    # Find the file matching the selection
    selected_filepath = mixing_patterns_dict[selected_mixing_patterns]
    print(f"Selected file: {selected_filepath}")

    return selected_filepath, selected_llp

def read_file(filepath):
    """
    Reads the file at the given filepath.
    Returns finalEvents, epsilon_polar, epsilon_azimuthal, br_visible_val, channels
    """
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        # Expected format:
        # Sampled {finalEvents} events inside SHiP volume. Total number of produced LLPs: {N_LLP_tot}. Polar acceptance: {epsilon_polar}. Azimuthal acceptance: {epsilon_azimuthal}. Averaged decay probability: {P_decay_averaged}. Visible Br Ratio: {br_visible_val:.6e}. Total number of events: {N_ev_tot}
        pattern = (
            r'Sampled\s+(?P<finalEvents>[\d\.\+\-eE]+)\s+events inside SHiP volume\. '
            r'Total number of produced LLPs:\s+(?P<N_LLP_tot>[\d\.\+\-eE]+)\. '
            r'Polar acceptance:\s+(?P<epsilon_polar>[\d\.\+\-eE]+)\. '
            r'Azimuthal acceptance:\s+(?P<epsilon_azimuthal>[\d\.\+\-eE]+)\. '
            r'Averaged decay probability:\s+(?P<P_decay_averaged>[\d\.\+\-eE]+)\. '
            r'Visible Br Ratio:\s+(?P<br_visible_val>[\d\.\+\-eE]+)\. '
            r'Total number of events:\s+(?P<N_ev_tot>[\d\.\+\-eE]+)'
        )
        match = re.match(pattern, first_line)
        if match:
            finalEvents = float(match.group('finalEvents'))
            N_LLP_tot = float(match.group('N_LLP_tot'))
            epsilon_polar = float(match.group('epsilon_polar'))
            epsilon_azimuthal = float(match.group('epsilon_azimuthal'))
            P_decay_averaged = float(match.group('P_decay_averaged'))
            br_visible_val = float(match.group('br_visible_val'))
            N_ev_tot = float(match.group('N_ev_tot'))
            print(f"finalEvents: {finalEvents}, N_LLP_tot: {N_LLP_tot}, epsilon_polar: {epsilon_polar}, "
                  f"epsilon_azimuthal: {epsilon_azimuthal}, P_decay_averaged: {P_decay_averaged}, "
                  f"br_visible_val: {br_visible_val}, N_ev_tot: {N_ev_tot}")
        else:
            print("Error: First line does not match expected format.")
            sys.exit(1)

        # Skip any empty lines
        while True:
            line = f.readline()
            if not line:
                break
            if line.strip() != '':
                break
        # Now process the rest of the file
        # Extract channels and sample_points
        channels = {}
        current_channel = None
        current_channel_size = 0
        current_data = []
        # If the line we just read is a channel header, process it
        if line.strip().startswith('#<process='):
            match = re.match(
                r'#<process=(?P<channel>.*?);\s*sample_points=(?P<channel_size>[\d\.\+\-eE]+)>', line.strip())
            if match:
                current_channel = match.group('channel')
                current_channel_size = int(float(match.group('channel_size')))
                current_data = []
            else:
                print(f"Error: Could not parse channel line: {line}")
        else:
            print("Error: Expected channel header after first line.")
            sys.exit(1)

        # Continue reading the file
        for line in f:
            line = line.strip()
            if line.startswith('#<process='):
                # This is a new channel
                match = re.match(
                    r'#<process=(?P<channel>.*?);\s*sample_points=(?P<channel_size>[\d\.\+\-eE]+)>', line)
                if match:
                    if current_channel is not None:
                        # Save the data of the previous channel
                        channels[current_channel] = {
                            'size': current_channel_size, 'data': current_data}
                    current_channel = match.group('channel')
                    current_channel_size = int(float(match.group('channel_size')))
                    current_data = []
                else:
                    print(f"Error: Could not parse channel line: {line}")
            elif line == '':
                # Empty line, skip
                continue
            else:
                # This is data
                current_data.append(line)

        # After the loop, save the last channel's data
        if current_channel is not None:
            channels[current_channel] = {
                'size': current_channel_size, 'data': current_data}

    return finalEvents, epsilon_polar, epsilon_azimuthal, br_visible_val, channels

def plot_channels(channels, finalEvents, output_path):
    """
    Plots histogram for channels.
    """
    channel_names = list(channels.keys())
    channel_sizes = [channels[ch]['size'] for ch in channel_names]

    plt.figure()
    plt.bar(channel_names, channel_sizes, color='skyblue', edgecolor='black')
    plt.title(f"$N_{{\\mathrm{{entries}}}}$ = {finalEvents:.0f}")
    plt.xlabel("Channel")
    plt.ylabel("Number of events")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "channels.pdf"), bbox_inches='tight')
    plt.close()

def extract_quantities(channels):
    """
    Extracts required quantities from the data lines.
    Returns a dictionary with the extracted quantities.
    """
    # Initialize lists for quantities
    quantities = {
        'px_mother': [],
        'py_mother': [],
        'pz_mother': [],
        'energy_mother': [],
        'm_mother': [],
        'PDG_mother': [],
        'P_decay_mother': [],
        'x_mother': [],
        'y_mother': [],
        'z_mother': [],
        'decay_products_counts': [],  # total decay products per event
        'charged_decay_products_counts': [],  # charged decay products per event
        'decay_products_per_event_counts': {},  # counts of dedicated products per event
    }

    # Initialize per-event counts for dedicated decay products
    decay_products_types = ['e', 'mu', 'pi±', 'K±', 'K_L', 'gamma', 'p', 'n', 'nu', 'other']
    quantities['decay_products_per_event_counts'] = {ptype: [] for ptype in decay_products_types}

    for channel, channel_data in channels.items():
        data_lines = channel_data['data']
        for data_line in data_lines:
            # Split the data line into numbers
            numbers = list(map(float, data_line.strip().split()))
            # First 10 numbers are px_mother, py_mother, pz_mother, energy_mother, m_mother, PDG_mother, P_decay_mother, x_mother, y_mother, z_mother
            if len(numbers) < 10:
                print(f"Error: Data line has less than 10 numbers: {data_line}")
                continue
            px_mother = numbers[0]
            py_mother = numbers[1]
            pz_mother = numbers[2]
            energy_mother = numbers[3]
            m_mother = numbers[4]
            PDG_mother = numbers[5]
            P_decay_mother = numbers[6]
            x_mother = numbers[7]
            y_mother = numbers[8]
            z_mother = numbers[9]

            quantities['px_mother'].append(px_mother)
            quantities['py_mother'].append(py_mother)
            quantities['pz_mother'].append(pz_mother)
            quantities['energy_mother'].append(energy_mother)
            quantities['m_mother'].append(m_mother)
            quantities['PDG_mother'].append(PDG_mother)
            quantities['P_decay_mother'].append(P_decay_mother)
            quantities['x_mother'].append(x_mother)
            quantities['y_mother'].append(y_mother)
            quantities['z_mother'].append(z_mother)

            # Now, count the number of decay products
            decay_products_count = 0
            charged_decay_products_count = 0

            # Initialize per-event Counter for dedicated products
            event_decay_products_counter = Counter()
            # The rest of the numbers are grouped into sets of 6: px, py, pz, e, mass, pdg
            decay_products = numbers[10:]
            for i in range(0, len(decay_products), 6):
                # Each decay product has 6 numbers
                if i+6 > len(decay_products):
                    print(f"Error: Decay product data incomplete in line: {data_line}")
                    break
                px = decay_products[i]
                py = decay_products[i+1]
                pz = decay_products[i+2]
                e = decay_products[i+3]
                mass = decay_products[i+4]
                pdg = decay_products[i+5]
                if pdg == -999.0:
                    # Do not count
                    continue
                else:
                    decay_products_count += 1
                    if pdg not in [22., 130., 310., 2112.]:
                        # These are neutral particles, so count as charged
                        charged_decay_products_count += 1
                    # Count dedicated products
                    pdg_int = int(pdg)
                    if pdg_int in [11, -11]:
                        event_decay_products_counter['e'] += 1
                    elif pdg_int in [13, -13]:
                        event_decay_products_counter['mu'] += 1
                    elif pdg_int in [211, -211]:
                        event_decay_products_counter['pi±'] += 1
                    elif pdg_int in [321, -321]:
                        event_decay_products_counter['K±'] += 1
                    elif pdg_int == 130:
                        event_decay_products_counter['K_L'] += 1
                    elif pdg_int == 22:
                        event_decay_products_counter['gamma'] += 1
                    elif pdg_int == 2212:
                        event_decay_products_counter['p'] += 1
                    elif pdg_int == 2112:
                        event_decay_products_counter['n'] += 1
                    elif pdg_int in [12, -12, 14, -14, 16, -16]:
                        event_decay_products_counter['nu'] +=1
                    else:
                        event_decay_products_counter['other'] +=1

            quantities['decay_products_counts'].append(decay_products_count)
            quantities['charged_decay_products_counts'].append(charged_decay_products_count)

            # For each particle type, append the count in this event
            for ptype in decay_products_types:
                count = event_decay_products_counter.get(ptype, 0)
                quantities['decay_products_per_event_counts'][ptype].append(count)

    return quantities

def plot_decay_volume(ax):
    """
    Plots the decay volume geometry as a trapezoidal prism on the given Axes3D object.
    The decay region extends in z from 32 m to 82 m.
    In x, its width is z-dependent: from -(0.02*(82-z) + 2/25*(z-32))/2 to +(0.02*(82-z) + 2/25*(z-32))/2.
    In y, from -(0.054*(82-z) + 0.124*(z-32))/2 to +(0.054*(82-z) + 0.124*(z-32))/2.
    The decay volume is colored light gray with gray edges.
    """
    # Define z boundaries
    z_min = 32  # in meters
    z_max = 82  # in meters

    # Calculate x and y boundaries at z_min and z_max
    # At z_min = 32
    x_min_zmin = -(0.02*(82 - z_min) + (2/25)*(z_min - 32))/2
    x_max_zmin = (0.02*(82 - z_min) + (2/25)*(z_min - 32))/2
    y_min_zmin = -(0.054*(82 - z_min) + 0.124*(z_min - 32))/2
    y_max_zmin = (0.054*(82 - z_min) + 0.124*(z_min - 32))/2

    # At z_max = 82
    x_min_zmax = -(0.02*(82 - z_max) + (2/25)*(z_max - 32))/2
    x_max_zmax = (0.02*(82 - z_max) + (2/25)*(z_max - 32))/2
    y_min_zmax = -(0.054*(82 - z_max) + 0.124*(z_max - 32))/2
    y_max_zmax = (0.054*(82 - z_max) + 0.124*(z_max - 32))/2

    # Define the 8 vertices of the trapezoidal prism
    vertices = [
        [x_min_zmin, y_min_zmin, z_min],
        [x_max_zmin, y_min_zmin, z_min],
        [x_max_zmin, y_max_zmin, z_min],
        [x_min_zmin, y_max_zmin, z_min],
        [x_min_zmax, y_min_zmax, z_max],
        [x_max_zmax, y_min_zmax, z_max],
        [x_max_zmax, y_max_zmax, z_max],
        [x_min_zmax, y_max_zmax, z_max]
    ]

    # Define the 12 edges of the prism
    edges = [
        [vertices[0], vertices[1]],
        [vertices[1], vertices[2]],
        [vertices[2], vertices[3]],
        [vertices[3], vertices[0]],
        [vertices[4], vertices[5]],
        [vertices[5], vertices[6]],
        [vertices[6], vertices[7]],
        [vertices[7], vertices[4]],
        [vertices[0], vertices[4]],
        [vertices[1], vertices[5]],
        [vertices[2], vertices[6]],
        [vertices[3], vertices[7]]
    ]

    # Plot the edges
    for edge in edges:
        xs, ys, zs = zip(*edge)
        ax.plot(xs, ys, zs, color='gray', linewidth=1)

    # Define the 6 faces of the prism
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
        [vertices[3], vertices[0], vertices[4], vertices[7]]   # Left face
    ]

    # Create a Poly3DCollection for the faces
    face_collection = Poly3DCollection(faces, linewidths=0.5, edgecolors='gray', alpha=0.3)
    face_collection.set_facecolor('lightgray')  # Light gray with transparency
    ax.add_collection3d(face_collection)

def plot_histograms(quantities, output_path):
    """
    Plots the required histograms and saves them in the output_path directory.
    All histograms are normalized to represent probability densities.
    """
    import os
    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Convert relevant quantities to numpy arrays for easier handling
    energy_mother = np.array(quantities['energy_mother'])
    P_decay_mother = np.array(quantities['P_decay_mother'])
    z_mother = np.array(quantities['z_mother'])
    x_mother = np.array(quantities['x_mother'])
    y_mother = np.array(quantities['y_mother'])

    # Energy of mother particle (unweighted)
    plt.figure()
    plt.hist(energy_mother, bins=50, color='skyblue', edgecolor='black', density=True)
    plt.yscale('log')  # Preserving original y-axis scaling
    plt.xlabel("$E_{\\mathrm{LLP}}$ [GeV]")
    plt.ylabel("Probability Density")
    plt.title("LLP Energy Distribution (Unweighted)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "energy_mother_unweighted.pdf"), bbox_inches='tight')
    plt.close()

    # Energy of mother particle (weighted by P_decay)
    plt.figure()
    plt.hist(energy_mother, bins=50, weights=P_decay_mother, color='salmon', edgecolor='black', density=True)
    plt.yscale('log')  # Preserving original y-axis scaling
    plt.xlabel("$E_{\\mathrm{LLP}}$ [GeV]")
    plt.ylabel("Probability Density")
    plt.title("LLP Energy Distribution (Weighted by $P_{\\mathrm{decay}}$)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "energy_mother_weighted.pdf"), bbox_inches='tight')
    plt.close()

    # P_decay of mother particle
    plt.figure()
    plt.hist(P_decay_mother, bins=50, color='lightgreen', edgecolor='black', density=True)
    plt.xscale('log')  # Preserving original x-axis scaling
    plt.yscale('log')  # Preserving original y-axis scaling
    plt.xlabel("$P_{\\mathrm{decay,LLP}}$")
    plt.ylabel("Probability Density")
    plt.title("LLP Decay Probability Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "P_decay_mother.pdf"), bbox_inches='tight')
    plt.close()

    # z_mother weighted by P_decay_mother
    plt.figure()
    plt.hist(z_mother, bins=50, weights=P_decay_mother, color='violet', edgecolor='black', density=True)
    plt.yscale('log')
    plt.xlabel("$z_{\\mathrm{decay,LLP}}$ [m]")
    plt.ylabel("Probability Density")
    plt.title("LLP Decay Positions (Weighted by $P_{\\mathrm{decay}}$)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "z_mother_weighted.pdf"), bbox_inches='tight')
    plt.close()

    # z_mother unweighted
    plt.figure()
    plt.hist(z_mother, bins=50, color='cyan', edgecolor='black', density=True)
    plt.yscale('log')
    plt.xlabel("$z_{\\mathrm{decay,LLP}}$ [m]")
    plt.ylabel("Probability Density")
    plt.title("LLP Decay Positions (Unweighted)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "z_mother_unweighted.pdf"), bbox_inches='tight')
    plt.close()

    # Merged histogram of total and charged decay products
    plt.figure()
    max_count = max(max(quantities['decay_products_counts']), max(quantities['charged_decay_products_counts']))
    bins = range(1, int(max_count) + 2)
    plt.hist(quantities['decay_products_counts'], bins=bins, alpha=0.5, label='Total decay products', color='blue', edgecolor='black', density=True)
    plt.hist(quantities['charged_decay_products_counts'], bins=bins, alpha=0.5, label='Charged decay products', color='yellow', edgecolor='black', density=True)
    plt.xlabel("Number of Decay Products")
    plt.ylabel("Probability Density")
    plt.title("Decay Products Multiplicity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "decay_products_counts_merged.pdf"), bbox_inches='tight')
    plt.close()

    # Histograms of counts per event for each decay product type
    decay_products_types = quantities['decay_products_per_event_counts'].keys()
    for ptype in decay_products_types:
        counts = quantities['decay_products_per_event_counts'][ptype]
        if counts:
            plt.figure()
            max_count = max(counts)
            bins = range(int(max_count) + 2)  # +2 to include max_count
            plt.hist(counts, bins=bins, align='left', edgecolor='black', color='lightcoral', density=True)
            plt.xlabel(f"Number of {ptype} per Event")
            plt.ylabel("Probability Density")
            plt.title(f"Amount of {ptype} per Event")
            plt.xticks(bins)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"decay_products_counts_{ptype}.pdf"), bbox_inches='tight')
            plt.close()

    # 3D scatter plot of (x_mother, y_mother, z_mother) unweighted
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_mother, y_mother, z_mother, s=1, alpha=0.5, c='blue')
    plot_decay_volume(ax)
    ax.set_xlabel(r"$x_{\mathrm{mother}}$")
    ax.set_ylabel(r"$y_{\mathrm{mother}}$")
    ax.set_zlabel(r"$z_{\mathrm{mother}}$")
    plt.title("Decay Positions of LLP (Unweighted)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "decay_positions_unweighted.pdf"), bbox_inches='tight')
    plt.close()

    # 3D scatter plot of (x_mother, y_mother, z_mother) weighted by P_decay
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_mother, y_mother, z_mother, s=1, alpha=0.1, c='blue', label='All Decays')
    plot_decay_volume(ax)
    ax.set_xlabel(r"$x_{\mathrm{mother}}$")
    ax.set_ylabel(r"$y_{\mathrm{mother}}$")
    ax.set_zlabel(r"$z_{\mathrm{mother}}$")
    plt.title("Decay Positions of LLP (Weighted by $P_{\\mathrm{decay}}$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "decay_positions_weighted.pdf"), bbox_inches='tight')
    plt.close()

    # ===========================
    # New 2D Point Plots Addition
    # ===========================

    # Create a mask for z_mother < 33
    mask_z = z_mother < 33

    # Unweighted 2D scatter plot of x and y decay coordinates for z_mother < 33
    plt.figure(figsize=(8,6))
    plt.scatter(x_mother[mask_z], y_mother[mask_z], s=1, alpha=0.5, c='blue')
    plt.xlabel(r"$x_{\mathrm{mother}}$ [m]")
    plt.ylabel(r"$y_{\mathrm{mother}}$ [m]")
    plt.title("Decay Positions (z < 33 m) Unweighted")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "decay_positions_xy_unweighted_z_less_33.pdf"), bbox_inches='tight')
    plt.close()

    # Weighted 2D scatter plot of x and y decay coordinates for z_mother < 33
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(x_mother[mask_z], y_mother[mask_z], s=1, alpha=0.5, c=P_decay_mother[mask_z], cmap='viridis')
    cbar = plt.colorbar(scatter)
    cbar.set_label("$P_{\\mathrm{decay}}$")
    plt.xlabel(r"$x_{\mathrm{mother}}$ [m]")
    plt.ylabel(r"$y_{\mathrm{mother}}$ [m]")
    plt.title("Decay Positions (z < 33 m) Weighted by $P_{\\mathrm{decay}}$")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "decay_positions_xy_weighted_z_less_33.pdf"), bbox_inches='tight')
    plt.close()

def main():
    # Directory containing the files
    directory = 'outputs'

    # Hardcoded export option
    ifExportData = True  # Set to True to export the data table

    # Step 1: Parse filenames
    llp_dict = parse_filenames(directory)

    # Step 2: User selection
    selected_file, selected_llp = user_selection(llp_dict)

    # Set plots_directory to 'plots/selected_llp'
    plots_directory = os.path.join('plots', selected_llp)

    # Get the output filename without extension
    output_filename = os.path.splitext(os.path.basename(selected_file))[0]
    output_path = os.path.join(plots_directory, output_filename)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Step 3: Read file
    filepath = os.path.join(directory, selected_file)
    finalEvents, epsilon_polar, epsilon_azimuthal, br_visible_val, channels = read_file(filepath)

    # Step 4: Plot channels
    plot_channels(channels, finalEvents, output_path)

    # Step 5: Extract quantities
    quantities = extract_quantities(channels)

    # Step 6: Export data table if option is True
    if ifExportData:
        # Convert lists to numpy arrays
        energy_mother = np.array(quantities['energy_mother'])
        P_decay_mother = np.array(quantities['P_decay_mother'])
        z_mother = np.array(quantities['z_mother'])

        # Stack the columns: P_decay, energy, z_mother
        data_table = np.column_stack((P_decay_mother, energy_mother, z_mother))

        # Save the data table to a single text file with space delimiter and no header
        np.savetxt(os.path.join(output_path, 'data_table.txt'), data_table, fmt='%.6e', delimiter=' ')

        print(f"Data table with P_decay, energy, z_mother has been exported to '{output_path}/data_table.txt'.")

    # Step 7: Plot histograms
    plot_histograms(quantities, output_path)

if __name__ == '__main__':
    main()

