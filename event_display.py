import os
import numpy as np
import re
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from random import sample
from matplotlib.lines import Line2D
from funcs.ship_volume import plot_decay_volume  # Imported from ship_volume.py

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
    Returns the selected filepath, the selected LLP name, mass, and lifetime.
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
        print(f"{i+1}. mass={mass:.2e} GeV, lifetime={lifetime:.2e}")

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
    print(f"Selected mass: {selected_mass:.2e} GeV, lifetime: {selected_lifetime:.2e}")

    # Get mixing patterns
    mixing_patterns_dict = llp_dict[selected_llp][selected_mass_lifetime]
    mixing_patterns_list = sorted(mixing_patterns_dict.keys())
    if any(mixing_patterns_list):
        print(f"Available mixing patterns for {selected_llp} with mass {selected_mass:.2e} GeV and lifetime {selected_lifetime:.2e}:")
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

    return selected_filepath, selected_llp, selected_mass, selected_lifetime

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

def get_pdg_color(pdg):
    """
    Returns the color corresponding to the pdg identifier.
    """
    if pdg in [22, 310]:
        return 'green'  # Neutral detectable
    elif pdg in [11, -11, 13, -13, 211, -211, 2112, -2112, 321, -321]:
        return 'blue'   # Charged
    elif pdg in [12, -12, 14, -14, 16, -16]:
        return 'gray'   # Neutrinos
    else:
        return 'cyan'   # Others

def main():
    # Directory containing the files
    directory = 'outputs'

    # Step 1: Parse filenames
    llp_dict = parse_filenames(directory)

    if not llp_dict:
        print("No LLP files found in the specified directory.")
        sys.exit(1)

    # Step 2: User selection
    selected_file, selected_llp, selected_mass, selected_lifetime = user_selection(llp_dict)

    # Format mass and lifetime for directory naming (retain scientific notation, remove '+' signs)
    mass_str = f"{selected_mass:.2e}".replace('+', '')
    lifetime_str = f"{selected_lifetime:.2e}".replace('+', '')

    # Set plots_directory to 'plots/<LLP>/EventDisplay/<LLP>_<mass>_<lifetime>'
    plots_directory = os.path.join('plots', selected_llp, 'EventDisplay', f"{selected_llp}_{mass_str}_{lifetime_str}")
    try:
        os.makedirs(plots_directory, exist_ok=True)
        print(f"Created directory: {plots_directory}")
    except Exception as e:
        print(f"Error creating directory {plots_directory}: {e}")
        sys.exit(1)

    # Step 3: Read file
    filepath = os.path.join(directory, selected_file)
    finalEvents, epsilon_polar, epsilon_azimuthal, br_visible_val, channels = read_file(filepath)

    if not channels:
        print("No channels found in the selected file.")
        sys.exit(1)

    # Step 4: User selects decay channel
    channel_names = sorted(channels.keys())
    print("\nAvailable Decay Channels:")
    for i, channel in enumerate(channel_names):
        print(f"{i+1}. {channel}")

    while True:
        try:
            channel_choice = int(input("Choose a decay channel by typing the number: "))
            if 1 <= channel_choice <= len(channel_names):
                break
            else:
                print(f"Please enter a number between 1 and {len(channel_names)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    selected_channel = channel_names[channel_choice - 1]
    print(f"Selected decay channel: {selected_channel}")

    # Get all data lines for the selected channel
    channel_data = channels[selected_channel]['data']
    total_events = len(channel_data)
    print(f"Total events in selected channel: {total_events}")

    if total_events < 10:
        print("Warning: Less than 10 events available in the selected channel. Proceeding with available events.")

    # Randomly select 10 events (or less if not enough)
    num_events_to_select = min(10, total_events)
    selected_events = sample(channel_data, num_events_to_select)
    print(f"Selected {num_events_to_select} events for plotting.")

    # Define legend elements for PDG color codes
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Neutral Detectable (pdg=22, 310)'),
        Line2D([0], [0], color='blue', lw=2, label='Charged (pdg=11, -11, 13, -13, 211, -211, 2112, -2112, 321, -321)'),
        Line2D([0], [0], color='gray', lw=2, label='Neutrinos (pdg=12, -12, 14, -14, 16, -16)'),
        Line2D([0], [0], color='cyan', lw=2, label='Others'),
        Line2D([0], [0], color='black', marker='o', linestyle='None', markersize=8, label='Decay Vertex'),
        Line2D([0], [0], color='red', lw=2, label='Mother Momentum')
    ]

    # Iterate over each selected event and create plots
    for idx, event_line in enumerate(selected_events, start=1):
        numbers = list(map(float, event_line.strip().split()))
        if len(numbers) < 10:
            print(f"Error: Event line {idx} has less than 10 numbers. Skipping.")
            continue

        # Extract x_mother, y_mother, z_mother (indices 7,8,9)
        x_mother = numbers[7]
        y_mother = numbers[8]
        z_mother = numbers[9]

        # Extract mother momentum (indices 0,1,2)
        px_mother = numbers[0]
        py_mother = numbers[1]
        pz_mother = numbers[2]
        p_mother = np.sqrt(px_mother**2 + py_mother**2 + pz_mother**2)
        if p_mother == 0:
            print(f"Error: Mother momentum is zero in event {idx}. Skipping momentum vector.")
            p_mother = 1  # To avoid division by zero

        # Normalize mother momentum
        p_mother_norm = np.array([px_mother, py_mother, pz_mother]) / p_mother

        # Extract decay products' momenta and assign colors
        decay_product_vectors = []
        decay_product_colors = []
        num_decay_products = (len(numbers) - 10) // 6
        for i in range(num_decay_products):
            pdg = numbers[10 + 6*i + 5]
            if pdg == -999.0:
                continue  # Skip fake particles
            px = numbers[10 + 6*i]
            py = numbers[10 + 6*i + 1]
            pz = numbers[10 + 6*i + 2]
            p = np.sqrt(px**2 + py**2 + pz**2)
            if p == 0:
                print(f"Warning: Decay product {i+1} in event {idx} has zero momentum. Skipping.")
                continue
            p_norm = np.array([px, py, pz]) / p
            decay_product_vectors.append(p_norm)
            decay_product_colors.append(get_pdg_color(int(pdg)))

        # Start plotting
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Rotate the plot so that the z-axis points to the right
        # To achieve this, set elev=0 and azim=-90
        ax.view_init(elev=0, azim=90)

        # Plot decay vertex
        ax.scatter(x_mother, y_mother, z_mother, color='black', s=50, label='Decay Vertex')

        # Plot mother momentum vector
        N = 10  # Scaling factor
        mother_vector = p_mother_norm * N  # Scale by N=10
        ax.quiver(x_mother, y_mother, z_mother,
                  mother_vector[0], mother_vector[1], mother_vector[2],
                  color='red', linewidth=2, label='Mother Momentum', arrow_length_ratio=0.1)

        # Plot decay products' momentum vectors with assigned colors
        for dp_vec, color in zip(decay_product_vectors, decay_product_colors):
            decay_product_vector = dp_vec * N  # Scale by N=10
            ax.quiver(x_mother, y_mother, z_mother,
                      decay_product_vector[0], decay_product_vector[1], decay_product_vector[2],
                      color=color, linewidth=1, arrow_length_ratio=0.1)

        # Plot decay volume
        plot_decay_volume(ax)

        # Set labels and title with reduced labelpad
        ax.set_xlabel(r"$x_{\mathrm{decay}}$ [m]", labelpad=2)
        ax.set_ylabel(r"$y_{\mathrm{decay}}$ [m]", labelpad=2)
        ax.set_zlabel(r"$z_{\mathrm{decay}}$ [m]", labelpad=2)
        ax.set_title(f"Event {idx}: {selected_llp} → {selected_channel}", pad=10)

        # Set fixed axis limits
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(25, 90)  # Fixed z-axis range from 25 to 90

        # Combine legend elements
        all_legend_elements = legend_elements

        # Create legend inside the plot, upper left corner
        ax.legend(handles=all_legend_elements, loc='upper left', fontsize='small', framealpha=0.7)

        # Reduce tick label padding
        ax.tick_params(axis='x', pad=2)
        ax.tick_params(axis='y', pad=2)
        ax.tick_params(axis='z', pad=2)

        # Adjust the layout and save the plot
        plt.tight_layout()
        plot_filename = f"{selected_llp}_{mass_str}_{lifetime_str}_{idx}.pdf"
        plot_path = os.path.join(plots_directory, plot_filename)
        try:
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            print(f"Saved plot for event {idx} to '{plot_path}'.")
        except Exception as e:
            print(f"Error saving plot for event {idx}: {e}")

    print("\nAll event display plots have been generated.")

if __name__ == '__main__':
    main()
