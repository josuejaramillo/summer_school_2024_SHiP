import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import re

def scan_outputs_folders(outputs_dir):
    """
    Scans the 'outputs' directory for non-empty folders and returns their names.
    """
    try:
        folders = [f for f in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, f))]
        non_empty_folders = []
        for folder in folders:
            folder_path = os.path.join(outputs_dir, folder)
            if any(os.scandir(folder_path)):
                non_empty_folders.append(folder)
        return non_empty_folders
    except FileNotFoundError:
        print(f"Error: The directory '{outputs_dir}' does not exist.")
        sys.exit(1)

def select_option(options, prompt):
    """
    Displays a list of options to the user and prompts for a selection.
    Returns the selected option.
    """
    if not options:
        print("No options available for selection.")
        sys.exit(1)
    
    print(prompt)
    for idx, option in enumerate(options, start=1):
        print(f"{idx}. {option}")
    
    while True:
        try:
            choice = int(input("Enter the number corresponding to your choice: "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                print(f"Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def extract_mixing_patterns(total_dir):
    """
    Extracts mixing patterns from filenames in the 'total' directory for HNL.
    Returns a list of tuples (filename, [XX, YY, ZZ]).
    """
    pattern = re.compile(r"HNL_([\d\.\+eE-]+)_([\d\.\+eE-]+)_([\d\.\+eE-]+)_total\.txt$")
    mixing_files = []
    
    try:
        files = os.listdir(total_dir)
    except FileNotFoundError:
        print(f"Error: The directory '{total_dir}' does not exist.")
        sys.exit(1)
    
    for file in files:
        match = pattern.match(file)
        if match:
            xx, yy, zz = match.groups()
            # Convert scientific notation to float with two decimal places
            try:
                xx_float = float(xx)
                yy_float = float(yy)
                zz_float = float(zz)
                mixing_pattern = f"[{xx_float:.2f}, {yy_float:.2f}, {zz_float:.2f}]"
                mixing_files.append((file, mixing_pattern))
            except ValueError:
                print(f"Warning: Unable to parse mixing pattern from file '{file}'. Skipping.")
    
    return mixing_files

def plot_data(data, save_path, title):
    """
    Plots the data as specified and saves the plot to the given path.
    """
    # Sort the data by 'mass' to ensure the lines connect logically
    data_sorted = data.sort_values(by='mass').reset_index(drop=True)
    
    # Extract required columns and convert to NumPy arrays
    try:
        mass = data_sorted['mass'].to_numpy()
        epsilon_polar = data_sorted['epsilon_polar'].to_numpy()
        epsilon_azimuthal = data_sorted['epsilon_azimuthal'].to_numpy()
        c_tau = data_sorted['c_tau'].to_numpy()
        P_decay_averaged = data_sorted['P_decay_averaged'].to_numpy()
    except KeyError as e:
        print(f"Error: Missing expected column: {e}")
        sys.exit(1)
    
    # Compute required products
    epsilon_geom = epsilon_polar * epsilon_azimuthal
    epsilon_geom_P_decay = c_tau * epsilon_geom * P_decay_averaged
    
    # Create the plot
    plt.figure(figsize=(10, 7))
    
    # Plot epsilon_polar vs mass
    plt.plot(mass, epsilon_polar, label=r'$\epsilon_{\mathrm{polar}}$', linewidth=2, marker='o')
    
    # Plot epsilon_geom vs mass
    plt.plot(mass, epsilon_geom, label=r'$\epsilon_{\mathrm{geom}}$', linewidth=2, marker='s')
    
    # Plot cτ <epsilon_geom * P_decay> vs mass
    plt.plot(mass, epsilon_geom_P_decay, label=r'$c\tau \langle \epsilon_{\mathrm{geom}} \cdot P_{\mathrm{decay}} \rangle$', linewidth=2, marker='^')
    
    # Set axis labels with LaTeX formatting
    plt.xlabel(r'$m_{\mathrm{LLP}}$ [GeV]', fontsize=14)
    plt.ylabel('Fraction', fontsize=14)
    
    # Set y-axis to logarithmic scale
    plt.yscale('log')
    
    # Set plot title
    plt.title(title, fontsize=16)
    
    # Add legend with a larger font size
    plt.legend(fontsize=12)
    
    # Enable grid for better readability
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Save the plot to the specified path
    try:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved successfully to '{save_path}'.")
    except Exception as e:
        print(f"Error saving the plot: {e}")
        sys.exit(1)
    
    # Display the plot
    plt.show()

def main():
    # Define the base directory (assuming the script is run from 'basedir')
    basedir = os.getcwd()
    
    # Define the outputs directory
    outputs_dir = os.path.join(basedir, 'outputs')
    
    # Scan for non-empty folders in 'outputs'
    llp_folders = scan_outputs_folders(outputs_dir)
    
    if not llp_folders:
        print(f"No non-empty folders found in '{outputs_dir}'. Exiting.")
        sys.exit(1)
    
    # Prompt user to select the LLP
    selected_llp = select_option(llp_folders, "Select the LLP:")
    print(f"Selected LLP: {selected_llp}")
    
    # Initialize variables
    data_file_path = ""
    plot_title = ""
    plot_filename = ""
    
    if selected_llp != "HNL":
        # Path for non-HNL LLPs
        data_file = f"{selected_llp}_total.txt"  # Corrected filename
        data_file_path = os.path.join(outputs_dir, selected_llp, 'total', data_file)
        
        # Check if the data file exists
        if not os.path.isfile(data_file_path):
            print(f"Error: Data file '{data_file_path}' not found.")
            sys.exit(1)
        
        # Define plot saving directory
        plot_save_dir = os.path.join(basedir, 'plots', selected_llp)
        os.makedirs(plot_save_dir, exist_ok=True)
        
        # Define plot filename
        plot_filename = f"{selected_llp}-plot.png"  # Corrected plot name
        
        # Define plot title
        plot_title = f"{selected_llp}"
    
    else:
        # Handle HNL LLP with multiple mixing patterns
        total_dir = os.path.join(outputs_dir, selected_llp, 'total')
        mixing_files = extract_mixing_patterns(total_dir)
        
        if not mixing_files:
            print(f"No mixing pattern files found in '{total_dir}'. Exiting.")
            sys.exit(1)
        
        # Prepare mixing pattern options
        mixing_patterns = [pattern for (_, pattern) in mixing_files]
        mixing_filenames = [filename for (filename, _) in mixing_files]
        
        # Prompt user to select mixing pattern
        selected_mix_pattern = select_option(mixing_patterns, "Select the mixing pattern:")
        selected_index = mixing_patterns.index(selected_mix_pattern)
        selected_file = mixing_filenames[selected_index]
        data_file_path = os.path.join(total_dir, selected_file)
        
        # Define plot saving directory
        plot_save_dir = os.path.join(basedir, 'plots', selected_llp)
        os.makedirs(plot_save_dir, exist_ok=True)
        
        # Extract XX, YY, ZZ from the selected mixing pattern
        # Example pattern: [1.00, 0.00, 0.00]
        pattern_numbers = selected_mix_pattern.strip("[]").split(", ")
        xx, yy, zz = pattern_numbers
        
        # Define plot filename including mixing pattern
        plot_filename = f"HNL-{xx}-{yy}-{zz}-plot.png"  # Corrected plot name
        
        # Define plot title including mixing pattern
        plot_title = f"{selected_llp} - Mixing Pattern {selected_mix_pattern}"
    
    # Define full plot save path
    plot_save_path = os.path.join(plot_save_dir, plot_filename)
    
    # Read the data using pandas
    try:
        data = pd.read_csv(data_file_path, delim_whitespace=True, header=0)
    except FileNotFoundError:
        print(f"Error: File '{data_file_path}' not found.")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"Error parsing '{data_file_path}': {e}")
        sys.exit(1)
    
    # Verify that all required columns are present
    required_columns = [
        'mass', 'coupling_squared', 'c_tau', 'N_LLP_tot',
        'epsilon_polar', 'epsilon_azimuthal', 'P_decay_averaged',
        'Br_visible', 'N_ev_tot'
    ]
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Error: Missing columns in data file: {missing_columns}")
        sys.exit(1)
    
    # Plot the data
    plot_data(data, plot_save_path, plot_title)

if __name__ == "__main__":
    main()

