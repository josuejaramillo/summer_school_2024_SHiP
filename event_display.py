import os
import numpy as np
import re
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from funcs.ship_volume import plot_decay_volume  # Ensure this module is available
import plotly.graph_objects as go
from random import sample
from matplotlib.lines import Line2D
from funcs.selecting_processing import parse_filenames, user_selection, read_file  # Importing common functions

def get_pdg_color(pdg):
    """
    Returns the color corresponding to the pdg identifier.
    """
    if pdg in [22, 130, 2212, -2212]:
        return 'green'  # Neutral detectable
    elif pdg in [11, -11, 13, -13, 211, -211, 2112, -2112, 321, -321]:
        return 'blue'   # Charged
    elif pdg in [12, -12, 14, -14, 16, -16]:
        return 'gray'   # Neutrinos
    else:
        return 'cyan'   # Others

def plot_event_matplotlib(ax, x_mother, y_mother, z_mother, p_mother_norm, decay_product_vectors, decay_product_colors):
    """
    Plots the event using Matplotlib.
    """
    # Plot decay vertex
    ax.scatter(x_mother, y_mother, z_mother, color='black', s=50, label='Decay Vertex')

    # Plot mother momentum vector
    N = 20  # Increased scaling factor for better visibility
    mother_vector = p_mother_norm * N  # Scale by N=20
    ax.quiver(x_mother, y_mother, z_mother,
              mother_vector[0], mother_vector[1], mother_vector[2],
              color='red', linewidth=2, label='Mother Momentum', arrow_length_ratio=0.1)

    # Plot decay products' momentum vectors with assigned colors
    for dp_vec, color in zip(decay_product_vectors, decay_product_colors):
        decay_product_vector = dp_vec * N  # Scale by N=20
        ax.quiver(x_mother, y_mother, z_mother,
                  decay_product_vector[0], decay_product_vector[1], decay_product_vector[2],
                  color=color, linewidth=1, label='Decay Product', arrow_length_ratio=0.1)

    # Plot decay volume
    plot_decay_volume(ax)

    # Set labels and title with reduced labelpad
    ax.set_xlabel("x [m]", labelpad=2)
    ax.set_ylabel("y [m]", labelpad=2)
    ax.set_zlabel("z [m]", labelpad=2)

def plot_event_plotly(fig, x_mother, y_mother, z_mother, p_mother_norm, decay_product_vectors, decay_product_colors, legend_elements):
    """
    Plots the event using Plotly.
    """
    # Plot decay vertex
    fig.add_trace(go.Scatter3d(
        x=[x_mother],
        y=[y_mother],
        z=[z_mother],
        mode='markers',
        marker=dict(size=5, color='black'),
        name='Decay Vertex',
        hoverinfo='text',
        text=[f"Decay Vertex<br>(x={x_mother:.2f}, y={y_mother:.2f}, z={z_mother:.2f})"],
        showlegend=False  # Hide in legend; will add a dummy trace later
    ))

    # Define scaling factors
    vector_scale = 20  # Length of the vectors (lines)
    cone_scale = 1.66  # Reduced size of the cones (arrows), 5 / 3 ≈ 1.66

    # Plot mother momentum vector as a line
    mother_end = [x_mother + p_mother_norm[0]*vector_scale,
                 y_mother + p_mother_norm[1]*vector_scale,
                 z_mother + p_mother_norm[2]*vector_scale]
    fig.add_trace(go.Scatter3d(
        x=[x_mother, mother_end[0]],
        y=[y_mother, mother_end[1]],
        z=[z_mother, mother_end[2]],
        mode='lines',
        line=dict(color='red', width=6),  # Increased line width
        name='Mother Momentum',
        hoverinfo='skip',
        showlegend=False  # Hide in legend; will add a dummy trace later
    ))

    # Plot mother momentum vector as a cone at the end
    fig.add_trace(go.Cone(
        x=[mother_end[0]],
        y=[mother_end[1]],
        z=[mother_end[2]],
        u=[p_mother_norm[0]],
        v=[p_mother_norm[1]],
        w=[p_mother_norm[2]],
        colorscale=[[0, 'red'], [1, 'red']],
        sizemode="absolute",
        sizeref=cone_scale,  # Reduced sizeref for smaller cones
        showscale=False,
        name='Mother Momentum Cone',
        hoverinfo='text',
        text=[f"Mother Momentum<br>(px={p_mother_norm[0]:.2f}, py={p_mother_norm[1]:.2f}, pz={p_mother_norm[2]:.2f})"],
        showlegend=False  # Hide in legend; will add a dummy trace later
    ))

    # Plot decay products' momentum vectors as lines and cones
    for dp_vec, color in zip(decay_product_vectors, decay_product_colors):
        dp_end = [x_mother + dp_vec[0]*vector_scale,
                  y_mother + dp_vec[1]*vector_scale,
                  z_mother + dp_vec[2]*vector_scale]
        # Plot line
        fig.add_trace(go.Scatter3d(
            x=[x_mother, dp_end[0]],
            y=[y_mother, dp_end[1]],
            z=[z_mother, dp_end[2]],
            mode='lines',
            line=dict(color=color, width=6),  # Increased line width
            name='Decay Product',
            hoverinfo='skip',
            showlegend=False  # Hide in legend; will add a dummy trace later
        ))
        # Plot cone at the end
        fig.add_trace(go.Cone(
            x=[dp_end[0]],
            y=[dp_end[1]],
            z=[dp_end[2]],
            u=[dp_vec[0]],
            v=[dp_vec[1]],
            w=[dp_vec[2]],
            colorscale=[[0, color], [1, color]],
            sizemode="absolute",
            sizeref=cone_scale,  # Reduced sizeref for smaller cones
            showscale=False,
            name='Decay Product Cone',
            hoverinfo='text',
            text=[f"Decay Product<br>(px={dp_vec[0]:.2f}, py={dp_vec[1]:.2f}, pz={dp_vec[2]:.2f})"],
            showlegend=False  # Hide in legend; will add a dummy trace later
        ))

    # Plot decay volume
    plot_decay_volume_plotly(fig)

    # Set layout properties with simple axis labels
    fig.update_layout(
        scene=dict(
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            zaxis_title="z [m]",
            xaxis=dict(range=[-10, 10]),
            yaxis=dict(range=[-10, 10]),
            zaxis=dict(range=[25, 90]),
            aspectratio=dict(x=1, y=1, z=1.5),  # Decreased z aspect ratio
            aspectmode='manual'
        ),
        title=f"Event: Momentum Vectors",
        legend=dict(
            itemsizing='constant',
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='Black',
            borderwidth=1
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # Create custom legend using dummy traces
    # Add dummy traces for legend items
    for elem in legend_elements:
        if 'marker' in elem and elem['marker']:
            # For markers like Decay Vertex
            fig.add_trace(go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode='markers',
                marker=dict(size=5, color=elem['color']),
                name=elem['label'],
                showlegend=True
            ))
        else:
            # For lines like Mother Momentum and particle categories
            fig.add_trace(go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode='lines',
                line=dict(color=elem['color'], width=6),  # Match line width
                name=elem['label'],
                showlegend=True
            ))

    return fig

def main():
    # Directory containing the files
    directory = 'outputs'

    # Step 1: Parse filenames
    llp_dict = parse_filenames(directory)

    if not llp_dict:
        print("No LLP files found in the specified directory.")
        sys.exit(1)

    # Step 2: User selection
    selected_file, selected_llp, selected_mass, selected_lifetime, selected_mixing_patterns = user_selection(llp_dict)

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

    # Initialize list to store PDG IDs for each event
    pdg_ids_list = []

    # Define legend elements for PDG color codes
    legend_elements = [
        {'label': 'Neutral Detectable (pdg = 22, 130, 2112)', 'color': 'green'},
        {'label': 'Charged (pdg = 11, 13, 211, 2212, 321)', 'color': 'blue'},
        {'label': 'Neutrinos (pdg = 12, 14, 16)', 'color': 'gray'},
        {'label': 'Others', 'color': 'cyan'},
        {'label': 'Decay Vertex', 'color': 'black', 'marker': True},
        {'label': 'Mother Momentum', 'color': 'red'}
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
        pdg_ids_in_event = []
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
            pdg_id = int(pdg)
            pdg_ids_in_event.append(pdg_id)
            decay_product_vectors.append(p_norm)
            decay_product_colors.append(get_pdg_color(pdg_id))

        # Append the PDG IDs of this event to the list
        pdg_ids_list.append(pdg_ids_in_event)

        # -------------------- Matplotlib Plotting --------------------
        fig_matplotlib = plt.figure(figsize=(10, 8))
        ax_matplotlib = fig_matplotlib.add_subplot(111, projection='3d')

        # Plot the event using Matplotlib
        plot_event_matplotlib(
            ax_matplotlib,
            x_mother,
            y_mother,
            z_mother,
            p_mother_norm,
            decay_product_vectors,
            decay_product_colors
        )

        # Set fixed axis limits
        ax_matplotlib.set_xlim(-10, 10)
        ax_matplotlib.set_ylim(-10, 10)
        ax_matplotlib.set_zlim(25, 90)  # Fixed z-axis range from 25 to 90

        # Create custom legend
        handles = []
        labels = []
        for elem in legend_elements:
            if 'marker' in elem and elem['marker']:
                handle = Line2D([0], [0], marker='o', color='w', label=elem['label'],
                                markerfacecolor=elem['color'], markersize=8)
            else:
                handle = Line2D([0], [0], color=elem['color'], lw=2, label=elem['label'])
            handles.append(handle)
            labels.append(elem['label'])
        ax_matplotlib.legend(handles, labels, loc='upper left', fontsize='small', framealpha=0.7)

        # Reduce tick label padding
        ax_matplotlib.tick_params(axis='x', pad=2)
        ax_matplotlib.tick_params(axis='y', pad=2)
        ax_matplotlib.tick_params(axis='z', pad=2)

        # Adjust the layout and save the plot
        plt.tight_layout()
        plot_filename_pdf = f"{selected_llp}_{mass_str}_{lifetime_str}_{idx}.pdf"
        plot_path_pdf = os.path.join(plots_directory, plot_filename_pdf)
        try:
            plt.savefig(plot_path_pdf, bbox_inches='tight')
            plt.close(fig_matplotlib)
            print(f"Saved Matplotlib plot for event {idx} to '{plot_path_pdf}'.")
        except Exception as e:
            print(f"Error saving Matplotlib plot for event {idx}: {e}")

        # -------------------- Plotly Plotting --------------------
        fig_plotly = go.Figure()

        # Plot the event using Plotly
        fig_plotly = plot_event_plotly(
            fig_plotly,
            x_mother,
            y_mother,
            z_mother,
            p_mother_norm,
            decay_product_vectors,
            decay_product_colors,
            legend_elements  # Pass legend_elements as a parameter
        )

        # Set layout properties with simple axis labels
        fig_plotly.update_layout(
            title=f"Event {idx}: {selected_llp} → {selected_channel}",
            legend=dict(
                itemsizing='constant',
                bgcolor='rgba(255,255,255,0.5)',
                bordercolor='Black',
                borderwidth=1
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )

        # Save as HTML
        plot_filename_html = f"{selected_llp}_{mass_str}_{lifetime_str}_{idx}.html"
        plot_path_html = os.path.join(plots_directory, plot_filename_html)
        try:
            fig_plotly.write_html(plot_path_html)
            print(f"Saved Plotly plot for event {idx} to '{plot_path_html}'.")
        except Exception as e:
            print(f"Error saving Plotly plot for event {idx}: {e}")

    # -------------------- Export PDG IDs to TXT --------------------
    pdg_ids_txt_path = os.path.join(plots_directory, 'pdg_ids.txt')
    try:
        with open(pdg_ids_txt_path, 'w') as f:
            for event_idx, pdg_ids in enumerate(pdg_ids_list, start=1):
                pdg_ids_str = ' '.join(map(str, pdg_ids)) if pdg_ids else 'None'
                f.write(f"Event {event_idx}: {pdg_ids_str}\n")
        print(f"Saved PDG identifiers to '{pdg_ids_txt_path}'.")
    except Exception as e:
        print(f"Error saving PDG identifiers to '{pdg_ids_txt_path}': {e}")

    print("\nAll event display plots and PDG identifiers have been generated.")

def plot_decay_volume_plotly(fig):
    """
    Adds the SHiP decay volume to the given Plotly figure.
    The decay volume is represented as a trapezoidal prism with visible boundaries.
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
        [x_min_zmin, y_min_zmin, z_min],  # vertex0
        [x_max_zmin, y_min_zmin, z_min],  # vertex1
        [x_max_zmin, y_max_zmin, z_min],  # vertex2
        [x_min_zmin, y_max_zmin, z_min],  # vertex3
        [x_min_zmax, y_min_zmax, z_max],  # vertex4
        [x_max_zmax, y_min_zmax, z_max],  # vertex5
        [x_max_zmax, y_max_zmax, z_max],  # vertex6
        [x_min_zmax, y_max_zmax, z_max]   # vertex7
    ]

    # Define the faces of the decay volume
    faces = [
        [0, 1, 2, 3],  # Bottom face
        [4, 5, 6, 7],  # Top face
        [0, 1, 5, 4],  # Front face
        [1, 2, 6, 5],  # Right face
        [2, 3, 7, 6],  # Back face
        [3, 0, 4, 7]   # Left face
    ]

    # Create mesh for the decay volume
    for face in faces:
        fig.add_trace(go.Mesh3d(
            x=[vertices[i][0] for i in face],
            y=[vertices[i][1] for i in face],
            z=[vertices[i][2] for i in face],
            color='rgba(200, 200, 200, 0.5)',  # Light gray with transparency
            opacity=0.5,
            name='Decay Volume',
            showscale=False,
            hoverinfo='skip'
        ))

    # Define the edges for the wireframe
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom edges
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top edges
        (0, 4), (1, 5), (2, 6), (3, 7)   # Side edges
    ]

    # Add edges as Scatter3d lines
    for edge in edges:
        fig.add_trace(go.Scatter3d(
            x=[vertices[edge[0]][0], vertices[edge[1]][0]],
            y=[vertices[edge[0]][1], vertices[edge[1]][1]],
            z=[vertices[edge[0]][2], vertices[edge[1]][2]],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

    return fig

if __name__ == '__main__':
    main()

