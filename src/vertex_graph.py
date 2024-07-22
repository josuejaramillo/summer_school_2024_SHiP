import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import numpy as np

def plot3D(path):
    # Define the vertices of the truncated pyramid

    # Vertices for the lower base (smaller rectangle), located at height 32
    v0 = [-0.5, -1.35, 32]  # Bottom-left
    v1 = [0.5, -1.35, 32]   # Bottom-right
    v2 = [0.5, 1.35, 32]    # Top-right
    v3 = [-0.5, 1.35, 32]   # Top-left

    # Vertices for the upper base (larger rectangle), located at height 82
    base_sup_x_offset = 2
    base_sup_y_offset = 3.1
    height = 82

    v4 = [-base_sup_x_offset, -base_sup_y_offset, height]  # Bottom-left
    v5 = [base_sup_x_offset, -base_sup_y_offset, height]   # Bottom-right
    v6 = [base_sup_x_offset, base_sup_y_offset, height]    # Top-right
    v7 = [-base_sup_x_offset, base_sup_y_offset, height]   # Top-left

    # Create a list of vertices for the truncated pyramid
    vertices = [v0, v1, v2, v3, v4, v5, v6, v7]

    # Define the faces of the truncated pyramid by connecting the vertices
    faces = [
        [v0, v1, v5, v4],  # Front face
        [v1, v2, v6, v5],  # Right face
        [v2, v3, v7, v6],  # Back face
        [v3, v0, v4, v7],  # Left face
        [v4, v5, v6, v7],  # Upper base
        [v0, v1, v2, v3]   # Lower base
    ]

    # Create a figure and 3D axes for plotting
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Add the faces of the truncated pyramid to the plot
    poly3d = Poly3DCollection(faces, edgecolors='k', linewidths=1, alpha=0.2)
    ax.add_collection3d(poly3d)

    # Set the limits for the x, y, and z axes
    ax.set_xlim([-2, 2])
    ax.set_ylim([-3, 3])
    ax.set_zlim([30, 85])

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Read data from a tab-separated file
    data = pd.read_csv(path + "/kinetic_sampling.dat", sep="\t")
    x, y, z = data["x"], data["y"], data["z"]

    # Apply a mask to filter the data based on the decay probability
    mask = data["P_decay"] >= np.random.rand(len(data["P_decay"]))

    # Plot the filtered data points
    ax.scatter(x[mask], y[mask], z[mask], color='k', s=0.5)  # Small points for clarity
    # ax.scatter(x, y, z, color='k', s=1)  # Alternative: scatter all points

    # Customize the plot appearance
    ax.grid(False)  # Turn off the grid
    ax.view_init(elev=0, azim=-60, roll=-90)  # Set the viewing angle for better visualization
    ax.tick_params(axis='both', which='major', labelsize=8)  # Adjust tick parameters
    ax.set_box_aspect([1, 1, 2])  # Set aspect ratio to 1:1:2 for a taller Z dimension

    # Save the figure to a file
    fig.savefig(path + "/vertices.png", dpi=300)

    # Display the plot
    plt.show()
