import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import numpy as np

# Define the vertices of the truncated pyramid
# Lower base (smaller rectangle, now at height 32)
v0 = [-0.5, -1.35, 32]
v1 = [0.5, -1.35, 32]
v2 = [0.5, 1.35, 32]
v3 = [-0.5, 1.35, 32]

# Upper base (larger rectangle, now at height 82)
base_sup_x_offset = 2
base_sup_y_offset = 3.1
height = 82

v4 = [-base_sup_x_offset, -base_sup_y_offset, height]
v5 = [base_sup_x_offset, -base_sup_y_offset, height]
v6 = [base_sup_x_offset, base_sup_y_offset, height]
v7 = [-base_sup_x_offset, base_sup_y_offset, height]

# Create a list of vertices
vertices = [v0, v1, v2, v3, v4, v5, v6, v7]

# Define the faces of the truncated pyramid
faces = [
    [v0, v1, v5, v4],  # Front face
    [v1, v2, v6, v5],  # Right face
    [v2, v3, v7, v6],  # Back face
    [v3, v0, v4, v7],  # Left face
    [v4, v5, v6, v7],  # Upper base
    [v0, v1, v2, v3]   # Lower base
]

# Create the figure and 3D axes
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Add the faces to the plot
poly3d = Poly3DCollection(faces, edgecolors='k', linewidths=1, alpha=0.3)
ax.add_collection3d(poly3d)

# Set the limits of the axes
ax.set_xlim([-3, 3])
ax.set_ylim([-4, 4])
ax.set_zlim([0, 100])

# Axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#Plot data
path = "./Distributions/Higgs_like_scalars"
# path = "./Distributions/Dark_photons"

data = pd.read_csv(path+"/kinetic_sampling.dat", sep="\t")
x,y,z = data["x"], data["y"], data["z"]

mask = data["P_decay"] >= np.random.rand(len(data["P_decay"]))

ax.scatter(x[mask], y[mask], z[mask], color='k', s=1)
# ax.scatter(x, y, z, color='k', s=1)

# Show the plot
ax.set_box_aspect([1, 1, 2])  # Aspect ratio is 1:1:2 for Z to appear taller
fig.savefig(path+"/vertices.png")
plt.show()
