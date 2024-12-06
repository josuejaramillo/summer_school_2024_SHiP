from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

