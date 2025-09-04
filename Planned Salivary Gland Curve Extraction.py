# extract the curve from the projection (on the head) of the planned salivary gland
# Author: Yao Li

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull
import alphashape

import os
import open3d as o3d
import matplotlib.pyplot as plt

def load_obj_as_pointcloud(filename, number_of_points=10000):
    mesh = o3d.io.read_triangle_mesh(filename)
    return mesh.sample_points_poisson_disk(number_of_points)

# change R to L and run again to get the Plannung_Parotid_L_Boundary
target_file = os.path.join(
    'C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - General/Auswertung/Obj/',
    "Plannung_Parotid_R.obj"
)


def save_pointcloud_as_obj(point_cloud, file_path):
    """
    Save the given point_cloud as a .obj file that Blender can import as a Curve object.

    :param point_cloud: open3d.geometry.PointCloud object
    :param file_path: Path to save the .obj file
    """
    # Get the vertices of the point cloud
    points = np.asarray(point_cloud.points)

    # Open the file to write
    with open(file_path, 'w') as file:
        # Write vertex information
        for point in points:
            file.write(f"v {point[0]} {point[1]} {point[2]}\n")

        # Write line information to connect vertices into a continuous line
        file.write("\n")
        for i in range(1, len(points) + 1):
            if i == len(points):
                # To close the curve, uncomment the following line to connect the last point to the first one
                # file.write(f"l {i} 1\n")
                continue
            file.write(f"l {i} {i + 1}\n")

    print(f"OBJ file saved to: {file_path}")


# Load the point cloud
point_cloud_o3d = load_obj_as_pointcloud(target_file, 10000)

# Convert the PointCloud object into a NumPy array
point_cloud_np = np.asarray(point_cloud_o3d.points)  # Extract the point cloud coordinates as a NumPy array

# Project onto a 2D plane by ignoring the Z axis, take the first two coordinates (X and Y)
projected_2d_points = point_cloud_np[:, :2]


# Calculate the Alpha shape, adjust alpha value as needed
alpha = 1.5  # Adjust this value for tighter or looser boundaries
alpha_shape = alphashape.alphashape(projected_2d_points, alpha)

# Get the boundary points of the Alpha shape
boundary_2d_points = np.array(alpha_shape.exterior.coords)

# Use KDTree to find the nearest original 3D point for each 2D boundary point
tree = KDTree(point_cloud_np[:, :2])  # Only consider XY coordinates

# Find the nearest original 3D point indices for each 2D boundary point
_, indices = tree.query(boundary_2d_points)

# Get the corresponding 3D boundary points
boundary_3d_points = point_cloud_np[indices]

# # Visualize the 2D points and the convex hull boundary
# plt.figure()
# plt.plot(projected_2d_points[:, 0], projected_2d_points[:, 1], 'o', markersize=2, label="Projected Points")  # Plot the projected points
# plt.plot(boundary_2d_points[:, 0], boundary_2d_points[:, 1], 'r-', label="Convex Hull")  # Plot the convex hull boundary
# plt.fill(boundary_2d_points[:, 0], boundary_2d_points[:, 1], 'r', alpha=0.3)  # Fill the convex hull region
#
# plt.title("2D Projection and Convex Hull")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend()
# plt.show()

# Create the boundary point cloud
boundary_pcd = o3d.geometry.PointCloud()
boundary_pcd.points = o3d.utility.Vector3dVector(boundary_3d_points)

# Create the full original point cloud for comparison
original_pcd = o3d.geometry.PointCloud()
original_pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

# Set different colors for distinction
boundary_pcd.paint_uniform_color([1, 0, 0])  # Boundary points in red
# original_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Original point cloud in gray

# change R to L and run again to get the Plannung_Parotid_L_Boundary
save_pointcloud_as_obj(boundary_pcd, "C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - General/Auswertung/Obj/Plannung_Parotid_R_Boundary.obj")
# Visualize using open3d
o3d.visualization.draw_geometries([boundary_pcd])