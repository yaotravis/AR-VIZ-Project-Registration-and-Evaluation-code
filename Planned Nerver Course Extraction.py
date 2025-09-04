# extract the line from the projected line of nerve course mesh (planning)
# Author: Yao Li
import numpy as np
import os
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.interpolate import splrep, splev
import gc

# 0 release the resources
def release_resources(point_cloud_o3d, middle_path_pcd=None):
    """
    Manually release point cloud resources and free memory.

    :param point_cloud_o3d: Open3D point cloud object.
    :param middle_path_pcd: Optional Open3D point cloud object for the path.
    """
    # delete Open3D object to release cache
    del point_cloud_o3d
    if middle_path_pcd:
        del middle_path_pcd

    #  release cache
    gc.collect()

# 1. Load the OBJ file as a point cloud
def load_obj_as_pointcloud(filename, number_of_points=10000):
    """
    Load a mesh from an OBJ file and sample it as a point cloud using Poisson disk sampling.

    :param filename: Path to the OBJ file.
    :param number_of_points: Number of points to sample.
    :return: Sampled point cloud.
    """
    mesh = o3d.io.read_triangle_mesh(filename)
    return mesh.sample_points_poisson_disk(number_of_points)


# 2. Save the point cloud as an OBJ file
def save_pointcloud_as_obj(point_cloud, file_path):
    """
    Save the given point cloud as a .obj file, representing it as a series of vertices.

    :param point_cloud: open3d.geometry.PointCloud object.
    :param file_path: Path to save the .obj file.
    """
    points = np.asarray(point_cloud.points)

    with open(file_path, 'w') as file:
        for point in points:
            file.write(f"v {point[0]} {point[1]} {point[2]}\n")

    print(f"OBJ file saved to: {file_path}")


# 3. Project the point cloud onto the XY plane and compute the middle points
def project_to_xy(vertices):
    # Ignore Z-axis and keep X and Y
    return vertices[:, :2]


def get_middle_y_per_x(vertices_2d, num_bins=100):
    """
    Scan the X-axis and find the middle Y value for each X interval.

    :param vertices_2d: 2D projected points onto the XY plane.
    :param num_bins: Number of bins along the X-axis.
    :return: Selected middle points of the path.
    """
    # Get the minimum and maximum of X-axis
    x_min, x_max = np.min(vertices_2d[:, 0]), np.max(vertices_2d[:, 0])

    # Create X-axis bins
    bins = np.linspace(x_min, x_max, num_bins + 1)

    # Store the middle points for each bin
    middle_points = []

    for i in range(num_bins):
        # Get the current bin range
        x_start, x_end = bins[i], bins[i + 1]

        # Find points within this bin
        bin_points = vertices_2d[(vertices_2d[:, 0] >= x_start) & (vertices_2d[:, 0] < x_end)]

        if len(bin_points) > 0:
            # Sort by Y-axis
            bin_points = bin_points[np.argsort(bin_points[:, 1])]

            # Select the middle Y value
            middle_index = len(bin_points) // 2
            middle_points.append(bin_points[middle_index])

    return np.array(middle_points)

# 4. reproject 3D space
def back_project_to_3d(middle_indices, original_points):
    """
        Back-project the 2D middle path back into 3D space.

        :param middle_indices: Indices of the selected middle path points.
        :param original_points: Original 3D points of the point cloud.
        :return: Back-projected 3D points.
    """
    return original_points[middle_indices]


# 5. Smooth the path using B-spline regression
def smooth_path_with_spline(middle_points, smoothing_factor=0):
    """
    Smooth the path using B-spline regression.

    :param middle_points: Middle points of the path (2D coordinates).
    :param smoothing_factor: Smoothing factor, default is 0. Higher values produce smoother results.
    :return: Smoothed path points.
    """

    # Extract X and Y coordinates
    X = middle_points[:, 0]
    Y = middle_points[:, 1]

    # Compute B-spline curve parameters using splrep
    tck = splrep(X, Y, s=smoothing_factor)

    # Compute smoothed Y values using splev
    Y_smooth = splev(X, tck)

    # Return the smoothed path points
    smoothed_path = np.column_stack((X, Y_smooth))
    return smoothed_path

# 6. Visualize 2D path
def visualize_2d(original_points, middle_points, smoothed_path):
    plt.scatter(original_points[:, 0], original_points[:, 1], label='Projected Points', s=10, color='blue')
    plt.plot(middle_points[:, 0], middle_points[:, 1], label='Middle Path', color='red', marker='o')
    plt.plot(smoothed_path[:, 0], smoothed_path[:, 1], label='Smoothed Path', color='green', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

# 7. Visualize 3D path
def visualize_3d(original_points, middle_points_3d, smoothed_points_3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], color=(0,0,1,0.2), s=10, label='Original Points')
    ax.plot(middle_points_3d[:, 0], middle_points_3d[:, 1], middle_points_3d[:, 2], color='red', marker='o', label='Middle Path')
    ax.plot(smoothed_points_3d[:, 0], smoothed_points_3d[:, 1], smoothed_points_3d[:, 2], color='green', label='Smoothed Path', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()


# Main function
if __name__ == "__main__":
    # Read the path to the OBJ file
    # change Plannung_Nerve_L to Plannung_Nerve_R for right side
    target_file = os.path.join(
        'C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - General/Auswertung/Obj/',
        "Plannung_Nerve_L.obj"
    )
    output_file = 'C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - General/Auswertung/Obj/Plannung_Nerve_L_Course.obj'

    # 1. Load the OBJ file and generate a point cloud
    point_cloud_o3d = load_obj_as_pointcloud(target_file, 10000)

    # Convert to NumPy array
    point_cloud_np = np.asarray(point_cloud_o3d.points)

    # 2. Project to the XY plane
    projected_2d_points = project_to_xy(point_cloud_np)

    # 3. Compute the middle Y value for each X interval
    middle_path_2d = get_middle_y_per_x(projected_2d_points, num_bins=200)

    # 4. Smooth the path using spline regression
    smoothed_path_2d = smooth_path_with_spline(middle_path_2d, smoothing_factor=3)

    # 5. Visualize the 2D path
    # visualize_2d(projected_2d_points, middle_path_2d, smoothed_path_2d)

    # 6. Back-project to 3D space using KDTree

    tree = KDTree(projected_2d_points)
    _, indices = tree.query(middle_path_2d)
    middle_path_3d = point_cloud_np[indices]

    _, smoothed_indices = tree.query(smoothed_path_2d)
    smoothed_path_3d = point_cloud_np[smoothed_indices]

    # 7. Visualize the 3D path
    visualize_3d(point_cloud_np, middle_path_3d, smoothed_path_3d)

    # 8. Save the smoothed path as an OBJ file
    smoothed_path_pcd = o3d.geometry.PointCloud()
    smoothed_path_pcd.points = o3d.utility.Vector3dVector(smoothed_path_3d)
    save_pointcloud_as_obj(smoothed_path_pcd, output_file)

    # Release resources and free memory
    release_resources(point_cloud_o3d, middle_path_pcd=None)