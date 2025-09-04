# this code is calculate the Euclidean distance between the drawm and planned NEPs and post the deviation on the x y z axis.
# Author: Yao Li
import os
import glob
import numpy as np
import open3d as o3d
import csv
from sklearn.decomposition import PCA
import pandas as pd
from scipy.optimize import linear_sum_assignment


def load_obj_as_pointcloud(filename):
    """
    Manually load OBJ file and extract the vertex (v) data as a point cloud.

    :param filename: Path to the OBJ file.
    :return: NumPy array of point cloud vertices.
    """
    points = []

    # Open the OBJ file and extract vertex lines
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Vertex line starts with 'v'
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                points.append([x, y, z])

    return np.array(points)


def compute_pca_axes(points):
    """
    Compute PCA axes for a set of points.

    :param points: NumPy array of shape (N, 3), representing the point cloud.
    :return: PCA basis vectors as a 3x3 NumPy array.
    """
    pca = PCA(n_components=3)
    pca.fit(points)
    return pca.components_  # Rows are principal axes


def align_pca_to_world(pca_axes):
    """
    Align PCA axes to world coordinate axes (X, Y, Z) by maximizing alignment and ensuring consistent direction.

    :param pca_axes: PCA axes as a 3x3 NumPy array (rows are the axes).
    :return: Aligned PCA axes as a 3x3 NumPy array.
    """
    world_axes = np.array([
        [1, 0, 0],  # World X
        [0, 1, 0],  # World Y
        [0, 0, 1]  # World Z
    ])

    aligned_axes = np.zeros_like(pca_axes)

    used_indices = set()  # Track assigned world axes
    for i in range(3):
        # Compute alignment scores
        similarities = np.dot(pca_axes, world_axes[i])  # Keep sign to detect direction
        best_match = np.argmax(np.abs(similarities))  # Find the best aligned PCA axis
        while best_match in used_indices:  # Ensure axes are not reused
            similarities[best_match] = -1  # Mark as used
            best_match = np.argmax(np.abs(similarities))

        used_indices.add(best_match)
        aligned_axes[i] = pca_axes[best_match] * np.sign(similarities[best_match])  # Adjust direction

    return aligned_axes



def visualize_with_open3d(points_head, points_planned, points_drawn, centroid_planned, centroid_drawn):
    """
    Visualize the planned and drawn point clouds along with their centroids and PCA-aligned coordinate frame.
    """
    # Create point clouds
    pcd_head = o3d.geometry.PointCloud()
    pcd_planned = o3d.geometry.PointCloud()
    pcd_drawn = o3d.geometry.PointCloud()

    # Assign points to the point clouds
    pcd_head.points = o3d.utility.Vector3dVector(points_head)
    pcd_planned.points = o3d.utility.Vector3dVector(points_planned)
    pcd_drawn.points = o3d.utility.Vector3dVector(points_drawn)

    # Color the point clouds
    pcd_head.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for head
    pcd_planned.paint_uniform_color([0, 1, 0])  # Green for planned
    pcd_drawn.paint_uniform_color([1, 0, 0])  # Red for drawn

    # Create spheres to represent centroids
    sphere_planned = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    sphere_drawn = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    sphere_planned.translate(centroid_planned)
    sphere_drawn.translate(centroid_drawn)
    sphere_planned.paint_uniform_color([0, 0, 1])  # Blue for planned centroid
    sphere_drawn.paint_uniform_color([1, 1, 0])  # Yellow for drawn centroid

    # PCA for points_head to determine new axes and align to world coordinates
    pca_axes = compute_pca_axes(points_head)
    aligned_axes = align_pca_to_world(pca_axes)

    # Create a coordinate frame using the aligned PCA axes
    pca_frame = o3d.geometry.LineSet()
    origin = np.mean(points_head, axis=0)
    points = [origin, origin + aligned_axes[0] * 50, origin + aligned_axes[1] * 50, origin + aligned_axes[2] * 50]
    pca_frame.points = o3d.utility.Vector3dVector(points)
    edges = [[0, 1], [0, 2], [0, 3]]  # Connect origin to PCA axes
    pca_frame.lines = o3d.utility.Vector2iVector(edges)
    pca_frame.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # RGB colors

# print
    print(aligned_axes)
    # Visualize all geometries
    o3d.visualization.draw_geometries(
        [pcd_head, pcd_planned, pcd_drawn, sphere_planned, sphere_drawn, pca_frame]
    )

def calculate_centroid(points):
    """
    Calculate the centroid of a point cloud.

    :param points: NumPy array of point cloud vertices.
    :return: NumPy array representing the centroid coordinates.
    """
    return np.mean(points, axis=0)


def calculate_euclidean_distance(point_a, point_b, pca_axes):
    """
    Calculate the Euclidean distance and its components along PCA axes.

    :param point_a: NumPy array representing the coordinates of point A.
    :param point_b: NumPy array representing the coordinates of point B.
    :param pca_axes: 3x3 NumPy array representing PCA axes.
    :return: Tuple containing the Euclidean distance and the components along PCA axes.
    """
    diff = point_b - point_a
    components = np.dot(diff, pca_axes.T)  # Project difference onto PCA axes
    distance = np.linalg.norm(components)
    distance = round(distance, 2)
    components = np.round(components, 2)
    return distance, components

# Open the main results CSV file to write Euclidean distances and their XYZ components
def process_group(NEP_planned_folder, NEP_drawn_folder, NEP_planned_proj_folder, results_path, results_projection_path, codes, suffixes, points_head):
    with open(results_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Remove ".obj" from suffixes for cleaner column names
        clean_suffixes = [suffix.replace(".obj", "") for suffix in suffixes]

        # Create header: ['Code', 'suffix', 'suffix_X', 'suffix_Y', 'suffix_Z', ...]
        header = ['Code'] + [f"{suffix}{metric}" for suffix in clean_suffixes for metric in ['', '_X', '_Y', '_Z']]
        writer.writerow(header)

        # Open the projection results CSV file (for PCA-projected planned curves)
        with open(results_projection_path, mode='w', newline='') as proj_file:
            proj_writer = csv.writer(proj_file)
            proj_writer.writerow(header)

            # Process each subject or case identified by code
            for code in codes:
                results = [f"{code}"] # Row for raw planned vs drawn distances
                results_proj = [f"{code}"] # Row for projected planned vs drawn distances

                # Process each suffix (each curve type, e.g., "MF_L.obj", )
                for suffix in suffixes:
                    # Find matching file paths for planned, drawn, and projected planned spheres
                    planned_files = glob.glob(os.path.join(NEP_planned_folder, f"*{suffix}"))
                    drawn_files = glob.glob(os.path.join(NEP_drawn_folder, f"*{code}*{suffix}"))
                    planned_files_proj = glob.glob(os.path.join(NEP_planned_proj_folder, f"*{suffix}"))

                    # Only proceed if both planned and drawn files are found
                    if planned_files and drawn_files:
                        for NEP_planned_file in planned_files:
                            for NEP_drawn_file in drawn_files:
                                # Load OBJ files as point clouds
                                points_planned = load_obj_as_pointcloud(NEP_planned_file)
                                points_drawn = load_obj_as_pointcloud(NEP_drawn_file)

                                # Compute PCA axes from the head reference and align to world axes
                                pca_axes = compute_pca_axes(points_head)
                                aligned_axes = align_pca_to_world(pca_axes)

                                # Compute centroids of planned and drawn spheres
                                centroid_planned = calculate_centroid(points_planned)
                                centroid_drawn = calculate_centroid(points_drawn)

                                # Calculate Euclidean distance and PCA components between centroids
                                euclidean_distance, components = calculate_euclidean_distance(centroid_planned, centroid_drawn, aligned_axes)

                                # Append distance and its XYZ components to the results list
                                results.extend([euclidean_distance, components[0], components[1], components[2]])

                                # Optional visualization for inspection/debugging
                                #visualize_with_open3d(points_head, points_planned, points_drawn, centroid_planned, centroid_drawn)

                                # If projected planned files are found, calculate distances similarly
                                if planned_files_proj:
                                    for NEP_planned_proj_file in planned_files_proj:
                                        points_planned_proj = load_obj_as_pointcloud(NEP_planned_proj_file)
                                        centroid_planned_proj = calculate_centroid(points_planned_proj)
                                        euclidean_distance_proj, components_proj = calculate_euclidean_distance(
                                            centroid_planned_proj, centroid_drawn, aligned_axes)

                                        # Append projected distances and components
                                        results_proj.extend(
                                            [euclidean_distance_proj, components_proj[0], components_proj[1], components_proj[2]]
                                        )
                                else:
                                    # If no projection available, add empty fields
                                    results_proj.extend(["", "", "", ""])
                    else:
                        # If no matching planned or drawn file, fill in empty fields
                        results.extend(["", "", "", ""])
                        results_proj.extend(["", "", "", ""])

                # Write the collected data to the result CSV files
                writer.writerow(results)
                proj_writer.writerow(results_proj)
def identify_outliers(results_path_yl, results_path_kg, outlier_path, threshold):
    """
    Compare YL and KG results and identify outliers.

    :param results_path_yl: Path to YL results CSV file.
    :param results_path_kg: Path to KG results CSV file.
    :param outlier_path: Path to save the outlier CSV file.
    :param threshold: Threshold for identifying outliers (default is 5.0 for Euclidean distance).
    """
    # Load the CSV files
    yl_data = pd.read_csv(results_path_yl)
    kg_data = pd.read_csv(results_path_kg)

    # Ensure the two dataframes have the same structure
    if list(yl_data.columns) != list(kg_data.columns):
        raise ValueError("The YL and KG results files have different structures.")

    # Compare the two results and calculate the absolute difference for each metric
    differences = yl_data.copy()
    for col in yl_data.columns[1:]:  # Skip the first column (Code)
        differences[col] = np.abs(yl_data[col] - kg_data[col])

    # Identify outliers based on the threshold
    outliers = differences[(differences.iloc[:, 1:] > threshold).any(axis=1)]

    # Save outliers to CSV
    outliers.to_csv(outlier_path, index=False)

    print(f"Outliers saved to {outlier_path}")


def main():
    # Define paths and parameters
    NEP_planned_folder = "C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Auswertung/Planned OBJ/"
    NEP_drawn_folder_YL = "C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Auswertung/OBJ_ALL_YL/"
    NEP_drawn_folder_KG = "C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Auswertung/OBJ_ALL_KG/"
    NEP_planned_proj_folder = "C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Auswertung/Planned OBJ/Projected NEP/"
    head_file = "C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Auswertung/Planned OBJ/Skin.obj"

    points_head = load_obj_as_pointcloud(head_file)

    # List of codes to search for
    codes = [
        "35CZI", "24KOR", "16NRE", "15VQS", "58IOV", "92CFA", "69UBX", "31JVF", "67VBD", "68LJB", "83JKF",
        "28AHB", "58ZFD", "04PNB", "35UIW", "23YDT", "12UMO", "10LSN", "47BPO", "70XVR", "47OIF", "50FVH",
        "23PCY", "48UDQ", "66WNA", "34KBM", "40KCT", "76QOV", "83STF", "15OOK", "66CYE", "56ZCK", "29KYY",
        "01TAS", "36IWS", "26VCB", "76OOD", "11RHZ"
    ]

    # Search for files ending with different suffixes
    suffixes = ["MF_L.obj", "MF_R.obj", "IF_L.obj", "IF_R.obj", "SF_L.obj", "SF_R.obj"]

    # File paths for results
    results_path_YL = 'C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Daten/Primary EP/results_YL.csv'
    results_projection_path_YL = 'C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Daten/Primary EP/results_projection_YL.csv'
    results_path_KG = 'C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Daten/Primary EP/results_KG.csv'
    results_projection_path_KG = 'C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Daten/Primary EP//results_projection_KG.csv'


    # Process both YL and KG groups
    process_group(NEP_planned_folder, NEP_drawn_folder_YL, NEP_planned_proj_folder, results_path_YL, results_projection_path_YL, codes, suffixes, points_head)
    process_group(NEP_planned_folder, NEP_drawn_folder_KG, NEP_planned_proj_folder, results_path_KG, results_projection_path_KG, codes, suffixes, points_head)

    #filter out outliner to redo the evaluation later

    outlier_path = 'C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Daten/Primary EP/outliers.csv'
    identify_outliers(results_path_YL, results_path_KG, outlier_path,1.0)
    outlier_path_proj = 'C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Daten/Primary EP/outliers_projection.csv'
    identify_outliers(results_projection_path_YL, results_projection_path_KG, outlier_path_proj,1.0)

if __name__ == "__main__":
    main()
