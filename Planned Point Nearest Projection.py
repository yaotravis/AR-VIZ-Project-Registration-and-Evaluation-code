# optional
# this code is to project the planned NEPs to the nearest points of head.
# in case anyone does not want to use blender.
# Author: Yao Li


import os
import glob
import numpy as np
import open3d as o3d
def load_obj_as_mesh(filename):
    """Load OBJ file and return it as a mesh."""
    mesh = o3d.io.read_triangle_mesh(filename)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.5, 0.5, 0.5])  #
    return mesh
def load_obj_as_pointcloud(filename):
    """
    Manually load OBJ file and extract vertex (v) data as point cloud.
    """
    points = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                points.append([x, y, z])
    return np.array(points)

def calculate_centroid(points):
    """
    Calculate the centroid of a point cloud.

    :param points: NumPy array of point cloud vertices.
    :return: NumPy array representing the centroid coordinates.
    """
    return np.mean(points, axis=0)
def place_sphere_at_nearest_point(NEP_file, head_pcd, radius=1.0):
    """
    Locate the center point of the NEP, then find the nearest point in head_pcd and place the sphere.
    """
    curve_points = load_obj_as_pointcloud(NEP_file)
    if len(curve_points) == 0:
        return None

    # 1. calculate centroid of the planned NEPs
    centroid = np.mean(curve_points, axis=0)

    # 2. Find the point in head_pcd closest to the center of mass.
    head_tree = o3d.geometry.KDTreeFlann(head_pcd)
    [_, idx, _] = head_tree.search_knn_vector_3d(centroid, 1)
    nearest_point = np.asarray(head_pcd.points)[idx[0]]

    # 3. Create and move the sphere to the nearest point
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    sphere.translate(nearest_point)
    sphere.paint_uniform_color([1, 0, 1])  # purple
    return sphere



def generate_spheres_and_save_separately(head_file, curve_planned_folder, save_folder):
    """
    Generate the nearest-point sphere for each NEP and save each as an .obj file, naming it NP_MF_L.obj.
    """
    suffix_bases = ["MF", "IF", "SF"]
    sides = ['L', 'R']

    # Create Save Path
    os.makedirs(save_folder, exist_ok=True)

    # Read the head point cloud
    head_points = load_obj_as_pointcloud(head_file)
    head_pcd = o3d.geometry.PointCloud()
    head_pcd.points = o3d.utility.Vector3dVector(head_points)

    for base in suffix_bases:
        for side in sides:
            suffix = f"{base}_{side}.obj"
            file_path = os.path.join(curve_planned_folder, suffix)
            if not os.path.exists(file_path):
                print(f" file not found：{file_path}")
                continue
            sphere = place_sphere_at_nearest_point(file_path, head_pcd, radius=1.0)
            if sphere is not None:
                save_path = os.path.join(save_folder, f"NP_{base}_{side}.obj")
                o3d.io.write_triangle_mesh(save_path, sphere)
                print(f" save spheres：{save_path}")
            else:
                print(f" cannot create spheres for {suffix} ")

head_file = r"C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Auswertung/OBJ_ALL_YL/Planned OBJ/Skin.obj"
NEP_planned_folder = r"C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Auswertung/OBJ_ALL_YL/Planned OBJ/"

save_folder = r"C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Auswertung/Planned OBJ/Nearest Points/"
generate_spheres_and_save_separately(head_file, NEP_planned_folder, save_folder)