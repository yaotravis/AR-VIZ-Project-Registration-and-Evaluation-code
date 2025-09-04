# This code is to calculate the soft tissue thickness by calculating the average shortest distance
# from each points of nerve courses and salivery glands to the head surface
#Author: Yao Li
import os
import numpy as np
import open3d as o3d

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

def visualize_head_nearest_point(source_point, nearest_point, head_points):
    """
    Visualize:
    - a single source point (red sphere)
    - its nearest point on the head (blue sphere)
    - the full head point cloud (gray)
    - a green line between them
    """
    # Red: source point
    sphere_source = o3d.geometry.TriangleMesh.create_sphere(radius=0.6)
    sphere_source.translate(source_point)
    sphere_source.paint_uniform_color([1, 0, 0])  # red

    # Blue: nearest point on head
    sphere_target = o3d.geometry.TriangleMesh.create_sphere(radius=0.6)
    sphere_target.translate(nearest_point)
    sphere_target.paint_uniform_color([0, 0, 1])  # blue

    # Green line
    line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([source_point, nearest_point]),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    line.colors = o3d.utility.Vector3dVector([[0, 1, 0]])

    # Full head point cloud
    pcd_head = o3d.geometry.PointCloud()
    pcd_head.points = o3d.utility.Vector3dVector(head_points)
    pcd_head.paint_uniform_color([0.5, 0.5, 0.5])

    o3d.visualization.draw_geometries([sphere_source, sphere_target, line, pcd_head])
def compute_average_shortest_distance(source_points, target_points):
    """
    For each point in source_points, compute its nearest distance to target_points.
    Returns: mean_distance, list of all distances
    """
    if len(source_points) == 0 or len(target_points) == 0:
        return None, []

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)
    kdtree = o3d.geometry.KDTreeFlann(target_pcd)

    distances = []
    for pt in source_points:
        [_, idx, dist2] = kdtree.search_knn_vector_3d(pt.tolist(), 1)
        dist = np.sqrt(dist2[0])
        distances.append(dist)
        nearest = target_points[idx[0]]
        #visualize
        # visualize_head_nearest_point(pt, nearest, target_points)

    mean_dist = np.mean(distances)
    return mean_dist




# === User Paths ===
skin_file = r"C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Auswertung/Planned OBJ/Skin.obj"
original_structure_folder = r"C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Auswertung/Planned OBJ/Original Structures/"

suffixes=["OG_NC_L.obj","OG_NC_R.obj","OG_P_L.obj","OG_P_R.obj","OG_SUB_L.obj","OG_SUB_R.obj"]
# === Run thickness computation ===
points_head = load_obj_as_pointcloud(skin_file)


for suffix in suffixes:
    structure_file = os.path.join(original_structure_folder, suffix)
    points_structure=load_obj_as_pointcloud(structure_file)
    print(suffix,":", compute_average_shortest_distance(points_structure,points_head)
)


