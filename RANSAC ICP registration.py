# this code is calculate the transformation matrix from scanned heads to the planned heads
# and apply them and save the transformed/registered heads
# to map all scanned heads to the planned head in blender.

# Author: Yao Li

import os
import glob
import open3d as o3d
import numpy as np
import copy
import shutil

# Functions to compute FPFH features
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsampling with a voxel size of %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    print(":: Estimating normal vectors.")
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    print(":: Computing FPFH feature.")
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return pcd_down, fpfh

# define preprocessing function
def prepare_dataset(source, target, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source_down, target_down, source_fpfh, target_fpfh

# RANSAC global registration funcitons
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: Apply global RANSAC registration with distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,  # mutual_filter 设置为 True
        distance_threshold,  # max_correspondence_distance
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

# load obj file as point cloud
def load_obj_as_pointcloud(filename, number_of_points=10000):
    mesh = o3d.io.read_triangle_mesh(filename)
    return mesh.sample_points_poisson_disk(number_of_points)

# visualization of the registration result
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])  # source pc is in red
    target_temp.paint_uniform_color([0, 1, 0])  # target pc is in green
    source_temp.transform(transformation)  # apply transformation matrix to the source
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      window_name="RANSAC Initial Registration",
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def compute_normals(pcd, radius=0.1):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
    return pcd

def remove_outliers(pcd, x_threshold, y_threshold, z_threshold):
    points = np.asarray(pcd.points)

    # get values in each direction
    x_values = points[:, 0]
    y_values = points[:, 1]
    z_values = points[:, 2]

    # calculate the threshold in each direciton
    x_threshold_value = np.percentile(x_values, x_threshold)
    y_threshold_value = np.percentile(y_values, y_threshold)
    z_threshold_value = np.percentile(z_values, z_threshold)

    # Filtering points larger than the threshold in all directions
    filtered_indices = np.where(
        (x_values > x_threshold_value) &
        (y_values > y_threshold_value) &
        (z_values > z_threshold_value)
    )[0]

    # Create new point clouds based on filtered indexes
    filtered_pcd = pcd.select_by_index(filtered_indices)
    return filtered_pcd

def transform_obj_and_save(input_filename, output_filename, transformation):
    mesh = o3d.io.read_triangle_mesh(input_filename)
    mesh.transform(transformation)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(output_filename, mesh)



def save_transformation(transformation, filename):
    # Speichert die Transformation in einer Datei.
    np.savetxt(filename, transformation)



# set path and parameter



def process_file(source_file,target_file):
    voxel_size = 5  # set voxel size

    # load source and target point cloud
    source = load_obj_as_pointcloud(source_file, number_of_points=10000)
    target = load_obj_as_pointcloud(target_file, number_of_points=10000)

    #Preprocessing point clouds (downsampling, normal estimation and FPFH feature extraction)
    source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)

    # excute RANSAC coarse registration
    ransac_result = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print("RANSAC Transformation Matrix:\n", ransac_result.transformation)

    # visualize RANSAC registration results
    # draw_registration_result(source, target, ransac_result.transformation)
    # apply transformation matrix to the source
    source.transform(ransac_result.transformation)

    #end of the coarse registration

    max_correspondence_distance =0.5
    trans_init = np.eye(4)


    source_filtered = remove_outliers(source, 40,0,30)  #
    target_filtered = remove_outliers(target, 40,0,30)


    source = compute_normals(source_filtered, radius=10)  # calculate source normal
    target = compute_normals(target_filtered, radius=10)  # calculate target normal



    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1.000000e-06,
                                                          relative_rmse=1.000000e-06,
                                                          max_iteration=100000))

    print("Transformation is:", ransac_result.transformation*reg_p2l.transformation)
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, max_correspondence_distance, reg_p2l.transformation)
    print("Fitness:", evaluation.fitness)
    print("Inlier RMSE:", evaluation.inlier_rmse)

    # print combined transformation matrix
    combined_transformation = reg_p2l.transformation @ ransac_result.transformation
    print("Combined Transformation Matrix:\n", combined_transformation)

    # save transformed obj file
    transformed_file = source_file.replace(".obj", "_TRANSFORMED.obj")
    transform_obj_and_save(source_file, transformed_file, combined_transformation)

    # save transformation matrix in to txt. file
    transformation_filename = source_file.replace(".obj", "_TRANSFORMATION.txt")
    save_transformation(combined_transformation, transformation_filename)

    # visualize the registration result
    draw_registration_result(source, target, reg_p2l.transformation)


directory_path = 'C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/3D Scannen/OBJ/'
obj_list = [os.path.basename(f) for f in glob.glob(os.path.join(directory_path, "*.obj"))]

for obj_file in obj_list:
    print(f"Processing {obj_file}...")
    source_file = os.path.join(directory_path, obj_file)
    target_file = os.path.join(
        'C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/3D Scannen/Plannung/',
        "FINAL_PLANNED_EVALUATION.obj")
    process_file(source_file, target_file)
