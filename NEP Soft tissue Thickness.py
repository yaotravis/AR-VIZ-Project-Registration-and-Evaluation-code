# This code is to calculate the soft tissue thickness by calculating the average shortest distance
# from each points of nerve courses and salivery glands to the head surface
#Author: Yao Li
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import glob
from scipy.spatial.distance import euclidean


def load_obj_as_pointcloud(filename):
    points = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                points.append([x, y, z])
    return np.array(points)


def calculate_centroid(points):
    return np.mean(points, axis=0)


def compare_pointclouds_by_suffix(NEP_planned_folder, NEP_planned_proj_folder, suffixes):
    results = []
    for suffix in suffixes:
        planned_file = glob.glob(os.path.join(NEP_planned_folder, f"*{suffix}"))
        planned_proj_file = glob.glob(os.path.join(NEP_planned_proj_folder, f"*{suffix}"))

        if planned_file and planned_proj_file:
            pc1 = load_obj_as_pointcloud(planned_file[0])
            pc2 = load_obj_as_pointcloud(planned_proj_file[0])
            centroid_pc1 = calculate_centroid(pc1)
            centroid_pc2 = calculate_centroid(pc2)
            d = euclidean(centroid_pc1, centroid_pc2)
            results.append((suffix, d))
        else:
            results.append((suffix, np.nan))  # Use NaN for missing

    return np.array(results, dtype=object)


def main():
    NEP_planned_folder = "C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Auswertung/OBJ_ALL_YL/Planned OBJ/"
    NEP_planned_proj_folder = "C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Auswertung/Planned OBJ/Nearest  Projected NEP_Blender/"
    suffixes = ["MF_L.obj", "MF_R.obj", "IF_L.obj", "IF_R.obj", "SF_L.obj", "SF_R.obj"]

    results_array = compare_pointclouds_by_suffix(NEP_planned_folder, NEP_planned_proj_folder, suffixes)

    print("\nSummary Table:")
    for row in results_array:
        print(f"{row[0]}: {row[1]}")


    valid_distances = [r[1] for r in results_array if not np.isnan(r[1])]
    overall_mean = np.mean(valid_distances) if valid_distances else np.nan

    # Calculate the average by category
    group_dict = {"MF": [], "IF": [], "SF": []}
    for suffix, value in results_array:
        if np.isnan(value):
            continue
        if "MF" in suffix:
            group_dict["MF"].append(value)
        elif "IF" in suffix:
            group_dict["IF"].append(value)
        elif "SF" in suffix:
            group_dict["SF"].append(value)

    mf_mean = np.mean(group_dict["MF"]) if group_dict["MF"] else np.nan
    if_mean = np.mean(group_dict["IF"]) if group_dict["IF"] else np.nan
    sf_mean = np.mean(group_dict["SF"]) if group_dict["SF"] else np.nan

    print(f"\nMF mean: {mf_mean}")
    print(f"IF mean: {if_mean}")
    print(f"SF mean: {sf_mean}")
    print(f"\nOverall mean distance: {overall_mean}")


if __name__ == "__main__":
    main()
