# this code is to calculate the relative distance between the NEPs
# Author: Yao Li

import os
import glob
import numpy as np
import open3d as o3d
import csv
import numpy as np
from scipy.spatial.distance import pdist, squareform



from scipy.spatial.distance import euclidean


def load_obj_as_pointcloud(filename):
    """Load vertex coordinates from OBJ file as NumPy array."""
    points = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                points.append([x, y, z])
    return np.array(points)


def calculate_centroid(points):
    """Return centroid of a point cloud."""
    return np.mean(points, axis=0)


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def get_centroids(NEP_planned_folder, NEP_drawn_folder, code, prefix_list, side_suffix):
    """
    Return centroids in the order [SF, IF, MF] for a given side.
    """
    centroids_plan = [None] * 3
    centroids_drawn = [None] * 3

    for i, prefix in enumerate(prefix_list):
        suffix = f"{prefix}_{side_suffix}.obj"
        planned_file = glob.glob(os.path.join(NEP_planned_folder, f"*{suffix}"))
        drawn_file = glob.glob(os.path.join(NEP_drawn_folder, f"*{code}*{suffix}"))

        if planned_file and drawn_file:
            plan = calculate_centroid(load_obj_as_pointcloud(planned_file[0]))
            drawn = calculate_centroid(load_obj_as_pointcloud(drawn_file[0]))
            centroids_plan[i] = plan
            centroids_drawn[i] = drawn

    return centroids_plan, centroids_drawn


def process_group_sequential_distance_diff(curve_planned_folder, curve_drawn_folder, codes, output_csv):
    prefix_order = ["SF", "IF", "MF"]

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Code', 'diff_D1_L', 'diff_D2_L', 'diff_D1_R', 'diff_D2_R'])

        for code in codes:
            diff_D1_L, diff_D2_L = '', ''
            diff_D1_R, diff_D2_R = '', ''

            # LEFT
            plan_L, drawn_L = get_centroids(curve_planned_folder, curve_drawn_folder, code, prefix_order, "L")
            if all(x is not None for x in plan_L) and all(x is not None for x in drawn_L):
                D1_plan = euclidean_distance(plan_L[0], plan_L[1])  # SF–IF
                D2_plan = euclidean_distance(plan_L[1], plan_L[2])  # IF–MF
                D1_drawn = euclidean_distance(drawn_L[0], drawn_L[1])
                D2_drawn = euclidean_distance(drawn_L[1], drawn_L[2])
                diff_D1_L = round(abs(D1_drawn - D1_plan), 2)
                diff_D2_L = round(abs(D2_drawn - D2_plan), 2)

            # RIGHT
            plan_R, drawn_R = get_centroids(curve_planned_folder, curve_drawn_folder, code, prefix_order, "R")
            if all(x is not None for x in plan_R) and all(x is not None for x in drawn_R):
                D1_plan = euclidean_distance(plan_R[0], plan_R[1])
                D2_plan = euclidean_distance(plan_R[1], plan_R[2])
                D1_drawn = euclidean_distance(drawn_R[0], drawn_R[1])
                D2_drawn = euclidean_distance(drawn_R[1], drawn_R[2])
                diff_D1_R = round(abs(D1_drawn - D1_plan), 2)
                diff_D2_R = round(abs(D2_drawn - D2_plan), 2)

            writer.writerow([code, diff_D1_L, diff_D2_L, diff_D1_R, diff_D2_R])


def main():
    NEP_planned_folder = "C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Auswertung/OBJ_ALL_YL/Planned OBJ/"
    NEP_drawn_folder_YL = "C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Auswertung/OBJ_ALL_YL/"
    NEP_drawn_folder_KG = "C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Auswertung/OBJ_ALL_KG/"

    codes = [
        "35CZI", "24KOR", "16NRE", "15VQS", "58IOV", "92CFA", "69UBX", "31JVF", "67VBD", "68LJB", "83JKF",
        "28AHB", "58ZFD", "04PNB", "35UIW", "23YDT", "12UMO", "10LSN", "47BPO", "70XVR", "47OIF", "50FVH",
        "23PCY", "48UDQ", "66WNA", "34KBM", "40KCT", "76QOV", "83STF", "15OOK", "66CYE", "56ZCK", "29KYY",
        "01TAS", "36IWS", "26VCB", "76OOD", "11RHZ"
    ]

    process_group_sequential_distance_diff(
        NEP_planned_folder,
        NEP_drawn_folder_YL,
        codes,
        'C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Daten/Secondary EP/sequential_distance_diff_YL.csv'
    )

    process_group_sequential_distance_diff(
        NEP_planned_folder,
        NEP_drawn_folder_KG,
        codes,
        'C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Daten/Secondary EP/sequential_distance_diff_KG.csv'
    )


if __name__ == "__main__":
    main()
