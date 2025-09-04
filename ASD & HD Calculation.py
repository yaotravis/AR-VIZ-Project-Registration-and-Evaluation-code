# this code is to calculate the HD and ASD of nerve course and salivary glands
# Author: Yao Li
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import pandas as pd
import glob
import os
from sklearn.cluster import DBSCAN

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


def calculate_average_surface_distance(points_a, points_b):
    """
    Calculate the Average Surface Distance (ASD) between two point clouds.

    :param points_a: NumPy array of points from curve A.
    :param points_b: NumPy array of points from curve B.
    :return: The Average Surface Distance (ASD) between the two curves.
    """
    # Create KDTree for each point cloud
    tree_a = KDTree(points_a)
    tree_b = KDTree(points_b)

    # For each point in A, find the closest point in B
    distances_a_to_b, _ = tree_b.query(points_a)

    # For each point in B, find the closest point in A
    distances_b_to_a, _ = tree_a.query(points_b)

    # Combine distances and calculate the average
    total_distance = np.sum(distances_a_to_b) + np.sum(distances_b_to_a)
    total_points = len(points_a) + len(points_b)
    asd = total_distance / total_points
    return round(asd, 2)



def calculate_hausdorff_distance(points_a, points_b):
    """
    Calculate the Hausdorff Distance (HD) between two point clouds.

    :param points_a: NumPy array of points from curve A.
    :param points_b: NumPy array of points from curve B.
    :return: The Hausdorff Distance (HD) between the two curves.
    """
    # Create KDTree for each point cloud
    tree_a = KDTree(points_a)
    tree_b = KDTree(points_b)

    # For each point in A, find the closest point in B and get the maximum distance
    distances_a_to_b, _ = tree_b.query(points_a)
    max_distance_a_to_b = np.max(distances_a_to_b)

    # For each point in B, find the closest point in A and get the maximum distance
    distances_b_to_a, _ = tree_a.query(points_b)
    max_distance_b_to_a = np.max(distances_b_to_a)

    # Hausdorff distance is the maximum of these two distances
    hausdorff_distance = max(max_distance_a_to_b, max_distance_b_to_a)

    return round(hausdorff_distance, 2)

# Load the two curves from their OBJ files
def process_group(curve_planned_folder, curve_drawn_folder, codes, suffixes, output_file):
    """
    Process a group of data and write the results to an Excel file.

    :param curve_planned_folder: Path to the folder containing planned OBJ files.
    :param curve_drawn_folder: Path to the folder containing drawn OBJ files.
    :param codes: List of codes to process.
    :param suffixes: List of suffixes to search for.
    :param output_file: Path to the output Excel file.
    """
    # Initialize a dictionary to store results
    results = {"Code": []}
    for suffix in suffixes:
        clean_suffix = suffix.replace(".obj", "")  # Remove .obj from suffix
        results[f"{clean_suffix} ASD"] = []
        results[f"{clean_suffix} HD"] = []

    incomplete_codes = []

    for code in codes:
        results["Code"].append(code)
        available_suffixes = set()

        for suffix in suffixes:
            # Find corresponding planned and drawn files with the same suffix
            planned_files = glob.glob(os.path.join(curve_planned_folder, f"*{suffix}"))
            drawn_files = glob.glob(os.path.join(curve_drawn_folder, f"*{code}*{suffix}"))

            if planned_files and drawn_files:
                for curve_planned_file in planned_files:
                    for curve_drawn_file in drawn_files:
                        points_planned = load_obj_as_pointcloud(curve_planned_file)
                        points_drawn = load_obj_as_pointcloud(curve_drawn_file)

                        # Calculate the Average Surface Distance (ASD) and Hausdorff Distance (HD)
                        asd = calculate_average_surface_distance(points_planned, points_drawn)
                        hd = calculate_hausdorff_distance(points_planned, points_drawn)

                        results[f"{suffix.replace('.obj', '')} ASD"].append(round(asd, 2))
                        results[f"{suffix.replace('.obj', '')} HD"].append(round(hd, 2))
                        available_suffixes.add(suffix)
                        break
                    break
            else:
                results[f"{suffix.replace('.obj', '')} ASD"].append(None)
                results[f"{suffix.replace('.obj', '')} HD"].append(None)

        if len(available_suffixes) < len(suffixes):
            incomplete_codes.append(code)

    # Convert results dictionary to a DataFrame
    df = pd.DataFrame(results)

    # Write to Excel file
    df.to_excel(output_file, index=False)
    print(f"Results written to {output_file}")

    # Print incomplete codes
    if incomplete_codes:
        print("\nCodes with incomplete suffixes:")
        for code in incomplete_codes:
            print(code)

def compare_groups(y_results, kg_results, suffixes, output_file, threshold=1.0):
    """
    Compare two groups (YL and KG) and find values exceeding the threshold.

    :param y_results: DataFrame containing results for YL group.
    :param kg_results: DataFrame containing results for KG group.
    :param suffixes: List of suffixes used for comparison.
    :param threshold: The threshold value for detecting outliers.
    :param output_file: The file to save outlier results.
    """
    outliers = []
    # Iterate over each row in YL and KG DataFrames
    for idx, yl_row in y_results.iterrows():
        code = yl_row["Code"]
        kg_row = kg_results[kg_results["Code"] == code]

        if not kg_row.empty:
            kg_row = kg_row.iloc[0]  # Get the matching row in KG

            for suffix in suffixes:
                clean_suffix = suffix.replace(".obj", "")
                asd_col = f"{clean_suffix} ASD"
                hd_col = f"{clean_suffix} HD"

                # Compare ASD
                if asd_col in y_results.columns and asd_col in kg_results.columns:
                    yl_asd = yl_row[asd_col]
                    kg_asd = kg_row[asd_col]
                    if pd.notna(yl_asd) and pd.notna(kg_asd) and abs(yl_asd - kg_asd) > threshold:
                        outliers.append({
                            "Code": code,
                            "Suffix": asd_col,
                            "YL Value": yl_asd,
                            "KG Value": kg_asd,
                            "Difference": round(abs(yl_asd - kg_asd), 2)
                        })

                # Compare HD
                if hd_col in y_results.columns and hd_col in kg_results.columns:
                    yl_hd = yl_row[hd_col]
                    kg_hd = kg_row[hd_col]
                    if pd.notna(yl_hd) and pd.notna(kg_hd) and abs(yl_hd - kg_hd) > threshold:
                        outliers.append({
                            "Code": code,
                            "Suffix": hd_col,
                            "YL Value": yl_hd,
                            "KG Value": kg_hd,
                            "Difference": round(abs(yl_hd - kg_hd), 2)
                        })

    # Save outliers to an Excel file
    if outliers:
        outliers_df = pd.DataFrame(outliers)
        outliers_df.to_excel(output_file, index=False)
        print(f"Outliers written to {output_file}")
    else:
        print("No outliers found.")
def main():
    # Define paths and parameters
    curve_planned_folder = "C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Auswertung/Planned OBJ/Original Structures/"
    curve_drawn_folder_YL = "C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Auswertung/OBJ_ALL_YL/"
    curve_drawn_folder_KG = "C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Auswertung/OBJ_ALL_KG/"

    # List of codes to search for
    codes = [
        "35CZI", "24KOR", "16NRE", "15VQS", "58IOV", "92CFA", "69UBX", "31JVF", "67VBD", "68LJB", "83JKF",
        "28AHB", "58ZFD", "04PNB", "35UIW", "23YDT", "12UMO", "10LSN", "47BPO", "70XVR", "47OIF", "50FVH",
        "23PCY", "48UDQ", "66WNA", "34KBM", "40KCT", "76QOV", "83STF", "15OOK", "66CYE", "56ZCK", "29KYY",
        "01TAS", "36IWS", "26VCB", "76OOD", "11RHZ"
    ]

    # Search for files ending with different suffixes
    suffixes = ["NC_L.obj", "NC_R.obj", "SUB_L.obj", "SUB_R.obj", "P_L.obj", "P_R.obj"]


    # Process YL group
    print("Processing YL group...")
    yl_output_file = "C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Daten/Secondary EP/results_YL.xlsx"
    process_group(
        curve_planned_folder, curve_drawn_folder_YL, codes, suffixes, yl_output_file
    )

    # Process KG group
    print("\nProcessing KG group...")
    kg_output_file = "C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Daten/Secondary EP/results_KG.xlsx"
    process_group(
        curve_planned_folder, curve_drawn_folder_KG, codes, suffixes, kg_output_file
    )

    # Load results for comparison
    yl_results = pd.read_excel(yl_output_file)
    kg_results = pd.read_excel(kg_output_file)

    # Compare groups and find outliers
    print("\nComparing YL and KG groups...")
    outlier_output_file= "C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Daten/Secondary EP/comparison_outliers.xlsx"

    compare_groups(yl_results, kg_results, suffixes, outlier_output_file,threshold=1.0)


if __name__ == "__main__":
    main()