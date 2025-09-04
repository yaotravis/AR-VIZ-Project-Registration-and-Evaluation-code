# This script reformats the Excel file: instead of one row per subject,
# each subject is represented by two rows, with each row corresponding
# to one side under different methods.
# Author: Yao Li

import pandas as pd

# Read the original data
# Read the original data
file_path = 'C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Daten/Evaluation AR-Viz.xlsx'  # Replace with your file path
df = pd.read_excel(file_path, sheet_name='Total Endpoints')

# Define the updated new columns
new_columns = [
    'ID', 'GROUP', 'METHOD', 'SIDE', 'ORDER', 'NASA_TLX', 'Likert', 'Time',
    'MF', 'MF_X', 'MF_Y', 'MF_Z', 'IF', 'IF_X', 'IF_Y', 'IF_Z',
    'SF', 'SF_X', 'SF_Y', 'SF_Z',
    'ST_MF',	'ST_IF',	'ST_SF',

    'Relative_D1','Relative_D2',
    'NC_ASD', 'NC_HD', 'SUB_ASD', 'SUB_HD','P_ASD', 'P_HD',
    'ST_NC', 'ST_P','ST_SUB',

    'SA_NC','SA_P',	'SA_SUB',
    'VL_NC','VL_P',	'VL_SUB',
    'MD_NC','MD_P',	'MD_SUB',	


    'MF_PROJ', 'MF_X_PROJ', 'MF_Y_PROJ', 'MF_Z_PROJ',
    'IF_PROJ', 'IF_X_PROJ', 'IF_Y_PROJ', 'IF_Z_PROJ',
    'SF_PROJ', 'SF_X_PROJ', 'SF_Y_PROJ', 'SF_Z_PROJ',
    'Sup_L1', 'Sup_L2', 'Sup_L3', 'Sup_L4', 'Sup_L5', 'Sup_L6', 'Sup_L7', 'Sup_L8', 'Sup_L9', 'Sup_L10',
    'Twin_L1', 'Twin_L2', 'Twin_L3', 'Twin_L4', 'Twin_L5', 'Twin_L6', 'Twin_L7', 'Twin_L8', 'Twin_L9', 'Twin_L10',
]

# Create an empty DataFrame to store the transformed data
new_df = pd.DataFrame(columns=new_columns)

# Iterate through the original DataFrame and split rows into separate ones based on method (Sup or Twin)
for _, row in df.iterrows():
    # Extract common data for both methods
    code = row['Code']
    group = row['Gruppe']

    # Process for the first method (Erste Methode)
    method_1 = row['Erste Methode']
    side_1 = 'R' if 'rechts' in method_1.lower() else 'L'
    method_1_name = 'Sup' if 'sup' in method_1.lower() else 'Twin'
    new_row_1 = {
        'ID': code,
        'GROUP': group,
        'METHOD': method_1_name,
        'SIDE': side_1,
        'ORDER': 1,
        'NASA_TLX': row[f'{method_1_name}_NASA_TLX'],
        'Likert': row[f'{method_1_name}_Likert'],
        'Time': str(row[f'{method_1_name}_Dauer']) + ' AM',
        'MF': row[f'MF_{side_1}'],
        'MF_X': row[f'MF_{side_1}_X'],
        'MF_Y': row[f'MF_{side_1}_Y'],
        'MF_Z': row[f'MF_{side_1}_Z'],
        'IF': row[f'IF_{side_1}'],
        'IF_X': row[f'IF_{side_1}_X'],
        'IF_Y': row[f'IF_{side_1}_Y'],
        'IF_Z': row[f'IF_{side_1}_Z'],
        'SF': row[f'SF_{side_1}'],
        'SF_X': row[f'SF_{side_1}_X'],
        'SF_Y': row[f'SF_{side_1}_Y'],
        'SF_Z': row[f'SF_{side_1}_Z'],
        'ST_MF': row[f'ST_MF_{side_1}'],
        'ST_IF': row[f'ST_IF_{side_1}'],
        'ST_SF': row[f'ST_SF_{side_1}'],
        'Relative_D1': row[f'Relative_D1_{side_1}'],
        'Relative_D2': row[f'Relative_D2_{side_1}'],
        'NC_ASD': row[f'NC_{side_1} ASD'],
        'NC_HD': row[f'NC_{side_1} HD'],
        'SUB_ASD': row[f'SUB_{side_1} ASD'],
        'SUB_HD': row[f'SUB_{side_1} HD'],
        'P_ASD': row[f'P_{side_1} ASD'],
        'P_HD': row[f'P_{side_1} HD'],

        'ST_NC': row[f'ST_NC_{side_1}'],
        'ST_P': row[f'ST_P_{side_1}'],
        'ST_SUB': row[f'ST_SUB_{side_1}'],

        'SA_NC': row[f'SA_NC_{side_1}'],
        'SA_P': row[f'SA_P_{side_1}'],
        'SA_SUB': row[f'SA_SUB_{side_1}'],
        'VL_NC': row[f'VL_NC_{side_1}'],
        'VL_P': row[f'VL_P_{side_1}'],
        'VL_SUB': row[f'VL_SUB_{side_1}'],
        'MD_NC': row[f'MD_NC_{side_1}'],
        'MD_P': row[f'MD_P_{side_1}'],
        'MD_SUB': row[f'MD_SUB_{side_1}'],

        'MF_PROJ': row[f'MF_{side_1}_PROJ'],
        'MF_X_PROJ': row[f'MF_{side_1}_X_PROJ'],
        'MF_Y_PROJ': row[f'MF_{side_1}_Y_PROJ'],
        'MF_Z_PROJ': row[f'MF_{side_1}_Z_PROJ'],
        'IF_PROJ': row[f'IF_{side_1}_PROJ'],
        'IF_X_PROJ': row[f'IF_{side_1}_X_PROJ'],
        'IF_Y_PROJ': row[f'IF_{side_1}_Y_PROJ'],
        'IF_Z_PROJ': row[f'IF_{side_1}_Z_PROJ'],
        'SF_PROJ': row[f'SF_{side_1}_PROJ'],
        'SF_X_PROJ': row[f'SF_{side_1}_X_PROJ'],
        'SF_Y_PROJ': row[f'SF_{side_1}_Y_PROJ'],
        'SF_Z_PROJ': row[f'SF_{side_1}_Z_PROJ'],
        'Sup_L1': row['Sup_L1'] if method_1_name == 'Sup' else None,
        'Sup_L2': row['Sup_L2'] if method_1_name == 'Sup' else None,
        'Sup_L3': row['Sup_L3'] if method_1_name == 'Sup' else None,
        'Sup_L4': row['Sup_L4'] if method_1_name == 'Sup' else None,
        'Sup_L5': row['Sup_L5'] if method_1_name == 'Sup' else None,
        'Sup_L6': row['Sup_L6'] if method_1_name == 'Sup' else None,
        'Sup_L7': row['Sup_L7'] if method_1_name == 'Sup' else None,
        'Sup_L8': row['Sup_L8'] if method_1_name == 'Sup' else None,
        'Sup_L9': row['Sup_L9'] if method_1_name == 'Sup' else None,
        'Sup_L10': row['Sup_L10'] if method_1_name == 'Sup' else None,
        'Twin_L1': row['Twin_L1'] if method_1_name == 'Twin' else None,
        'Twin_L2': row['Twin_L2'] if method_1_name == 'Twin' else None,
        'Twin_L3': row['Twin_L3'] if method_1_name == 'Twin' else None,
        'Twin_L4': row['Twin_L4'] if method_1_name == 'Twin' else None,
        'Twin_L5': row['Twin_L5'] if method_1_name == 'Twin' else None,
        'Twin_L6': row['Twin_L6'] if method_1_name == 'Twin' else None,
        'Twin_L7': row['Twin_L7'] if method_1_name == 'Twin' else None,
        'Twin_L8': row['Twin_L8'] if method_1_name == 'Twin' else None,
        'Twin_L9': row['Twin_L9'] if method_1_name == 'Twin' else None,
        'Twin_L10': row['Twin_L10'] if method_1_name == 'Twin' else None,
    }
    new_df = new_df.append(new_row_1, ignore_index=True)

    # Process for the second method (Zweite Methode)
    method_2 = row['Zweite Methode']
    side_2 = 'R' if 'rechts' in method_2.lower() else 'L'
    method_2_name = 'Sup' if 'sup' in method_2.lower() else 'Twin'
    new_row_2 = {
        'ID': code,
        'GROUP': group,
        'METHOD': method_2_name,
        'SIDE': side_2,
        'ORDER': 2,
        'NASA_TLX': row[f'{method_2_name}_NASA_TLX'],
        'Likert': row[f'{method_2_name}_Likert'],
        'Time': str(row[f'{method_2_name}_Dauer']) + ' AM',
        'MF': row[f'MF_{side_2}'],
        'MF_X': row[f'MF_{side_2}_X'],
        'MF_Y': row[f'MF_{side_2}_Y'],
        'MF_Z': row[f'MF_{side_2}_Z'],
        'IF': row[f'IF_{side_2}'],
        'IF_X': row[f'IF_{side_2}_X'],
        'IF_Y': row[f'IF_{side_2}_Y'],
        'IF_Z': row[f'IF_{side_2}_Z'],
        'SF': row[f'SF_{side_2}'],
        'SF_X': row[f'SF_{side_2}_X'],
        'SF_Y': row[f'SF_{side_2}_Y'],
        'SF_Z': row[f'SF_{side_2}_Z'],
        'ST_MF': row[f'ST_MF_{side_2}'],
        'ST_IF': row[f'ST_IF_{side_2}'],
        'ST_SF': row[f'ST_SF_{side_2}'],

        'SA_NC': row[f'SA_NC_{side_2}'],
        'SA_P': row[f'SA_P_{side_2}'],
        'SA_SUB': row[f'SA_SUB_{side_2}'],
        'VL_NC': row[f'VL_NC_{side_2}'],
        'VL_P': row[f'VL_P_{side_2}'],
        'VL_SUB': row[f'VL_SUB_{side_2}'],
        'MD_NC': row[f'MD_NC_{side_2}'],
        'MD_P': row[f'MD_P_{side_2}'],
        'MD_SUB': row[f'MD_SUB_{side_2}'],

        'Relative_D1': row[f'Relative_D1_{side_2}'],
        'Relative_D2': row[f'Relative_D2_{side_2}'],
        'NC_ASD': row[f'NC_{side_2} ASD'],
        'NC_HD': row[f'NC_{side_2} HD'],
        'SUB_ASD': row[f'SUB_{side_2} ASD'],
        'SUB_HD': row[f'SUB_{side_2} HD'],
        'P_ASD': row[f'P_{side_2} ASD'],
        'P_HD': row[f'P_{side_2} HD'],

        'ST_NC': row[f'ST_NC_{side_2}'],
        'ST_P': row[f'ST_P_{side_2}'],
        'ST_SUB': row[f'ST_SUB_{side_2}'],

        'MF_PROJ': row[f'MF_{side_2}_PROJ'],
        'MF_X_PROJ': row[f'MF_{side_2}_X_PROJ'],
        'MF_Y_PROJ': row[f'MF_{side_2}_Y_PROJ'],
        'MF_Z_PROJ': row[f'MF_{side_2}_Z_PROJ'],
        'IF_PROJ': row[f'IF_{side_2}_PROJ'],
        'IF_X_PROJ': row[f'IF_{side_2}_X_PROJ'],
        'IF_Y_PROJ': row[f'IF_{side_2}_Y_PROJ'],
        'IF_Z_PROJ': row[f'IF_{side_2}_Z_PROJ'],
        'SF_PROJ': row[f'SF_{side_2}_PROJ'],
        'SF_X_PROJ': row[f'SF_{side_2}_X_PROJ'],
        'SF_Y_PROJ': row[f'SF_{side_2}_Y_PROJ'],
        'SF_Z_PROJ': row[f'SF_{side_2}_Z_PROJ'],
        'Sup_L1': row['Sup_L1'] if method_2_name == 'Sup' else None,
        'Sup_L2': row['Sup_L2'] if method_2_name == 'Sup' else None,
        'Sup_L3': row['Sup_L3'] if method_2_name == 'Sup' else None,
        'Sup_L4': row['Sup_L4'] if method_2_name == 'Sup' else None,
        'Sup_L5': row['Sup_L5'] if method_2_name == 'Sup' else None,
        'Sup_L6': row['Sup_L6'] if method_2_name == 'Sup' else None,
        'Sup_L7': row['Sup_L7'] if method_2_name == 'Sup' else None,
        'Sup_L8': row['Sup_L8'] if method_2_name == 'Sup' else None,
        'Sup_L9': row['Sup_L9'] if method_2_name == 'Sup' else None,
        'Sup_L10': row['Sup_L10'] if method_2_name == 'Sup' else None,
        'Twin_L1': row['Twin_L1'] if method_2_name == 'Twin' else None,
        'Twin_L2': row['Twin_L2'] if method_2_name == 'Twin' else None,
        'Twin_L3': row['Twin_L3'] if method_2_name == 'Twin' else None,
        'Twin_L4': row['Twin_L4'] if method_2_name == 'Twin' else None,
        'Twin_L5': row['Twin_L5'] if method_2_name == 'Twin' else None,
        'Twin_L6': row['Twin_L6'] if method_2_name == 'Twin' else None,
        'Twin_L7': row['Twin_L7'] if method_2_name == 'Twin' else None,
        'Twin_L8': row['Twin_L8'] if method_2_name == 'Twin' else None,
        'Twin_L9': row['Twin_L9'] if method_2_name == 'Twin' else None,
        'Twin_L10': row['Twin_L10'] if method_2_name == 'Twin' else None,
    }
    new_df = new_df.append(new_row_2, ignore_index=True)

# Save the transformed DataFrame to a new Excel file
output_path = 'C:/Users/yli84/Uniklinik RWTH Aachen/Paper AR VIZ - Documents/General/Daten/Evaluation AR-Viz_transformed_data.xlsx'
new_df.to_excel(output_path, sheet_name='Final',index=False)

# Provide the output path for user reference
output_path