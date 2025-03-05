# import pandas as pd 
# import numpy as np
# # import matplotlib.pyplot as plt

# csvPath = '/Users/yilinwu/Desktop/honours data/ModelOutput.csv'
# df = pd.read_csv(csvPath)
# print(df.head())
# print(df.shape)


# # Define variable names
# variables = [
#     "LWJC", "RWJC", "Thorax1", "Thorax2", "Thorax3", "ThoraxO", 
#     "leftHand1", "leftHand2", "leftHand3", "leftHandO", 
#     "rightHand1", "rightHand2", "rightHand3", "rightHandO"
# ]

# # Generate mock data (replace with actual data input)
# data = {var: np.random.rand(100, 3) for var in variables}  # Assuming 100 time points, 3D (x, y, z)

# # Save each variable separately
# for var, values in data.items():
#     df = pd.DataFrame(values, columns=["x", "y", "z"])
#     df.to_csv(f"Yilin_{var}.csv", index=False)
#     print(f"Saved: Yilin_{var}.csv")


# import pandas as pd
# import numpy as np
# import os

# # Define file path
# csvPath = '/Users/yilinwu/Desktop/honours data/ModelOutput.csv'

# # Read the CSV file
# df = pd.read_csv(csvPath)
# print(df.head())
# print(f"Dataset shape: {df.shape}")

# # Extract unique subjects dynamically
# subjects = set(col.split(":")[0] for col in df.columns if ":" in col)
# print(f"Subjects found: {subjects}")

# # Define variable names (assumed known)
# variables = [
#     "LWJC", "RWJC", "Thorax1", "Thorax2", "Thorax3", "ThoraxO", 
#     "leftHand1", "leftHand2", "leftHand3", "leftHandO", 
#     "rightHand1", "rightHand2", "rightHand3", "rightHandO"
# ]

# # Define suffixes for different coordinate sets
# coordinate_suffixes = ["", "'", "''"]  # "", "'", "''" represent different versions

# # Create an output directory
# output_dir = "Processed_Data"
# os.makedirs(output_dir, exist_ok=True)

# # Keep track of metadata columns
# metadata_columns = ["Frame", "Sub Frame"]  # Ensure these exist in your dataset

# # Process each subject separately
# for subject in subjects:
#     for var in variables:
#         for suffix in coordinate_suffixes:
#             col_name_x = f"{subject}:{var}_X{suffix}"
#             col_name_y = f"{subject}:{var}_Y{suffix}"
#             col_name_z = f"{subject}:{var}_Z{suffix}"

#             # Check if columns exist
#             if col_name_x in df.columns and col_name_y in df.columns and col_name_z in df.columns:
#                 df_var = df[metadata_columns + [col_name_x, col_name_y, col_name_z]].copy()
#                 df_var.columns = metadata_columns + ["x", "y", "z"]  # Standardize column names
                
#                 # Save as CSV
#                 suffix_clean = suffix.replace("'", "p")  # Avoid issues with file naming
#                 file_path = os.path.join(output_dir, f"{subject}_{var}{suffix_clean}.csv")
#                 df_var.to_csv(file_path, index=False)
#                 print(f"Saved: {file_path}")

# print("Processing completed.")

# print(output_dir)



# structure of the dictionary 
# Dictionary: Subject name, Variable name (e.g., "LWJC", "RWJC") > Coordinate set name ("", "'", "''") > DataFrame with columns x, y, z
# import pandas as pd
# import numpy as np
# import os

# # Define file path
# csvPath = '/Users/yilinwu/Desktop/honours data/ModelOutput.csv'

# # Read the CSV file
# df = pd.read_csv(csvPath)
# print(df.head())
# print(f"Dataset shape: {df.shape}")

# # Extract unique subjects dynamically
# subjects = set(col.split(":")[0] for col in df.columns if ":" in col)
# print(f"Subjects found: {subjects}")

# # Define variable names (assumed known)
# variables = [
#     "LWJC", "RWJC", "Thorax1", "Thorax2", "Thorax3", "ThoraxO", 
#     "leftHand1", "leftHand2", "leftHand3", "leftHandO", 
#     "rightHand1", "rightHand2", "rightHand3", "rightHandO"
# ]

# # Define suffixes for different coordinate sets
# coordinate_suffixes = ["", "'", "''"]  # "", "'", "''" represent different versions

# # Keep track of metadata columns
# metadata_columns = ["Frame", "Sub Frame"]  # Ensure these exist in your dataset

# # Initialize dictionary to store processed data
# processed_data = {}

# # Process each subject separately
# for subject in subjects:
#     processed_data[subject] = {}
    
#     for var in variables:
#         processed_data[subject][var] = {}
        
#         for suffix in coordinate_suffixes:
#             col_name_x = f"{subject}:{var}_X{suffix}"
#             col_name_y = f"{subject}:{var}_Y{suffix}"
#             col_name_z = f"{subject}:{var}_Z{suffix}"

#             # Check if columns exist
#             if col_name_x in df.columns and col_name_y in df.columns and col_name_z in df.columns:
#                 df_var = df[metadata_columns + [col_name_x, col_name_y, col_name_z]].copy()
#                 df_var.columns = metadata_columns + ["x", "y", "z"]  # Standardize column names
                
#                 # Store in dictionary
#                 processed_data[subject][var][suffix] = df_var

# print("Processing completed.")

# import pprint
# pprint.pprint(processed_data)  # Limits depth to avoid too much output

import pandas as pd
import numpy as np
import os

# Define file path
csvPath = '/Users/yilinwu/Desktop/honours data/ModelOutput.csv'

# Read the CSV file
df = pd.read_csv(csvPath)
print(df.head())
print(f"Dataset shape: {df.shape}")

# Extract unique subjects dynamically
subjects = set(col.split(":")[0] for col in df.columns if ":" in col)
print(f"Subjects found: {subjects}")

# Define variable names (assumed known)
variables = [
    "LWJC", "RWJC", "Thorax1", "Thorax2", "Thorax3", "ThoraxO", 
    "leftHand1", "leftHand2", "leftHand3", "leftHandO", 
    "rightHand1", "rightHand2", "rightHand3", "rightHandO"
]

# Define suffixes for different coordinate sets
coordinate_suffixes = ["", "'", "''"]  # "", "'", "''" represent different versions

# Keep track of metadata columns
metadata_columns = ["Frame", "Sub Frame"]  # Ensure these exist in your dataset

# Initialize dictionary to store processed data
processed_data = {}

# Process each subject separately
for subject in subjects:
    processed_data[subject] = {}
    
    for var in variables:
        processed_data[subject][var] = {}
        
        for suffix in coordinate_suffixes:
            col_name_x = f"{subject}:{var}_X{suffix}"
            col_name_y = f"{subject}:{var}_Y{suffix}"
            col_name_z = f"{subject}:{var}_Z{suffix}"

            # Check if columns exist
            if col_name_x in df.columns and col_name_y in df.columns and col_name_z in df.columns:
                df_var = df[metadata_columns + [col_name_x, col_name_y, col_name_z]].copy()
                df_var.columns = metadata_columns + ["X", "Y", "Z"]  # Standardize column names for position

                # Calculate velocity and acceleration
                df_var['VX'] = df_var['X'].diff() / df_var['Frame'].diff()  # Velocity in mm/s
                df_var['VY'] = df_var['Y'].diff() / df_var['Frame'].diff()  # Velocity in mm/s
                df_var['VZ'] = df_var['Z'].diff() / df_var['Frame'].diff()  # Velocity in mm/s
                
                df_var['AX'] = df_var['VX'].diff() / df_var['Frame'].diff()  # Acceleration in mm/s²
                df_var['AY'] = df_var['VY'].diff() / df_var['Frame'].diff()  # Acceleration in mm/s²
                df_var['AZ'] = df_var['VZ'].diff() / df_var['Frame'].diff()  # Acceleration in mm/s²

                # Store in dictionary with position, velocity, and acceleration using X, Y, Z notation
                processed_data[subject][var][suffix] = {
                    'Position': df_var[['X', 'Y', 'Z']],
                    'Velocity': df_var[['VX', 'VY', 'VZ']],
                    'Acceleration': df_var[['AX', 'AY', 'AZ']]
                }

print("Processing completed.")


