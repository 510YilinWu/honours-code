import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# File path
file_path = '/Users/yilinwu/Desktop/honours data/Yilin-Honours/Subject/Model Angles/06/12/Test04.csv'
# Read CSV
df = pd.read_csv(file_path, skiprows=5, sep=r"\s+|,", engine="python")

# Column indices for joints
column_indices = {
    "YW:RWristAnglesXZY": [2, 3, 4],
    "YW:RShoAnglesYZY": [5, 6, 7],
    "YW:RShoAnglesXZY": [8, 9, 10],
    "YW:RElbAnglesXZY": [11, 12, 13],
    "YW:LWristAnglesXZY": [14, 15, 16],
    "YW:LShoAnglesYZY": [17, 18, 19],
    "YW:LShoAnglesXZY": [20, 21, 22],
    "YW:LElbAnglesXZY": [23, 24, 25],
}

import matplotlib.pyplot as plt

# Visualize all joints with subplots
fig, axes = plt.subplots(len(column_indices), 1, figsize=(10, 20), sharex=True)

for i, (joint_name, indices) in enumerate(column_indices.items()):
    joint_data = df.iloc[:, indices]
    axes[i].plot(joint_data, label=['X', 'Y', 'Z'])
    axes[i].set_title(joint_name)
    axes[i].legend()
    axes[i].grid()

plt.xlabel('Time')
plt.tight_layout()
plt.show()

def perform_pca_and_calculate_ldlj(df, column_start, column_end):
    # Select columns based on adjustable indices
    selected_columns = df.iloc[:, column_start:column_end]

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_columns)

    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)

    # Loadings for PC1
    loadings_pc1 = pca.components_[0]
    feature_names = selected_columns.columns
    pc1_loadings_df = pd.DataFrame({
        'Feature': feature_names,
        'PC1 Loading': loadings_pc1
    }).sort_values(by='PC1 Loading', key=abs, ascending=False)

    print("\nContribution of each input feature to PC1:")
    print(pc1_loadings_df)

    # Print contributions
    print("\n--- PC1 Composition ---")
    for _, row in pc1_loadings_df.iterrows():
        direction = "positive" if row['PC1 Loading'] > 0 else "negative"
        strength = abs(row['PC1 Loading'])
        contribution = (
            "strong" if strength >= 0.35 else
            "moderate" if strength >= 0.2 else
            "weak" if strength >= 0.1 else
            "negligible"
        )
        # Map feature to joint name and axis
        for joint_name, indices in column_indices.items():
            if row['Feature'] in df.columns[indices]:
                axis = ['X', 'Y', 'Z'][df.columns[indices].tolist().index(row['Feature'])]
                print(f"Joint: {joint_name}, Axis: {axis}, Direction: {direction}, Contribution: {contribution}")
                break

    # Create a DataFrame for PCA results
    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])

    # Create a DataFrame for explained variance ratio
    explained_variance_df = pd.DataFrame({
        'Principal Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'Explained Variance (%)': pca.explained_variance_ratio_ * 100
    })

    # Print the table
    print(explained_variance_df)

    # Extract PC1 and PC2 values
    pc1 = pca_df['PC1']
    pc2 = pca_df['PC2']

    # Time variable (assuming equally spaced time intervals)
    time = np.arange(len(pc1))

    # Calculate speed, acceleration, and jerk for PC1 and PC2
    speed_pc1 = np.gradient(pc1, time)
    speed_pc2 = np.gradient(pc2, time)

    acceleration_pc1 = np.gradient(speed_pc1, time)
    acceleration_pc2 = np.gradient(speed_pc2, time)

    jerk_pc1 = np.gradient(acceleration_pc1, time)
    jerk_pc2 = np.gradient(acceleration_pc2, time)

    # Define start and end indices for the time window
    Tstart_idx = 0
    Tend_idx = len(time) - 1

    # Calculate LDLJ for PC1
    jerk_squared_integral_pc1 = np.trapezoid(jerk_pc1[Tstart_idx:Tend_idx]**2, time[Tstart_idx:Tend_idx])
    vpeak_pc1 = speed_pc1[Tstart_idx:Tend_idx].max()
    dimensionless_jerk_pc1 = (jerk_squared_integral_pc1 * (time[Tend_idx] - time[Tstart_idx])**3) / (vpeak_pc1**2)
    LDLJ_pc1 = -np.log(dimensionless_jerk_pc1)

    # Calculate LDLJ for PC2
    jerk_squared_integral_pc2 = np.trapezoid(jerk_pc2[Tstart_idx:Tend_idx]**2, time[Tstart_idx:Tend_idx])
    vpeak_pc2 = speed_pc2[Tstart_idx:Tend_idx].max()
    dimensionless_jerk_pc2 = (jerk_squared_integral_pc2 * (time[Tend_idx] - time[Tstart_idx])**3) / (vpeak_pc2**2)
    LDLJ_pc2 = -np.log(dimensionless_jerk_pc2)

    # Print LDLJ values
    print(f"LDLJ for PC1: {LDLJ_pc1}")
    print(f"LDLJ for PC2: {LDLJ_pc2}")

    # Plot PC1 and PC2 values
    plt.figure(figsize=(10, 6))
    plt.plot(time, pc1, label='PC1', color='blue')
    plt.plot(time, pc2, label='PC2', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Principal Component Value')
    plt.title('PC1 and PC2 Values Over Time')
    plt.legend()
    plt.grid()
    plt.show()


    def extract_movement(df, column_indices):
        movement = {}

        # Right Arm Movements
        movement["Right Arm"] = {
            "Shoulder Abduction": df.iloc[:, column_indices["YW:RShoAnglesXZY"][2]],  # Z (+)
            "Shoulder Flexion": -df.iloc[:, column_indices["YW:RShoAnglesXZY"][0]],  # X (-)
            "Shoulder Inner Rotation": df.iloc[:, column_indices["YW:RShoAnglesXZY"][1]],  # Y (+)
            "Elbow Flexion": df.iloc[:, column_indices["YW:RElbAnglesXZY"][0]],  # X (+)
            "Pronation": df.iloc[:, column_indices["YW:RElbAnglesXZY"][2]],  # Z (+)
            "Wrist Flexion": df.iloc[:, column_indices["YW:RWristAnglesXZY"][0]],  # X (+)
            "Ulnar Deviation": -df.iloc[:, column_indices["YW:RWristAnglesXZY"][1]],  # Y (-)
        }

        # Left Arm Movements
        movement["Left Arm"] = {
            "Shoulder Abduction": -df.iloc[:, column_indices["YW:LShoAnglesXZY"][2]],  # Z (-)
            "Shoulder Flexion": df.iloc[:, column_indices["YW:LShoAnglesXZY"][0]],  # X (+)
            "Shoulder Inner Rotation": -df.iloc[:, column_indices["YW:LShoAnglesXZY"][1]],  # Y (-)
            "Elbow Flexion": df.iloc[:, column_indices["YW:LElbAnglesXZY"][0]],  # X (+)
            "Pronation": -df.iloc[:, column_indices["YW:LElbAnglesXZY"][2]],  # Z (-)
            "Wrist Flexion": df.iloc[:, column_indices["YW:LWristAnglesXZY"][0]],  # X (+)
            "Ulnar Deviation": df.iloc[:, column_indices["YW:LWristAnglesXZY"][1]],  # Y (+)
        }

        return movement

    # Example usage
    movement_data = extract_movement(df, column_indices)

    # Visualize movements for the Right Arm
    fig_right, axes_right = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    movement_names_right = list(movement_data["Right Arm"].keys())
    movement_values_right = list(movement_data["Right Arm"].values())

    # Determine global min and max for y-axis across all movements
    global_min_right = min([values.min() for values in movement_values_right])
    global_max_right = max([values.max() for values in movement_values_right])

    for i in range(3):  # Rows
        for j in range(3):  # Columns
            movement_idx = i * 3 + j
            if movement_idx < len(movement_names_right):
                axes_right[i, j].plot(movement_values_right[movement_idx], label=movement_names_right[movement_idx])
                axes_right[i, j].set_title(f"Right Arm - {movement_names_right[movement_idx]}")
                axes_right[i, j].legend()
                axes_right[i, j].grid()
                axes_right[i, j].set_ylim(global_min_right, global_max_right)  # Set consistent y-axis limits
            else:
                axes_right[i, j].axis('off')  # Turn off unused subplots

    plt.tight_layout()
    plt.show()

    # Visualize movements for the Left Arm
    fig_left, axes_left = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    movement_names_left = list(movement_data["Left Arm"].keys())
    movement_values_left = list(movement_data["Left Arm"].values())

    # Determine global min and max for y-axis across all movements
    global_min_left = min([values.min() for values in movement_values_left])
    global_max_left = max([values.max() for values in movement_values_left])

    for i in range(3):  # Rows
        for j in range(3):  # Columns
            movement_idx = i * 3 + j
            if movement_idx < len(movement_names_left):
                axes_left[i, j].plot(movement_values_left[movement_idx], label=movement_names_left[movement_idx])
                axes_left[i, j].set_title(f"Left Arm - {movement_names_left[movement_idx]}")
                axes_left[i, j].legend()
                axes_left[i, j].grid()
                axes_left[i, j].set_ylim(global_min_left, global_max_left)  # Set consistent y-axis limits
            else:
                axes_left[i, j].axis('off')  # Turn off unused subplots

    plt.tight_layout()
    plt.show()

# Example usage
perform_pca_and_calculate_ldlj(df, column_start=2, column_end=15)



