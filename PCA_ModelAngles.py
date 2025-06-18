import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# File path
FILE_PATH = '/Users/yilinwu/Desktop/honours data/Yilin-Honours/Subject/Model Angles/06/12/Test04.csv'

# Column indices for joints
COLUMN_INDICES = {
    "YW:RWristAnglesXZY": [2, 3, 4],
    "YW:RShoAnglesYZY": [5, 6, 7],
    "YW:RShoAnglesXZY": [8, 9, 10],
    "YW:RElbAnglesXZY": [11, 12, 13],
    "YW:LWristAnglesXZY": [14, 15, 16],
    "YW:LShoAnglesYZY": [17, 18, 19],
    "YW:LShoAnglesXZY": [20, 21, 22],
    "YW:LElbAnglesXZY": [23, 24, 25],
}

def load_data(file_path):
    """
    Load the CSV file into a DataFrame.
    """
    return pd.read_csv(file_path, skiprows=5, sep=r"\s+|,", engine="python")

def visualize_joints(df, COLUMN_INDICES):
    """
    Visualize all joints with subplots, adding unit 'deg' to the labels.
    """
    _, axes = plt.subplots(len(COLUMN_INDICES), 1, figsize=(10, 20), sharex=True)
    for i, (joint_name, indices) in enumerate(COLUMN_INDICES.items()):
        joint_data = df.iloc[:, indices]
        axes[i].plot(joint_data, label=['X (deg)', 'Y (deg)', 'Z (deg)'])
        axes[i].set_title(joint_name)
        axes[i].legend()
        axes[i].grid()
    plt.xlabel('Time')
    plt.tight_layout()
    plt.show()

def perform_pca(df, column_start, column_end):
    """
    Perform PCA on selected columns and return PCA results and explained variance.
    """
    selected_columns = df.iloc[:, column_start:column_end]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_columns)
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_ * 100
    return pca, pca_result, explained_variance, selected_columns

def analyze_pc1_contributions(pca, selected_columns, COLUMN_INDICES, df):
    """
    Analyze contributions of each input feature to PC1.
    """
    loadings_pc1 = pca.components_[0]
    feature_names = []
    for col in selected_columns.columns:
        for joint_name, indices in COLUMN_INDICES.items():
            if col in df.columns[indices]:
                axis = ['X', 'Y', 'Z'][df.columns[indices].tolist().index(col)]
                feature_names.append(f"{joint_name} - {axis}")
                break
    pc1_loadings_df = pd.DataFrame({
        'Feature': feature_names,
        'PC1 Loading': loadings_pc1
    }).sort_values(by='PC1 Loading', key=abs, ascending=False)

    pc1_loadings_df['Direction'] = pc1_loadings_df['PC1 Loading'].apply(lambda x: "positive" if x > 0 else "negative")
    pc1_loadings_df['Strength'] = pc1_loadings_df['PC1 Loading'].abs().apply(
        lambda x: "strong" if x >= 0.35 else "moderate" if x >= 0.2 else "weak" if x >= 0.1 else "negligible"
    )

    print("\nContribution of each input feature to PC1:")
    print(pc1_loadings_df)


def print_explained_variance(pca):
    """
    Print the explained variance ratio for each principal component.
    """
    explained_variance_df = pd.DataFrame({
        'Principal Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'Explained Variance (%)': pca.explained_variance_ratio_ * 100
    })

    print("\nExplained Variance Ratio:")
    print(explained_variance_df)

def plot_principal_components(pca_result):
    """
    Plot PC1 and PC2 values over time.
    """
    time = np.arange(len(pca_result[:, 0]))
    plt.figure(figsize=(10, 6))
    plt.plot(time, pca_result[:, 0], label='PC1', color='blue')
    plt.plot(time, pca_result[:, 1], label='PC2', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Principal Component Value')
    plt.title('PC1 and PC2 Values Over Time')
    plt.legend()
    plt.grid()
    plt.show()

def calculate_ldlj(pca_result):
    """
    Calculate LDLJ for PC1 and PC2.
    """
    pc1 = pca_result[:, 0]
    pc2 = pca_result[:, 1]
    time = np.arange(len(pc1))
    speed_pc1 = np.gradient(pc1, time)
    speed_pc2 = np.gradient(pc2, time)
    acceleration_pc1 = np.gradient(speed_pc1, time)
    acceleration_pc2 = np.gradient(speed_pc2, time)
    jerk_pc1 = np.gradient(acceleration_pc1, time)
    jerk_pc2 = np.gradient(acceleration_pc2, time)
    Tstart_idx = 0
    Tend_idx = len(time) - 1
    jerk_squared_integral_pc1 = np.trapezoid(jerk_pc1[Tstart_idx:Tend_idx]**2, time[Tstart_idx:Tend_idx])
    vpeak_pc1 = speed_pc1[Tstart_idx:Tend_idx].max()
    dimensionless_jerk_pc1 = (jerk_squared_integral_pc1 * (time[Tend_idx] - time[Tstart_idx])**3) / (vpeak_pc1**2)
    LDLJ_pc1 = -np.log(dimensionless_jerk_pc1)
    jerk_squared_integral_pc2 = np.trapezoid(jerk_pc2[Tstart_idx:Tend_idx]**2, time[Tstart_idx:Tend_idx])
    vpeak_pc2 = speed_pc2[Tstart_idx:Tend_idx].max()
    dimensionless_jerk_pc2 = (jerk_squared_integral_pc2 * (time[Tend_idx] - time[Tstart_idx])**3) / (vpeak_pc2**2)
    LDLJ_pc2 = -np.log(dimensionless_jerk_pc2)
    return LDLJ_pc1, LDLJ_pc2

def extract_movement(df, COLUMN_INDICES):
    """
    Extract movement data for right and left arms.
    """
    movement = {
        "Right Arm": {
            "Shoulder Abduction": df.iloc[:, COLUMN_INDICES["YW:RShoAnglesXZY"][2]],
            "Shoulder Flexion": -df.iloc[:, COLUMN_INDICES["YW:RShoAnglesXZY"][0]],
            "Shoulder Inner Rotation": df.iloc[:, COLUMN_INDICES["YW:RShoAnglesXZY"][1]],
            "Elbow Flexion": df.iloc[:, COLUMN_INDICES["YW:RElbAnglesXZY"][0]],
            "Pronation": df.iloc[:, COLUMN_INDICES["YW:RElbAnglesXZY"][2]],
            "Wrist Flexion": df.iloc[:, COLUMN_INDICES["YW:RWristAnglesXZY"][0]],
            "Ulnar Deviation": -df.iloc[:, COLUMN_INDICES["YW:RWristAnglesXZY"][1]],
        },
        "Left Arm": {
            "Shoulder Abduction": -df.iloc[:, COLUMN_INDICES["YW:LShoAnglesXZY"][2]],
            "Shoulder Flexion": df.iloc[:, COLUMN_INDICES["YW:LShoAnglesXZY"][0]],
            "Shoulder Inner Rotation": -df.iloc[:, COLUMN_INDICES["YW:LShoAnglesXZY"][1]],
            "Elbow Flexion": df.iloc[:, COLUMN_INDICES["YW:LElbAnglesXZY"][0]],
            "Pronation": -df.iloc[:, COLUMN_INDICES["YW:LElbAnglesXZY"][2]],
            "Wrist Flexion": df.iloc[:, COLUMN_INDICES["YW:LWristAnglesXZY"][0]],
            "Ulnar Deviation": df.iloc[:, COLUMN_INDICES["YW:LWristAnglesXZY"][1]],
        }
    }
    return movement

def visualize_movements(movement_data, arm_name):
    """
    Visualize movements for the specified arm.
    """
    movements = movement_data[arm_name]
    _, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    movement_names = list(movements.keys())
    movement_values = list(movements.values())
    global_min = min([values.min() for values in movement_values])
    global_max = max([values.max() for values in movement_values])
    for i in range(3):
        for j in range(3):
            movement_idx = i * 3 + j
            if movement_idx < len(movement_names):
                axes[i, j].plot(movement_values[movement_idx], label=movement_names[movement_idx])
                axes[i, j].set_title(f"{arm_name} - {movement_names[movement_idx]}")
                axes[i, j].legend()
                axes[i, j].grid()
                axes[i, j].set_ylim(global_min, global_max)
            else:
                axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to execute all tasks.
    """
    df = load_data(FILE_PATH)
    visualize_joints(df, COLUMN_INDICES)
    pca, pca_result, _, selected_columns = perform_pca(df, column_start=2, column_end=15)
    analyze_pc1_contributions(pca, selected_columns, COLUMN_INDICES, df)
    print_explained_variance(pca)
    plot_principal_components(pca_result)
    LDLJ_pc1, LDLJ_pc2 = calculate_ldlj(pca_result)
    # print(f"LDLJ for PC1: {LDLJ_pc1}")
    # print(f"LDLJ for PC2: {LDLJ_pc2}")
    movement_data = extract_movement(df, COLUMN_INDICES)
    visualize_movements(movement_data, "Right Arm")
    visualize_movements(movement_data, "Left Arm")

if __name__ == "__main__":
    main()




