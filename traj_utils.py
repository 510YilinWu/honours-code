import pandas as pd
import os
import datetime
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np

def CSV_To_traj_data(file_path):
    """
    Reads a CSV file containing trajectory data, processes it, and extracts specific columns based on predefined indices.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        dict: A dictionary containing extracted data for each prefix.
    """
    # Read the file, skipping the first 5 rows
    df = pd.read_csv(
        file_path,
        skiprows=4,
        sep=r"\s+|,",  # Split on whitespace or commas
        engine="python"
    )

    # Extract Frame from column 1 (second column)
    Frame = df.iloc[:, 0]

    # Calculate the time for each frame based on a 200 Hz frame capture rate
    time = Frame / 200  # Time in seconds

    # Define a function to extract X, Y, Z, VX, VY, VZ, AX, AY, AZ, MX, MVX, MAX for a given prefix and column indices
    def extract_columns(prefix, indices):
        return {
            f"{prefix}_X": df.iloc[:, indices[0]],
            f"{prefix}_Y": df.iloc[:, indices[1]],
            f"{prefix}_Z": df.iloc[:, indices[2]],
            f"{prefix}_VX": df.iloc[:, indices[3]],
            f"{prefix}_VY": df.iloc[:, indices[4]],
            f"{prefix}_VZ": df.iloc[:, indices[5]],
            f"{prefix}_AX": df.iloc[:, indices[6]],
            f"{prefix}_AY": df.iloc[:, indices[7]],
            f"{prefix}_AZ": df.iloc[:, indices[8]],
            f"{prefix}_radial_pos": df.iloc[:, indices[9]], # radial position; the distance from the origin 
            f"{prefix}_radial_vel": df.iloc[:, indices[10]], # radial velocity; How fast is the distance from the origin changing over time
            f"{prefix}_radial_acc": df.iloc[:, indices[11]],# radial acceleration
        }

    # Define the column indices for each prefix
    column_indices = {
        "C7": [2, 3, 4, 74, 75, 76, 146, 147, 148, 218, 242, 266],
        "T10": [5, 6, 7, 77, 78, 79, 149, 150, 151, 219, 243, 267],
        "CLAV": [8, 9, 10, 80, 81, 82, 152, 153, 154, 220, 244, 268],
        "STRN": [11, 12, 13, 83, 84, 85, 155, 156, 157, 221, 245, 269],
        "LSHO": [14, 15, 16, 86, 87, 88, 158, 159, 160, 222, 246, 270],
        "LUPA": [17, 18, 19, 89, 90, 91, 161, 162, 163, 223, 247, 271],
        "LUPB": [20, 21, 22, 92, 93, 94, 164, 165, 166, 224, 248, 272],
        "LUPC": [23, 24, 25, 95, 96, 97, 167, 168, 169, 225, 249, 273],
        "LELB": [26, 27, 28, 98, 99, 100, 170, 171, 172, 226, 250, 274],
        "LMEP": [29, 30, 31, 101, 102, 103, 173, 174, 175, 227, 251, 275],
        "LWRA": [32, 33, 34, 104, 105, 106, 176, 177, 178, 228, 252, 276],
        "LWRB": [35, 36, 37, 107, 108, 109, 179, 180, 181, 229, 253, 277],
        "LFRA": [38, 39, 40, 110, 111, 112, 182, 183, 184, 230, 254, 278],
        "LFIN": [41, 42, 43, 113, 114, 115, 185, 186, 187, 231, 255, 279],
        "RSHO": [44, 45, 46, 116, 117, 118, 188, 189, 190, 232, 256, 280],
        "RUPA": [47, 48, 49, 119, 120, 121, 191, 192, 193, 233, 257, 281],
        "RUPB": [50, 51, 52, 122, 123, 124, 194, 195, 196, 234, 258, 282],
        "RUPC": [53, 54, 55, 125, 126, 127, 197, 198, 199, 235, 259, 283],
        "RELB": [56, 57, 58, 128, 129, 130, 200, 201, 202, 236, 260, 284],
        "RMEP": [59, 60, 61, 131, 132, 133, 203, 204, 205, 237, 261, 285],
        "RWRA": [62, 63, 64, 134, 135, 136, 206, 207, 208, 238, 262, 286],
        "RWRB": [65, 66, 67, 137, 138, 139, 209, 210, 211, 239, 263, 287],
        "RFRA": [68, 69, 70, 140, 141, 142, 212, 213, 214, 240, 264, 288],
        "RFIN": [71, 72, 73, 143, 144, 145, 215, 216, 217, 241, 265, 289],
    }

    # Extract the traj_data for each prefix
    traj_data = {}
    for prefix, indices in column_indices.items():
        traj_data.update(extract_columns(prefix, indices))

    return traj_data, Frame, time

def Traj_Space_data(traj_data):

    """
    Processes trajectory data to calculate position, speed, acceleration, and jerk for each marker.

    Parameters:
        traj_data (dict): A dictionary containing trajectory data with keys for position (X, Y, Z), 
                            velocity (VX, VY, VZ), and acceleration (AX, AY, AZ) for each marker.

    Returns:
        dict: A dictionary where each key is a marker name, and the value is a tuple containing:
            - Position (pd.Series): The calculated position in space using the X, Y, Z coordinates.
            - Speed (pd.Series): The calculated speed in space using the VX, VY, VZ components.
            - Acceleration (pd.Series): The calculated acceleration in space using the AX, AY, AZ components.
            - Jerk (pd.Series): The calculated change in acceleration over time (jerk).
    """
    Traj_Space_data = {}

    # Iterate through each marker in the trajectory data
    for marker in set(k.split('_')[0] for k in traj_data.keys()):
        # Calculate the position in space using X, Y, Z
        Position = (traj_data[f"{marker}_X"]**2 + traj_data[f"{marker}_Y"]**2 + traj_data[f"{marker}_Z"]**2)**0.5

        # Calculate the speed in space using VX, VY, VZ
        Speed = (traj_data[f"{marker}_VX"]**2 + traj_data[f"{marker}_VY"]**2 + traj_data[f"{marker}_VZ"]**2)**0.5

        # Calculate the acceleration in space using AX, AY, AZ
        Acceleration = (traj_data[f"{marker}_AX"]**2 + traj_data[f"{marker}_AY"]**2 + traj_data[f"{marker}_AZ"]**2)**0.5

        # Calculate the change in acceleration over time (jerk)
        Jerk = Acceleration.diff()

        # Store the results for the current marker
        Traj_Space_data[marker] = (Position, Speed, Acceleration, Jerk)

    return Traj_Space_data

def plot_single_marker_traj(traj_data, time, marker_name, save_path):
    """
    Plots position (X, Y, Z) and velocity (VX, VY, VZ) components of a marker.

    Args:
        traj_data (dict): Trajectory data with position and velocity keys.
        time (pd.Series): Time for each frame.
        marker_name (str): Marker name to plot (e.g., "RFIN").
        save_path (str, optional): Directory to save the figure. If None, the figure is not saved.
    """
    fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    components = [("X", "Position", "mm"), ("Y", "Position", "mm"), ("Z", "Position", "mm"),
                    ("VX", "Velocity", "mm/s"), ("VY", "Velocity", "mm/s"), ("VZ", "Velocity", "mm/s")]
    colors = ["blue", "green", "red"]

    for i, (comp, label, unit) in enumerate(components):
        axs[i].plot(time, traj_data[f"{marker_name}_{comp}"], color=colors[i % 3])
        axs[i].set_title(f"{marker_name} {comp} {label}")
        axs[i].set_ylabel(f"{comp} ({unit})")
        axs[i].grid()

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    if save_path:
        unique_name = f"{marker_name}_traj_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        fig.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()
    
def plot_single_marker_space(Traj_Space_data, time, marker_name, save_path):
    """
    Plots position, speed, and acceleration for a marker.

    Args:
        Traj_Space_data (dict): Dictionary with position, speed, and acceleration data.
        time (pd.Series): Time for each frame.
        marker_name (str): Marker name for plot titles.
        save_path (str, optional): Directory to save the figure. If None, the figure is not saved.
    """
    position, speed, acceleration, _ = Traj_Space_data[marker_name]
    titles = ["Position (mm)", "Speed (mm/s)", "Acceleration (mm/s²)"]
    colors = ["blue", "green", "red"]

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for ax, data, title, color in zip(axs, [position, speed, acceleration], titles, colors):
        ax.plot(time, data, color=color)
        ax.set_title(f"{marker_name} {title}")
        ax.set_ylabel(title)
        ax.grid()
    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    if save_path:
        unique_name = f"{marker_name}_space_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        fig.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def find_local_minima_peaks(data, prominence_threshold):
    """
    Finds local minima and peaks in the given data.

    Args:
        data (pd.Series): The data to analyze (e.g., position, speed, etc.).
        prominence_threshold (float): Prominence threshold for minima and peaks.

    Returns:
        tuple: Two pandas Series containing the local minima and peaks.
    """
    # Find minima and peaks
    minima = data[find_peaks(-data, prominence=prominence_threshold)[0]]
    peaks = data[find_peaks(data, prominence=prominence_threshold)[0]]

    return minima, peaks

def plot_pos_speed_speed_minima_space(time, Traj_Space_data, speed_minima, marker_name, save_path):
    """
    Plots position and speed for a marker, highlighting speed minima.

    Args:
        time (pd.Series): Time for each frame.
        Traj_Space_data (dict): Dictionary containing trajectory space data.
        speed_minima (pd.Series): Indices of local minima in speed.
        marker_name (str): Marker name for plot titles.
        save_path (str, optional): Directory to save the figure. If None, the figure is not saved.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    position, speed, _, _ = Traj_Space_data[marker_name]

    for ax, data, title, ylabel, color in zip(
        axs, [position, speed],
        [f"{marker_name} Position with Speed Minima", f"{marker_name} Speed with Local Minima"],
        ["Position (mm)", "Speed (mm/s)"], ["blue", "green"]
    ):
        ax.plot(time, data, color=color, label=title.split()[1])
        ax.plot(time[speed_minima.index], data[speed_minima.index], "ro", label="Speed Minima")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid()

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    if save_path:
        unique_name = f"{marker_name}_position_speed_minima_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        fig.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def find_speed_segments(marker_name, Traj_Space_data, time, speed_threshold, speed_peaks):
    """
    Finds segments of time where the speed exceeds a given threshold around speed peaks.

    Args:
        marker_name (str): Name of the marker.
        Traj_Space_data (dict): Dictionary containing trajectory space data.
        time (pd.Series): Time for each frame.
        speed_threshold (float): Speed threshold to define segments.
        prominence_threshold_speed (float): Prominence threshold for speed peaks.

    Returns:
        list: A list of tuples containing start and end times for each segment.
    """
    _, speed, _, _ = Traj_Space_data[marker_name]

    # Initialize a list to store the start and end times for each peak
    speed_segments = []

    # Iterate through each speed peak
    for peak in speed_peaks.index:
        # Find the earliest time before the peak where speed drops below the threshold
        start_index = peak
        while start_index > 0 and speed[start_index] > speed_threshold:
            start_index -= 1

        # Find the latest time after the peak where speed drops below the threshold
        end_index = peak
        while end_index < len(speed) - 1 and speed[end_index] > speed_threshold:
            end_index += 1

        # Store the start and end times
        speed_segments.append((time[start_index], time[end_index]))

    # # Print the segments
    # for i, (start, end) in enumerate(speed_segments):
    #     print(f"Segment {i + 1}: Start = {start:.2f}s, End = {end:.2f}s, Duration = {end - start:.2f}s")

    return speed_segments

def plot_speed_position_segmentsByspeed_space(marker_name, time, Traj_Space_data, speed_segments, speed_threshold, prominence_threshold_speed, save_path):
    """
    Plots speed and position for a marker, highlighting speed segment start/end points.

    Args:
        marker_name (str): Marker name.
        time (pd.Series): Time data.
        speed (pd.Series): Speed data.
        position (pd.Series): Position data.
        speed_segments (list): List of (start, end) tuples for speed segments.
        save_path (str): Directory to save the figure.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    position, speed, _, _ = Traj_Space_data[marker_name]


    for ax, data, label, color in zip(
        axs, [position, speed], ["Position", "Speed"], ["blue", "green"]
    ):
        ax.plot(time, data, label=f"{marker_name} {label}", color=color)
        for start, end in speed_segments:
            ax.plot(start, data[time == start].values[0], "go")
            ax.plot(end, data[time == end].values[0], "ro")
        ax.set_title(f"{marker_name} {label} with segmentsByspeed (Threshold: {speed_threshold}, Prominence: {prominence_threshold_speed})")
        ax.set_ylabel(f"{label} (mm/s)" if label == "Speed" else f"{label} (mm)")
        ax.legend()
        ax.grid()

    axs[-1].set_xlabel("Time (s)")

    # Adjust layout and show the plot
    plt.tight_layout()

    if save_path:
        unique_name = f"{marker_name}_speed_position_segmentsByspeed_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        fig.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def plot_marker_xyz_with_peaks_troughs_traj(traj_data, time, marker_name, save_path, prominence_threshold_position):
    """
    Plots X, Y, Z components of a marker with peaks and troughs.

    Args:
        traj_data (dict): Trajectory data with X, Y, Z components.
        time (pd.Series): Time for each frame.
        marker_name (str): Marker name (e.g., "RFIN").
        save_path (str, optional): Directory to save the figure.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    colors = ["blue", "green", "red"]
    for i, coord in enumerate(["X", "Y", "Z"]):
        data = traj_data[f"{marker_name}_{coord}"]

        troughs, peaks = find_local_minima_peaks(data, prominence_threshold_position)

        axs[i].plot(time, data, label=f"{marker_name}_{coord}", color=colors[i])
        axs[i].plot(time[peaks.index], data.iloc[peaks.index], "ro", label="Peaks")
        axs[i].plot(time[troughs.index], data.iloc[troughs.index], "go", label="Troughs")
        axs[i].set_title(f"{marker_name}_{coord} with Peaks and Troughs")
        axs[i].set_ylabel(f"{coord} (mm)")
        axs[i].legend()
        axs[i].grid()
    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    if save_path:
        path = os.path.join(save_path, f"{marker_name}_xyz_peaks_troughs_{datetime.datetime.now():%Y%m%d_%H%M%S}.png")
        fig.savefig(path)
        print(f"Figure saved to {path}")
    plt.show()

def plot_marker_trajectory_components(time, traj_data, Traj_Space_data, marker_name, save_path):
    """
    Plots X, Y, Z coordinates, speed, acceleration, and jerk for a chosen marker.

    Args:
        time (pd.Series): Time for each frame.
        traj_data (dict): Dictionary containing trajectory data (X, Y, Z components).
        Traj_Space_data (dict): Dictionary containing trajectory space data (position, speed, acceleration, jerk).
        marker_name (str): Marker name to plot (e.g., "RFIN").
        save_path (str): Directory to save the figure.
    """
    position, speed, acceleration, jerk = Traj_Space_data[marker_name]
    x = traj_data[f"{marker_name}_X"]
    y = traj_data[f"{marker_name}_Y"]
    z = traj_data[f"{marker_name}_Z"]
    components = [x, y, z, position, speed, acceleration, jerk]

    titles = [f"{marker_name} X", f"{marker_name} Y", f"{marker_name} Z", f"{marker_name} position", f"{marker_name} Speed", 
              f"{marker_name} Acceleration", f"{marker_name} Jerk"]
    ylabels = ["mm", "mm", "mm", "mm", "mm/s", "mm/s²", "mm/s³"]
    colors = ["purple", "orange", "cyan", "blue", "green", "red", "black"]

    fig, axs = plt.subplots(7, 1, figsize=(12, 18), sharex=True)
    for ax, title, data, color, ylabel in zip(axs, titles, components, colors, ylabels):
        ax.plot(time, data, color=color)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid()

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    if save_path:
        path = os.path.join(save_path, f"{marker_name}_trajectory_components_{datetime.datetime.now():%Y%m%d_%H%M%S}.png")
        fig.savefig(path)
        print(f"Figure saved to {path}")

    plt.show()

def plot_marker_3d_trajectory_traj(traj_data, marker_name, start_frame, end_frame, save_path):
    """
    Plots a 3D trajectory of a marker for a selectable duration.

    Args:
        traj_data (dict): Dictionary containing trajectory data (X, Y, Z components).
        marker_name (str): Marker name to plot (e.g., "RFIN").
        start_frame (int): Starting frame for the plot.
        end_frame (int): Ending frame for the plot.
        save_path (str, optional): Directory to save the figure. If None, the figure is not saved.
    """
    X, Y, Z = (traj_data[f"{marker_name}_{axis}"][start_frame:end_frame] for axis in "XYZ")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X, Y, Z, label=f"{marker_name} Trajectory", color="blue")
    ax.set(title=f"3D Trajectory of {marker_name}", xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
    ax.legend()
    if save_path:
        path = os.path.join(save_path, f"{marker_name}_3d_traj_{start_frame}_{end_frame}.png")
        fig.savefig(path)
        print(f"Figure saved to {path}")
    plt.show()

def plot_marker_radial_components_space(time, traj_data, marker_name, save_path):
    """
    Plots the radial position, velocity, and acceleration for a given marker.

    Args:
        time (pd.Series): Time for each frame.
        traj_data (dict): Dictionary containing trajectory data.
        marker_name (str): Marker name to plot (e.g., "RFIN").
        save_path (str, optional): Directory to save the figure. If None, the figure is not saved.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    titles = [f"{marker_name} Radial Position", f"{marker_name} Radial Velocity", f"{marker_name} Radial Acceleration"]
    ylabels = ["Radial Position (mm)", "Radial Velocity (mm/s)", "Radial Acceleration (mm/s²)"]
    colors = ["blue", "green", "red"]
    keys = [f"{marker_name}_radial_pos", f"{marker_name}_radial_vel", f"{marker_name}_radial_acc"]

    for ax, title, ylabel, color, key in zip(axs, titles, ylabels, colors, keys):
        ax.plot(time, traj_data[key], label=title, color=color)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid()

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    if save_path:
        unique_name = f"{marker_name}_radial_components_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        fig.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def extract_marker_start_end(marker_name, traj_data, Traj_Space_data, time, x_threshold, z_range, speed_range, accel_range):
    """
    Extracts the start and end times for a marker based on specified conditions.

    Args:
        marker_name (str): Name of the marker (e.g., "RFIN").
        traj_data (dict): Dictionary containing trajectory data.
        time (pd.Series): Time for each frame.
        x_threshold (float): Threshold for the X coordinate.
        z_range (tuple): Range for the Z coordinate (min, max).
        speed_range (tuple): Range for the speed (min, max).
        accel_range (tuple): Range for the acceleration (min, max).

    Returns:
        tuple: Two numpy arrays containing the start and end times.
    """
    # Extract relevant data for the marker
    X = traj_data[f"{marker_name}_X"]
    Z = traj_data[f"{marker_name}_Z"]
    Speed = Traj_Space_data[marker_name][1]  # Speed from the provided Traj_Space_data dictionary
    Acceleration = Traj_Space_data[marker_name][2]  # Acceleration from the provided Traj_Space_data dictionary

    # Define masks for start and end conditions
    tStart_mask = (X < x_threshold) & Z.between(*z_range) & Speed.between(*speed_range) & Acceleration.between(*accel_range)
    tEnd_mask = (X > x_threshold) & Z.between(*z_range) & Speed.between(*speed_range) & Acceleration.between(*accel_range)

    # Extract start and end times
    tStart_indices = time[tStart_mask].to_numpy()
    tEnd_indices = time[tEnd_mask].to_numpy()

    # Check if indices are empty
    if not len(tStart_indices) or not len(tEnd_indices):
        raise ValueError("Error: tStart_indices or tEnd_indices is empty.")

    return tStart_indices, tEnd_indices

def cluster_and_find_representatives(tStart_indices, tEnd_indices, top_n=15):
    """
    Clusters indices based on differences and finds representative values for each cluster.

    Args:
        tStart_indices (np.ndarray): Array of start indices.
        tEnd_indices (np.ndarray): Array of end indices.
        top_n (int): Number of top differences to consider for clustering.

    Returns:
        tuple: Two lists containing representative values for start and end clusters.
    """
    def cluster_indices(indices, diffs):
        clusters, start = [], 0
        for end in np.sort(np.argsort(diffs)[-top_n:]):
            clusters.append(indices[start:end + 1])
            start = end + 1
        clusters.append(indices[start:])
        return [np.median(cluster) if len(cluster) % 2 != 0 else sorted(cluster)[len(cluster) // 2 - 1] for cluster in clusters]

    tStart_representatives = cluster_indices(tStart_indices, np.diff(tStart_indices))
    tEnd_representatives = cluster_indices(tEnd_indices, np.diff(tEnd_indices))

    return tStart_representatives, tEnd_representatives

def plot_marker_with_start_end_representatives(time, traj_data, Traj_Space_data, marker_name, tStart_indices, tEnd_indices, tStart_representatives, tEnd_representatives, save_path, x_threshold, z_range, speed_range, accel_range):
    """
    Plots X, Z, Speed, and Acceleration for a marker, highlighting start/end indices and their representatives.

    Args:
        time (pd.Series): Time for each frame.
        traj_data (dict): Dictionary containing trajectory data.
        Traj_Space_data (dict): Dictionary containing trajectory space data.
        marker_name (str): Marker name to plot (e.g., "RFIN").
        tStart_indices (np.ndarray): Start indices for the marker.
        tEnd_indices (np.ndarray): End indices for the marker.
        tStart_representatives (list): Representative start indices.
        tEnd_representatives (list): Representative end indices.
        save_path (str, optional): Directory to save the figure. If None, the figure is not saved.
        x_threshold (float): Threshold for the X coordinate.
        z_range (tuple): Range for the Z coordinate (min, max).
        speed_range (tuple): Range for the speed (min, max).
        accel_range (tuple): Range for the acceleration (min, max).
    """
    X = traj_data[f"{marker_name}_X"]
    Z = traj_data[f"{marker_name}_Z"]
    Speed = Traj_Space_data[marker_name][1]
    Acceleration = Traj_Space_data[marker_name][2]

    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    titles = [f"{marker_name} X", f"{marker_name} Z", f"{marker_name} Speed", f"{marker_name} Acceleration"]
    data = [X, Z, Speed, Acceleration]
    colors = ["blue", "blue", "green", "orange"]
    ylabels = ["X (mm)", "Z (mm)", "Speed (mm/s)", "Acceleration (mm/s²)"]

    for ax, title, d, color, ylabel in zip(axs, titles, data, colors, ylabels):
        ax.plot(time, d, color=color, label=title)
        ax.plot(tStart_indices, d[np.isin(time, tStart_indices)], "go", label="tStart")
        ax.plot(tEnd_indices, d[np.isin(time, tEnd_indices)], "ro", label="tEnd")
        ax.plot(tStart_representatives, d[np.isin(time, tStart_representatives)], "k^", label="tStart Representative", color="black")
        ax.plot(tEnd_representatives, d[np.isin(time, tEnd_representatives)], "d", label="tEnd Representative", color="darkgrey")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid()

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    if save_path:
        unique_name = (f"{marker_name}_start_end_representatives_x{x_threshold}_z{z_range[0]}-{z_range[1]}"
                        f"_speed{speed_range[0]}-{speed_range[1]}_accel{accel_range[0]}-{accel_range[1]}"
                        f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        full_path = os.path.join(save_path, unique_name)
        fig.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()
