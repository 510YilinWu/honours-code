import pandas as pd
import os
import datetime
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
from scipy.signal import butter, filtfilt

'''This module provides functions to read trajectory data from a CSV file, process it to extract relevant information, and perform various analyses on the trajectory data.'''

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

def find_local_minima_peaks(data, prominence_threshold):
    """
    Finds local minima and peaks in the given data.

    Args:
        data (pd.Series): The data to analyze (e.g., position, speed, etc.).
        prominence_threshold (float): Prominence threshold for minima and peaks.

    Returns:
        tuple: Two pandas Series containing the local minima and peaks.
    """
    # # Find minima and peaks
    # minima = data[find_peaks(-data, prominence=prominence_threshold)[0]]
    # peaks = data[find_peaks(data, prominence=prominence_threshold)[0]]

    # Find minima and peaks
    minima_indices = find_peaks(-data, prominence=prominence_threshold)[0]
    peaks_indices = find_peaks(data, prominence=prominence_threshold)[0]
    minima = data[minima_indices]
    peaks = data[peaks_indices]


    # # Filter minima and peaks based on the duration between consecutive peaks
    # minima = minima[minima.index.to_series().diff().fillna(0) / 200 > 0.5]
    # peaks = peaks[peaks.index.to_series().diff().fillna(0) / 200 > 0.5]

    return minima, peaks, minima_indices, peaks_indices

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

        # Ensure no repetitive reach with identical start_index and end_index
        if speed_segments and speed_segments[-1] == (time[start_index], time[end_index]):
            continue

        # Store the start and end times
        speed_segments.append((time[start_index], time[end_index]))

    # # Print the segments
    # for i, (start, end) in enumerate(speed_segments):
    #     print(f"Segment {i + 1}: Start = {start:.2f}s, End = {end:.2f}s, Duration = {end - start:.2f}s")

    return speed_segments

def calculate_rangesByBoxTraj(Box_Traj_file):
    # Read CSV
    df = pd.read_csv(Box_Traj_file[0], skiprows=4, sep=r"\s+|,", engine="python")

    # Column indices
    column_indices = {
        "BOX:RTC": [2, 3, 4],
        "BOX:LTC": [5, 6, 7],
        "BOX:LBC": [8, 9, 10],
        "BOX:PT": [11, 12, 13],
        "BOX:PB": [14, 15, 16],
    }

    # Extract data without filtering
    Box_data = {}
    for prefix, (ix, iy, iz) in column_indices.items():
        x = df.iloc[:, ix]
        y = df.iloc[:, iy]
        z = df.iloc[:, iz]
        Box_data[prefix] = (x, y, z)

    BoxRange = (Box_data["BOX:LTC"][0].mean(),Box_data["BOX:PT"][0].mean(), Box_data["BOX:RTC"][0].mean())

    return BoxRange

def classify_speed_segments(speed_segments, traj_data, marker_name, time, BoxRange):
    """
    Classifies speed segments into reach and return segments based on X-coordinate changes and predefined ranges.

    Args:
        speed_segments (list): List of (start_time, end_time) tuples for speed segments.
        traj_data (dict): Dictionary containing trajectory data.
        marker_name (str): Marker name to classify segments for.
        time (pd.Series): Time for each frame.
        lfin_x_range (tuple): Range for LFIN marker classification.
        rfin_x_range (tuple): Range for RFIN marker classification.
        # LFin X Range: (np.float64(-301.652802), np.float64(-37.812771))
        # RFin X Range: (np.float64(-38.066681), np.float64(229.149918))

    Returns:
        tuple: Two lists containing reach and return speed segments with start/end times.
    """
    reach_speed_segments = []
    return_speed_segments = []

    for segment in speed_segments:
        start_time, end_time = segment
        start_index = time[time == start_time].index[0]
        end_index = time[time == end_time].index[0]

        start_x = traj_data[f"{marker_name}_X"][start_index]
        end_x = traj_data[f"{marker_name}_X"][end_index]


        if marker_name == "RFIN":
            # if start_x < end_x and BoxRange[0] > end_x > BoxRange[1] and BoxRange[1] > start_x > BoxRange[2]:
            if BoxRange[0] > end_x > BoxRange[1] and BoxRange[1] > start_x > BoxRange[2]:
                reach_speed_segments.append(segment)
            else:
                return_speed_segments.append(segment)

        elif marker_name == "LFIN":
            # if start_x > end_x and BoxRange[0] > start_x > BoxRange[1] and BoxRange[1] > end_x > BoxRange[2]:
            if BoxRange[0] > start_x > BoxRange[1] and BoxRange[1] > end_x > BoxRange[2]:
                reach_speed_segments.append(segment)
            else:
                return_speed_segments.append(segment)
    return reach_speed_segments, return_speed_segments

def calculate_reach_durations(reach_speed_segments):
    """
    Calculates the duration of each reach segment.

    Args:
        reach_speed_segments (list): List of (start_time, end_time) tuples for reach segments.

    Returns:
        list: A list of durations for each reach segment.
    """
    reach_durations = []
    for start_time, end_time in reach_speed_segments:
        duration = end_time - start_time
        reach_durations.append(duration)
    return reach_durations

def calculate_reach_distances(Traj_Space_data, marker_name, reach_speed_segments, time):
    """
    Calculates the reach position change (reach distance) for each reach segment.

    Args:
        Traj_Space_data (dict): Dictionary containing trajectory space data (position, speed, acceleration, jerk).
        marker_name (str): Marker name to calculate reach distances for (e.g., "RFIN").
        reach_speed_segments (list): List of (start_time, end_time) tuples for reach speed segments.
        time (pd.Series): Time for each frame.

    Returns:
        list: A list of reach distances for each reach segment.
    """
    position = Traj_Space_data[marker_name][0]  # Extract position data
    reach_distances = []

    for start_time, end_time in reach_speed_segments:
        # Find the indices corresponding to the start and end times
        start_index = time[time >= start_time].index[0]
        end_index = time[time >= end_time].index[0]

        # Calculate the position change (distance) for the segment
        distance = abs(position.iloc[end_index] - position.iloc[start_index])
        reach_distances.append(distance)

    return reach_distances

def calculate_reach_path_distances(Traj_Space_data, marker_name, reach_speed_segments, time):
    """
    Calculates the path_distance traveled for each reach segment, considering the entire trajectory.

    Args:
        Traj_Space_data (dict): Dictionary containing trajectory space data (position, speed, acceleration, jerk).
        marker_name (str): Marker name to calculate path_distance for (e.g., "RFIN").
        reach_speed_segments (list): List of (start_time, end_time) tuples for reach speed segments.
        time (pd.Series): Time for each frame.

    Returns:
        list: A list of path_distance traveled for each reach segment.
    """
    position = Traj_Space_data[marker_name][0]  # Extract position data
    path_distances = []

    for start_time, end_time in reach_speed_segments:
        # Find the indices corresponding to the start and end times
        start_index = time[time >= start_time].index[0]
        end_index = time[time >= end_time].index[0]

        # Calculate the path_distance traveled by summing the absolute differences between consecutive positions
        path_distance = position.iloc[start_index:end_index].diff().abs().sum()
        path_distances.append(path_distance)

    return path_distances

def calculate_v_peaks(Traj_Space_data, marker_name, time, reach_speed_segments):
    """
    Calculates the peak velocities (v_peaks) and their corresponding time indices for all reach segments of a given marker.

    Args:
        Traj_Space_data (dict): Dictionary containing trajectory space data (position, speed, acceleration, jerk).
        marker_name (str): Marker name to calculate peak velocities for (e.g., "RFIN").
        time (pd.Series): Time for each frame.
        reach_speed_segments (list): List of (start_time, end_time) tuples for reach segments.

    Returns:
        tuple: Two lists:
            - v_peaks: A list of peak velocities for each reach segment.
            - v_peak_indices: A list of corresponding time indices for each peak velocity.
    """
    speed = Traj_Space_data[marker_name][1]

    v_peaks = []
    v_peak_indices = []

    for segment in reach_speed_segments:
        # Convert start_time and end_time from time values to indices
        start_idx = time[time >= segment[0]].index[0]
        end_idx = time[time >= segment[1]].index[0]

        # Find the peak velocity and its corresponding time index
        v_peak = speed.iloc[start_idx:end_idx].max()
        v_peak_time_index = speed.iloc[start_idx:end_idx].idxmax()
        v_peaks.append(v_peak)
        v_peak_indices.append(v_peak_time_index)

    return v_peaks, v_peak_indices

def split_segments_at_peak_speed(reach_speed_segments, Traj_Space_data, time, v_peaks, marker_name):
    """
    Splits reach_speed_segments into two parts: start to peak speed, and peak to end.

    Args:
        reach_speed_segments (list): List of (start_time, end_time) tuples for reach speed segments.
        Traj_Space_data (dict): Dictionary containing trajectory space data (position, speed, acceleration, jerk).
        time (pd.Series): Time for each frame.
        v_peaks (list): List of peak velocities for each reach segment.
        marker_name (str): Marker name to analyze (e.g., "RFIN").

    Returns:
        tuple: Two lists containing segments split at peak speed:
            - start_to_peak_segments: Segments from start to peak speed.
            - peak_to_end_segments: Segments from peak speed to end.
    """
    speed = Traj_Space_data[marker_name][1]  # Extract speed data

    start_to_peak_segments = []
    peak_to_end_segments = []

    for (start_time, end_time), v_peak in zip(reach_speed_segments, v_peaks):
        # Find the indices corresponding to the start and end times
        start_index = time[time >= start_time].index[0]
        end_index = time[time >= end_time].index[0]

        # Find the index of the peak speed within the segment
        peak_index = speed[start_index:end_index][speed[start_index:end_index] == v_peak].index[0]

        # Split the segment into two parts
        start_to_peak_segments.append((start_time, time[peak_index]))
        peak_to_end_segments.append((time[peak_index], end_time))

    return start_to_peak_segments, peak_to_end_segments

def calculate_ldlj_for_all_reaches(Traj_Space_data, marker_name, time, test_windows):
    """
    Calculates the Log Dimensionless Jerk (LDLJ) for all test windows of a given marker and saves the peak acceleration and jerk for each window.

    Args:
        Traj_Space_data (dict): Dictionary containing trajectory space data (position, speed, acceleration, jerk).
        marker_name (str): Marker name to calculate LDLJ for (e.g., "RFIN").
        time (pd.Series): Time for each frame.
        test_windows (list): List of (start_time, end_time) tuples for test windows.

    Returns:
        tuple: A tuple containing:
            - list: A list of LDLJ values for each test window.
                High jerk → large DJM → large log(DJM) → low negative log(DJM)
                Low jerk (smoother motion) → small DJM → small log(DJM) → high negative log(DJM)
            - list: A list of peak accelerations for each test window.
            - list: A list of peak jerks for each test window.
    """
    speed = Traj_Space_data[marker_name][1]
    acceleration = Traj_Space_data[marker_name][2]
    jerk = Traj_Space_data[marker_name][3]

    LDLJ_values = []
    acc_peaks = []
    jerk_peaks = []

    for window in test_windows:
        # Convert Tstart and Tend from time values to indices
        Tstart_idx = time[time >= window[0]].index[0]
        Tend_idx = time[time >= window[1]].index[0]

        # Calculate the integral of the squared jerk
        jerk_squared_integral = np.trapezoid(jerk.iloc[Tstart_idx:Tend_idx]**2, time.iloc[Tstart_idx:Tend_idx])

        # Find the peak acceleration
        acc_peak = acceleration.iloc[Tstart_idx:Tend_idx].max()
        acc_peaks.append(acc_peak)

        # Find the peak jerk
        jerk_peak = jerk.iloc[Tstart_idx:Tend_idx].max()
        jerk_peaks.append(jerk_peak)

        # Avoid division by zero
        vpeak_ForTimeWindow = speed.iloc[Tstart_idx:Tend_idx].max()  # Renamed to vpeak_ForTimeWindow for time window
        if vpeak_ForTimeWindow == 0:
            LDLJ_values.append(None)  # Append None if calculation is not possible
            continue

        # Calculate the dimensionless jerk
        dimensionless_jerk = (jerk_squared_integral * (time.iloc[Tend_idx] - time.iloc[Tstart_idx])**3) / (vpeak_ForTimeWindow**2)

        # Calculate and append the Log Dimensionless Jerk (LDLJ)
        LDLJ = -np.log(dimensionless_jerk)
        LDLJ_values.append(LDLJ)

    return LDLJ_values, acc_peaks, jerk_peaks

'''Plotting functions for trajectory data visualization.'''

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

def plot_pos_speed_one_extrema_space(time, Traj_Space_data, minima, peaks, marker_name, save_path):
    """
    Plots position and speed for a marker, highlighting minima and peaks.

    Args:
        time (pd.Series): Time for each frame.
        Traj_Space_data (dict): Dictionary containing trajectory space data.
        minima (pd.Series): Indices of local minima.
        peaks (pd.Series): Indices of local peaks.
        marker_name (str): Marker name for plot titles.
        save_path (str, optional): Directory to save the figure. If None, the figure is not saved.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    position, speed, _, _ = Traj_Space_data[marker_name]

    for ax, data, title, ylabel, color in zip(
        axs, [position, speed],
        [f"{marker_name} Position with Extrema", f"{marker_name} Speed with Extrema"],
        ["Position (mm)", "Speed (mm/s)"], ["blue", "green"]
    ):
        ax.plot(time, data, color=color, label=title.split()[1])
        ax.plot(time[minima.index], data[minima.index], "ro", label="Minima")
        ax.plot(time[peaks.index], data[peaks.index], "go", label="Peaks")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid()

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    if save_path:
        unique_name = f"{marker_name}_position_speed_1extrema_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        fig.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def plot_x_speed_one_extrema_space(time, Traj_Space_data, traj_data, minima, peaks, marker_name, save_path):
    """
    Plots X coordinate and speed for a marker, highlighting minima and peaks.

    Args:
        time (pd.Series): Time for each frame.
        traj_data (dict): Dictionary containing trajectory data.
        minima (pd.Series): Indices of local minima.
        peaks (pd.Series): Indices of local peaks.
        marker_name (str): Marker name for plot titles.
        save_path (str, optional): Directory to save the figure. If None, the figure is not saved.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    x = traj_data[f"{marker_name}_X"]
    speed = Traj_Space_data[marker_name][1]

    for ax, data, title, ylabel, color in zip(
        axs, [x, speed],
        [f"{marker_name} X with Extrema", f"{marker_name} Speed with Extrema"],
        ["X (mm)", "Speed (mm/s)"], ["blue", "green"]
    ):
        ax.plot(time, data, color=color, label=title.split()[1])
        ax.plot(time[minima.index], data[minima.index], "ro", label="Minima")
        ax.plot(time[peaks.index], data[peaks.index], "go", label="Peaks")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid()

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    if save_path:
        unique_name = f"{marker_name}_x_speed_1extrema_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        fig.savefig(full_path)
        # plt.close(fig)
        print(f"Figure saved to {full_path}")

    plt.show()

def plot_pos_speed_two_extrema_space(time, Traj_Space_data, minima_1, peaks_1, minima_2, peaks_2, marker_name, save_path):
    """
    Plots position and speed for a marker, highlighting two sets of minima and peaks.

    Args:
        time (pd.Series): Time for each frame.
        Traj_Space_data (dict): Dictionary containing trajectory space data.
        minima_1 (pd.Series): Indices of the first set of local minima.
        peaks_1 (pd.Series): Indices of the first set of local peaks.
        minima_2 (pd.Series): Indices of the second set of local minima.
        peaks_2 (pd.Series): Indices of the second set of local peaks.
        marker_name (str): Marker name for plot titles.
        save_path (str, optional): Directory to save the figure. If None, the figure is not saved.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    position, speed, _, _ = Traj_Space_data[marker_name]

    for ax, data, title, ylabel, color in zip(
        axs, [position, speed],
        [f"{marker_name} Position with Extrema", f"{marker_name} Speed with Extrema"],
        ["Position (mm)", "Speed (mm/s)"], ["blue", "green"]
    ):
        ax.plot(time, data, color=color, label=title.split()[1])
        ax.plot(time[minima_1.index], data[minima_1.index], "ro", label="Minima Set 1")
        ax.plot(time[peaks_1.index], data[peaks_1.index], "go", label="Peaks Set 1")
        ax.plot(time[minima_2.index], data[minima_2.index], "mo", label="Minima Set 2")
        ax.plot(time[peaks_2.index], data[peaks_2.index], "yo", label="Peaks Set 2")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid()

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    if save_path:
        unique_name = f"{marker_name}_position_speed_2extrema_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        fig.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def plot_speed_x_segmentsByspeed_space(marker_name, time, Traj_Space_data, traj_data, speed_segments, speed_minima, speed_peaks, speed_threshold, prominence_threshold_speed, save_path):
    """
    Plots speed and X-coordinate for a marker, highlighting speed segment start/end points, minima, and peaks.

    Args:
        marker_name (str): Marker name.
        time (pd.Series): Time data.
        Traj_Space_data (dict): Dictionary containing trajectory space data.
        traj_data (dict): Dictionary containing trajectory data.
        speed_segments (list): List of (start, end) tuples for speed segments.
        speed_minima (pd.Series): Indices of local minima in speed.
        speed_peaks (pd.Series): Indices of local peaks in speed.
        speed_threshold (float): Speed threshold to define segments.
        prominence_threshold_speed (float): Prominence threshold for speed peaks.
        save_path (str): Directory to save the figure.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    x = traj_data[f"{marker_name}_X"]
    speed = Traj_Space_data[marker_name][1]

    for ax, data, label, color in zip(
        axs, [x, speed], ["X", "Speed"], ["blue", "green"]
    ):
        ax.plot(time, data, label=f"{marker_name} {label}", color=color)
        for start, end in speed_segments:
            ax.plot(start, data[time == start].values[0], "ro", label="Segment Start")
            ax.plot(end, data[time == end].values[0], "ro", label="Segment End")
            ax.plot(time[speed_minima.index], data[speed_minima.index], "go", label="Speed Minima")
            # ax.plot(time[speed_peaks.index], data[speed_peaks.index], "go", label="Speed Peaks")
        ax.set_title(f"{marker_name} {label} with segmentsByspeed (Threshold: {speed_threshold}, Prominence: {prominence_threshold_speed})")
        ax.set_ylabel(f"{label} (mm/s)" if label == "Speed" else f"{label} (mm)")
        ax.grid()

    axs[-1].set_xlabel("Time (s)")

    # Adjust layout and show the plot
    plt.tight_layout()

    if save_path:
        unique_name = f"{marker_name}_speed_x_segmentsByspeed_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        fig.savefig(full_path)
        plt.close(fig)
        print(f"Figure saved to {full_path}")

    # plt.show()

def plot_speed_position_segmentsByspeed_space(marker_name, time, Traj_Space_data, speed_segments, speed_minima, speed_threshold, prominence_threshold_speed, save_path):
    """
    Plots speed and position for a marker, highlighting speed segment start/end points and minima.

    Args:
        marker_name (str): Marker name.
        time (pd.Series): Time data.
        Traj_Space_data (dict): Dictionary containing trajectory space data.
        speed_segments (list): List of (start, end) tuples for speed segments.
        speed_minima (pd.Series): Indices of local minima in speed.
        speed_threshold (float): Speed threshold to define segments.
        prominence_threshold_speed (float): Prominence threshold for speed peaks.
        save_path (str): Directory to save the figure.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    position, speed, _, _ = Traj_Space_data[marker_name]

    for ax, data, label, color in zip(
        axs, [position, speed], ["Position", "Speed"], ["blue", "green"]
    ):
        ax.plot(time, data, label=f"{marker_name} {label}", color=color)
        for start, end in speed_segments:
            ax.plot(start, data[time == start].values[0], "go", label="Segment Start")
            ax.plot(end, data[time == end].values[0], "ro", label="Segment End")
        ax.plot(time[speed_minima.index], data[speed_minima.index], "mo", label="Speed Minima")
        ax.set_title(f"{marker_name} {label} with segmentsByspeed (Threshold: {speed_threshold}, Prominence: {prominence_threshold_speed})")
        ax.set_ylabel(f"{label} (mm/s)" if label == "Speed" else f"{label} (mm)")
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
        unique_name = f"{marker_name}_3d_traj_{start_frame}_{end_frame}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        fig.savefig(full_path)
        print(f"Figure saved to {full_path}")

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

def plot_x_position_and_speed_with_segments(time, traj_data, Traj_Space_data, marker_name, speed_segments, save_path, file_path):
    """
    Plots the X coordinate, position, and speed for a given marker, highlighting speed segments.

    Args:
        time (pd.Series): Time for each frame.
        traj_data (dict): Dictionary containing trajectory data.
        Traj_Space_data (dict): Dictionary containing trajectory space data.
        marker_name (str): Marker name to plot (e.g., "RFIN").
        speed_segments (list): List of (start, end) tuples for speed segments.
        save_path (str, optional): Directory to save the figure. If None, the figure is not saved.
    """
    X = traj_data[f"{marker_name}_X"]
    position, speed, _, _ = Traj_Space_data[marker_name]

    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    titles = [f"{marker_name} X Coordinate ({os.path.basename(file_path)})", f"{marker_name} Position", f"{marker_name} Speed"]
    data = [X, position, speed]
    colors = ["blue", "green", "red"]
    ylabels = ["X (mm)", "Position (mm)", "Speed (mm/s)"]

    for ax, title, d, color, ylabel in zip(axs, titles, data, colors, ylabels):
        ax.plot(time, d, color=color)
        for start, end in speed_segments:
            ax.axvspan(start, end, color='yellow', alpha=0.3, label="Speed Segment")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid()

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    if save_path:
        unique_name = f"{marker_name}_x_position_speed_segments_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        fig.savefig(full_path)
        plt.close(fig)
        print(f"Figure saved to {full_path}")

    # plt.show()

def plot_aligned_segments(time, Traj_Space_data, reach_speed_segments, marker_name, save_path, file_path):
    """
    Plots aligned position, speed, acceleration, and jerk segments for a marker in a single figure with 4 subplots.

    Args:
        time (pd.Series): Time for each frame.
        Traj_Space_data (dict): Dictionary containing trajectory space data (position, speed, acceleration, jerk).
        reach_speed_segments (list): List of (start, end) tuples for reach speed segments.
        marker_name (str): Marker name to plot (e.g., "RFIN").
        save_path (str): Directory to save the figure.
    """

    # Define color gradients for green (1-4), orange (5-8), blue (9-12), and red (13-16)
    green_colors = list(mcolors.LinearSegmentedColormap.from_list("", ["lightgreen", "green"])(range(4)))
    orange_colors = list(mcolors.LinearSegmentedColormap.from_list("", ["peachpuff", "orange"])(range(4)))
    blue_colors = list(mcolors.LinearSegmentedColormap.from_list("", ["lightblue", "blue"])(range(4)))
    red_colors = list(mcolors.LinearSegmentedColormap.from_list("", ["lightcoral", "red"])(range(4)))
    all_colors = green_colors + orange_colors + blue_colors + red_colors

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    titles = [f"Aligned Position Segments ({os.path.basename(file_path)})", 
              "Aligned Speed Segments", 
              "Aligned Acceleration Segments", 
              "Aligned Jerk Segments"]
    ylabels = ["Position (mm)", "Speed (mm/s)", "Acceleration (mm/s²)", "Jerk (mm/s³)"]

    # Iterate through the subplots and plot the data
    for i, (ax, title, ylabel) in enumerate(zip(axs, titles, ylabels)):
        for idx, (start, end) in enumerate(reach_speed_segments):
            mask = (time >= start) & (time <= end)
            color = all_colors[idx % 16]  # Cycle through the 16 colors
            ax.plot(time[mask] - start, Traj_Space_data[marker_name][i][mask], 
                    alpha=0.7, label=f"Reach {idx + 1}", color=color)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    axs[-1].set_xlabel("Time (s)")
    axs[0].legend(title="Reach Order", loc="upper right")
    plt.tight_layout()

    if save_path:
        unique_name = f"{marker_name}_aligned_segments_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        fig.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def plot_aligned_segments_xyz(time, traj_data, reach_speed_segments, marker_name, save_path, file_path):
    """
    Plots aligned X, Y, Z segments for a marker in a single figure with 3 subplots.

    Args:
        time (pd.Series): Time for each frame.
        traj_data (dict): Dictionary containing trajectory data (X, Y, Z components).
        reach_speed_segments (list): List of (start, end) tuples for reach speed segments.
        marker_name (str): Marker name to plot (e.g., "RFIN").
        save_path (str): Directory to save the figure.
    """

    # Define color gradients for green (1-4), orange (5-8), blue (9-12), and red (13-16)
    green_colors = list(mcolors.LinearSegmentedColormap.from_list("", ["lightgreen", "green"])(range(4)))
    orange_colors = list(mcolors.LinearSegmentedColormap.from_list("", ["peachpuff", "orange"])(range(4)))
    blue_colors = list(mcolors.LinearSegmentedColormap.from_list("", ["lightblue", "blue"])(range(4)))
    red_colors = list(mcolors.LinearSegmentedColormap.from_list("", ["lightcoral", "red"])(range(4)))
    all_colors = green_colors + orange_colors + blue_colors + red_colors

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    titles = [f"Aligned X Segments ({os.path.basename(file_path)})", "Aligned Y Segments", "Aligned Z Segments"]
    ylabels = ["X (mm)", "Y (mm)", "Z (mm)"]

    # Iterate through the subplots and plot the data
    for i, (ax, title, ylabel, coord) in enumerate(zip(axs, titles, ylabels, ["X", "Y", "Z"])):
        for idx, (start, end) in enumerate(reach_speed_segments):
            mask = (time >= start) & (time <= end)
            color = all_colors[idx % 16]  # Cycle through the 16 colors
            ax.plot(time[mask] - start, traj_data[f"{marker_name}_{coord}"][mask], 
                    alpha=0.7, label=f"Reach {idx + 1}", color=color)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    axs[-1].set_xlabel("Time (s)")
    axs[0].legend(title="Reach Order", loc="upper right")
    plt.tight_layout()

    if save_path:
        unique_name = f"{marker_name}_aligned_segments_xyz_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        fig.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def plot_reach_acceleration_with_ldlj_normalised(time, Traj_Space_data, marker_name, reach_speed_segments, LDLJ_values, save_path):
    """
    Plots 16 reach segments in a 4x4 format, overlays normalised position, speed, acceleration, and jerk, 
    and indicates their LDLJ as the title.

    Args:
        time (pd.Series): Time for each frame.
        Traj_Space_data (dict): Dictionary containing trajectory space data (position, speed, acceleration, jerk).
        marker_name (str): Marker name to plot (e.g., "RFIN").
        reach_speed_segments (list): List of (start, end) tuples for reach speed segments.
        LDLJ_values (list): List of LDLJ values for each reach segment.
        save_path (str): Directory to save the figure.
    """
    # Ensure LDLJ_values matches the number of reach_speed_segments
    if len(LDLJ_values) != len(reach_speed_segments):
        raise ValueError("Mismatch between the number of reach_speed_segments and LDLJ_values.")

    position = Traj_Space_data[marker_name][0]
    speed = Traj_Space_data[marker_name][1]
    acceleration = Traj_Space_data[marker_name][2]
    jerk = Traj_Space_data[marker_name][3]

    fig, axs = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)
    axs = axs.flatten()

    for idx, (start, end) in enumerate(reach_speed_segments[:16]):
        mask = (time >= start) & (time <= end)
        segment_time = time[mask] - start
        segment_position = position[mask]
        segment_speed = speed[mask]
        segment_acceleration = acceleration[mask]
        segment_jerk = jerk[mask]

        # Normalise each component to have a maximum value of 1
        norm_position = segment_position / segment_position.max()
        norm_speed = segment_speed / segment_speed.max()
        norm_acceleration = segment_acceleration / segment_acceleration.max()
        norm_jerk = segment_jerk / segment_jerk.max()

        axs[idx].plot(segment_time, norm_acceleration, color="red", label="Acceleration")
        axs[idx].plot(segment_time, norm_jerk, color="orange", linestyle="--", label="Jerk")
        # axs[idx].plot(segment_time, norm_position, color="blue", label="Position")
        axs[idx].plot(segment_time, norm_speed, color="green", label="Speed")
        axs[idx].set_title(f"Reach {idx + 1}\nLDLJ: {LDLJ_values[idx]:.2f}" if LDLJ_values[idx] is not None else f"Reach {idx + 1}\nLDLJ: N/A")
        axs[idx].grid()
        axs[idx].legend()

    for ax in axs[len(reach_speed_segments):]:
        ax.axis("off")

    fig.suptitle(f"{marker_name} Reach Position, Speed, Acceleration, and Jerk (Normalised) with LDLJ", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        unique_name = f"{marker_name}_reach_position_speed_acceleration_jerk_ldlj_normalised_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        fig.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def plot_reach_speed_and_jerk(time, Traj_Space_data, marker_name, reach_speed_segments, LDLJ_values, save_path, file_path):
    """
    Plots 16 reach segments in a 4x4 format, overlaying speed and jerk with twin axes, 
    and indicates their LDLJ as the title. All speed and jerk y-axes use the same range.

    Args:
        time (pd.Series): Time for each frame.
        Traj_Space_data (dict): Dictionary containing trajectory space data (position, speed, acceleration, jerk).
        marker_name (str): Marker name to plot (e.g., "RFIN").
        reach_speed_segments (list): List of (start, end) tuples for reach speed segments.
        LDLJ_values (list): List of LDLJ values for each reach segment.
        save_path (str): Directory to save the figure.
    """
    # Ensure LDLJ_values matches the number of reach_speed_segments
    if len(LDLJ_values) != len(reach_speed_segments):
        raise ValueError("Mismatch between the number of reach_speed_segments and LDLJ_values.")

    speed = Traj_Space_data[marker_name][1]
    jerk = Traj_Space_data[marker_name][3]

    # Determine global y-axis ranges for speed and jerk
    global_speed_min, global_speed_max = speed.min(), speed.max()
    global_jerk_min, global_jerk_max = jerk.min(), jerk.max()

    fig, axs = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=False)
    axs = axs.flatten()

    for idx, (start, end) in enumerate(reach_speed_segments[:16]):
        mask = (time >= start) & (time <= end)
        segment_time = time[mask] - start

        ax = axs[idx]
        ax2 = ax.twinx()  # Create a twin axis for jerk

        # Plot speed on the primary axis
        ax.plot(segment_time, speed[mask], color="green", label="Speed")
        ax.set_ylabel("Speed (mm/s)", color="green")
        ax.tick_params(axis='y', labelcolor="green")
        ax.set_ylim(global_speed_min, global_speed_max)  # Set global y-axis range for speed

        # Plot jerk on the secondary axis
        ax2.plot(segment_time, jerk[mask], color="orange", linestyle="--", label="Jerk")
        ax2.set_ylabel("Jerk (mm/s³)", color="orange")
        ax2.tick_params(axis='y', labelcolor="orange")
        ax2.set_ylim(global_jerk_min, global_jerk_max)  # Set global y-axis range for jerk

        # Set title with LDLJ value
        ax.set_title(f"Reach {idx + 1}\nLDLJ: {LDLJ_values[idx]:.2f}" if LDLJ_values[idx] is not None else f"Reach {idx + 1}\nLDLJ: N/A")
        ax.grid()

    # Turn off unused subplots
    for ax in axs[len(reach_speed_segments):]:
        ax.axis("off")

    fig.suptitle(f"{marker_name} Reach Speed and Jerk with LDLJ ({os.path.basename(file_path)})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        unique_name = f"{marker_name}_reach_speed_jerk_ldlj_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        fig.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def plot_correlation_matrix(LDLJ_values, reach_distances, path_distances, v_peaks, acc_peaks, reach_durations, jerk_peaks, speed_threshold, save_path, file_path):
    """
    Plots a heatmap of the Pearson correlation matrix for various metrics, masking non-significant correlations.

    Args:
        LDLJ_values (list): List of LDLJ values for each reach segment.
        reach_distances (list): List of reach distances for each reach segment.
        path_distances (list): List of Path distances traveled for each reach segment.
        v_peaks (list): List of peak velocities for each reach segment.
        acc_peaks (list): List of peak accelerations for each reach segment.
        reach_durations (list): List of reach durations for each reach segment.
        jerk_peaks (list): List of peak jerks for each reach segment.
        speed_threshold (float): Speed threshold used in the analysis.
        save_path (str): Directory to save the heatmap plot.
    """
    # Create a DataFrame for the metrics
    data = {
        "LDLJ": LDLJ_values,
        "Reach Distance": reach_distances,
        "Path Distance": path_distances,
        "Velocity Peaks": v_peaks,
        "Acceleration Peaks": acc_peaks,
        "Jerk Peaks": jerk_peaks,
        "Reach Durations": reach_durations,
    }
    df = pd.DataFrame(data)

    # Calculate the Pearson correlation matrix and p-values
    correlation_matrix = df.corr()
    p_values = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(len(df.columns))

    # Mask non-significant correlations (p-value > 0.05) and upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool)) | (p_values > 0.05)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, mask=mask)
    plt.title(f"Correlation Matrix Heatmap (Non-Significant Correlations Masked)({os.path.basename(file_path)})\nSpeed Threshold: {speed_threshold}")
    plt.tight_layout()

    if save_path:
        unique_name = f"correlation_matrix_heatmap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        plt.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def plot_split_segments_speed(time, Traj_Space_data, marker_name, start_to_peak_segments, peak_to_end_segments, save_path):
    """
    Plots speed for start-to-peak and peak-to-end segments in the same figure with different colors of highlights.

    Args:
        time (pd.Series): Time for each frame.
        Traj_Space_data (dict): Dictionary containing trajectory space data.
        marker_name (str): Marker name to plot (e.g., "RFIN").
        start_to_peak_segments (list): List of (start_time, peak_time) tuples for start-to-peak segments.
        peak_to_end_segments (list): List of (peak_time, end_time) tuples for peak-to-end segments.
        save_path (str, optional): Directory to save the figure. If None, the figure is not saved.
    """
    speed = Traj_Space_data[marker_name][1]  # Extract speed data

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, speed, label=f"{marker_name} Speed", color="blue")

    for start, peak in start_to_peak_segments:
        mask = (time >= start) & (time <= peak)
        ax.plot(time[mask], speed[mask], color="green", label="Start-to-Peak Segment" if start == start_to_peak_segments[0][0] else None)

    for peak, end in peak_to_end_segments:
        mask = (time >= peak) & (time <= end)
        ax.plot(time[mask], speed[mask], color="orange", label="Peak-to-End Segment" if peak == peak_to_end_segments[0][0] else None)

    ax.set_title(f"{marker_name} Speed with Split Segments")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (mm/s)")
    ax.legend()
    ax.grid()

    plt.tight_layout()

    if save_path:
        unique_name = f"{marker_name}_split_segments_speed_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        fig.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def plot_combined_correlations(reach_durations, reach_distances, path_distances, v_peaks, LDLJ_values,
                                corr_dur_dist, p_dur_dist, corr_dur_path, p_dur_path,
                                corr_dist_vpeaks, p_dist_vpeaks, corr_dist_ldlj, p_dist_ldlj,
                                save_path, file_paths):
    """
    Plots combined correlation subplots for various metrics with specific colors and shapes for each value.

    Args:
        reach_durations (list): List of reach durations.
        reach_distances (list): List of reach distances for each reach segment.
        path_distances (list): List of path distances traveled for each reach segment.
        v_peaks (list): List of peak velocities for each reach segment.
        LDLJ_values (list): List of LDLJ values for each reach segment.
        corr_dur_dist (float): Correlation coefficient for reach durations vs reach distances.
        p_dur_dist (float): P-value for reach durations vs reach distances.
        corr_dur_path (float): Correlation coefficient for reach durations vs path distances.
        p_dur_path (float): P-value for reach durations vs path distances.
        corr_dist_vpeaks (float): Correlation coefficient for reach distances vs peak velocities.
        p_dist_vpeaks (float): P-value for reach distances vs peak velocities.
        corr_dist_ldlj (float): Correlation coefficient for reach distances vs LDLJ values.
        p_dist_ldlj (float): P-value for reach distances vs LDLJ values.
        save_path (str): Directory to save the combined plot.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()

    # Define colors and shapes
    base_colors = ["orange", "green", "blue", "purple"]
    base_shapes = ["^", "s", "p", "h"]

    # Define reach-specific colors and shapes
    base_reach_styles = {}

    # Populate the dictionary
    reach_number = 1
    for shape in base_shapes:
        for color in base_colors:
            base_reach_styles[f"reach {reach_number}"] = {
                "color": color,
                "shape": shape
            }
            reach_number += 1

    reach_styles = {}
    total_reaches = len(reach_durations)
    for i in range(total_reaches):
        reach_styles[f"reach {i + 1}"] = base_reach_styles[f"reach {(i % 16) + 1}"]

    # Plot Reach Duration vs Reach Distance
    for i, (duration, distance) in enumerate(zip(reach_durations, reach_distances)):
        style = reach_styles[f"reach {i + 1}"]
        axs[0].scatter(duration, distance, color=style["color"], marker=style["shape"], label=f'{i + 1}', s=40)
    axs[0].set_title(f'Reach Duration vs Reach Distance\nCorrelation: {corr_dur_dist:.2f}, P-value: {p_dur_dist:.3f}')
    axs[0].set_xlabel('Reach Duration (s)')
    axs[0].set_ylabel('Reach Distance (mm)')
    axs[0].grid(True)

    # Plot Reach Duration vs Path Distance
    for i, (duration, path) in enumerate(zip(reach_durations, path_distances)):
        style = reach_styles[f"reach {i + 1}"]
        axs[1].scatter(duration, path, color=style["color"], marker=style["shape"], label=f'{i + 1}', s=40)
    axs[1].set_title(f'Reach Duration vs Path Distance\nCorrelation: {corr_dur_path:.2f}, P-value: {p_dur_path:.3f}')
    axs[1].set_xlabel('Reach Duration (s)')
    axs[1].set_ylabel('Path Distance (mm)')
    axs[1].grid(True)

    # Plot Reach Distance vs Peak Velocity
    for i, (distance, v_peak) in enumerate(zip(reach_distances, v_peaks)):
        style = reach_styles[f"reach {i + 1}"]
        axs[2].scatter(distance, v_peak, color=style["color"], marker=style["shape"], label=f'{i + 1}', s=40)
    axs[2].set_title(f'Reach Distance vs Peak Velocity\nCorrelation: {corr_dist_vpeaks:.2f}, P-value: {p_dist_vpeaks:.3f}')
    axs[2].set_xlabel('Reach Distance (mm)')
    axs[2].set_ylabel('Peak Velocity (mm/s)')
    axs[2].grid(True)

    # Plot Reach Distance vs LDLJ
    for i, (distance, ldlj) in enumerate(zip(reach_distances, LDLJ_values)):
        style = reach_styles[f"reach {i + 1}"]
        axs[3].scatter(distance, ldlj, color=style["color"], marker=style["shape"], label=f'{i + 1}', s=40)
    axs[3].set_title(f'Reach Distance vs LDLJ\nCorrelation: {corr_dist_ldlj:.2f}, P-value: {p_dist_ldlj:.3f}')
    axs[3].set_xlabel('Reach Distance (mm)')
    axs[3].set_ylabel('LDLJ Values')
    axs[3].grid(True)

    # Add legend for all reaches
    handles, labels = axs[0].get_legend_handles_labels()

    N = len(handles)
    ncol = 4
    nrows = (N + ncol - 1) // ncol

    pairs = list(zip(handles[::-1], labels[::-1]))
    grid = [pairs[i * ncol:(i + 1) * ncol] for i in range(nrows)]

    # Fill columns from the row-wise grid
    new_order = [grid[row][col] 
                for col in range(ncol) 
                for row in range(nrows) 
                if col < len(grid[row])]

    handles_new, labels_new = zip(*new_order)
    

    if len(reach_styles) == 16:
        legend_title = f"File: {os.path.basename(file_paths)}"
    else:
        legend_title = f"Files: {', '.join([os.path.basename(fp) for fp in file_paths])}"

    legend = fig.legend(handles_new, labels_new, loc="center", ncol=ncol,
                        title=legend_title,
                        fontsize=5,
                        bbox_to_anchor=(0.525, 0.96))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        unique_name = f"combined_correlations_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        plt.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def rank_and_visualize_ldlj(reach_speed_segments, LDLJ_values, save_path, file_path):
    """
    Ranks reach segments by LDLJ values and visualizes them in a bar plot.

    Args:
        reach_speed_segments (list): List of (start, end) tuples for reach speed segments.
        LDLJ_values (list): List of LDLJ values for each reach segment.
        save_path (str): Directory to save the visualization plot.
    """
    # Ensure LDLJ_values matches the number of reach_speed_segments
    if len(LDLJ_values) != len(reach_speed_segments):
        raise ValueError("Mismatch between the number of reach_speed_segments and LDLJ_values.")

    # Rank LDLJ values and sort reach segments accordingly
    ranked_indices = sorted(range(len(LDLJ_values)), key=lambda i: LDLJ_values[i], reverse=True)
    ranked_segments = [reach_speed_segments[i] for i in ranked_indices]
    ranked_ldlj = [LDLJ_values[i] for i in ranked_indices]


    # Create a bar plot for ranked LDLJ values
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(ranked_ldlj) + 1), ranked_ldlj, color="blue", alpha=0.7)
    plt.xlabel("Ranked Reach Segments")
    plt.ylabel("LDLJ Values")
    plt.title(f"Ranked Reach Segments by LDLJ Values ({os.path.basename(file_path)})")
    plt.xticks(range(1, len(ranked_ldlj) + 1), [f"Reach {i + 1}" for i in ranked_indices], rotation=45)
    plt.tight_layout()

    if save_path:
        unique_name = f"ranked_ldlj_visualization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        plt.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

'''for combined files'''

def summarize_data_across_files(results):
    """
    Summarizes data across all files by aggregating reach durations, distances, path distances, 
    peak velocities, LDLJ values, peak accelerations, and peak jerks.

    Args:
        results (dict): Dictionary containing results from multiple files. Each file's result should 
                        have a 'parameters' key with relevant metrics.

    Returns:
        dict: A dictionary containing aggregated lists for each metric.
    """
    summary = {
        "reach_durations": [],
        "reach_distances": [],
        "path_distances": [],
        "v_peaks": [],
        "LDLJ_values": [],
        "acc_peaks": [],
        "jerk_peaks": [],
    }

    for file_result in results.values():
        summary["reach_durations"].extend(file_result["parameters"]["reach_durations"])
        summary["reach_distances"].extend(file_result["parameters"]["reach_distances"])
        summary["path_distances"].extend(file_result["parameters"]["path_distances"])
        summary["v_peaks"].extend(file_result["parameters"]["v_peaks"])
        summary["LDLJ_values"].extend(file_result["parameters"]["LDLJ_values"])
        summary["acc_peaks"].extend(file_result["parameters"]["acc_peaks"])
        summary["jerk_peaks"].extend(file_result["parameters"]["jerk_peaks"])

    return summary

def plot_summarize_correlation_matrix(summary, save_path, file_paths):
    """
    Plots a heatmap of the Pearson correlation matrix for metrics, masking non-significant correlations and upper triangle.

    Args:
        summary (dict): Dictionary containing aggregated metrics.
        save_path (str, optional): Directory to save the plot.
    """
    metrics = np.array([
        summary["LDLJ_values"], summary["reach_distances"], summary["path_distances"],
        summary["v_peaks"], summary["acc_peaks"], summary["jerk_peaks"], summary["reach_durations"]
    ])
    correlation_matrix = np.corrcoef(metrics)
    significance_mask = np.zeros_like(correlation_matrix, dtype=bool)

    for i in range(metrics.shape[0]):
        for j in range(metrics.shape[0]):
            if i != j:
                _, p_value = pearsonr(metrics[i], metrics[j])
                significance_mask[i, j] = p_value > 0.05
            if i <= j:
                significance_mask[i, j] = True

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", mask=significance_mask, cmap="coolwarm",
                xticklabels=["LDLJ", "Reach Distance", "Path Distance", "Velocity Peaks", "Acceleration Peaks", "Jerk Peaks", "Reach Durations"],
                yticklabels=["LDLJ", "Reach Distance", "Path Distance", "Velocity Peaks", "Acceleration Peaks", "Jerk Peaks", "Reach Durations"])
    plt.title(f"Pearson Correlation Matrix (Masked Non-Significant)\nFiles: {', '.join([os.path.basename(fp) for fp in file_paths])}")

    plt.tight_layout()

    if save_path:
        unique_name = f"summary_correlation_matrix_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        plt.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def rank_and_visualize_ldlj_files(results, save_path):
    """
    Ranks reach segments by LDLJ values for each file and visualizes them as subplots.

    Args:
        results (dict): Dictionary containing results from multiple files. Each file's result should 
                        have a 'parameters' key with LDLJ_values and reach_speed_segments.
        save_path (str): Directory to save the visualization plot.
    """
    num_files = len(results)
    fig, axs = plt.subplots(num_files, 1, figsize=(12, 6 * num_files), sharex=False)

    if num_files == 1:
        axs = [axs]  # Ensure axs is iterable for a single subplot

    for ax, (file_path, file_result) in zip(axs, results.items()):
        LDLJ_values = file_result["parameters"]["LDLJ_values"]

        # Rank LDLJ values and sort reach segments accordingly
        ranked_indices = sorted(range(len(LDLJ_values)), key=lambda i: LDLJ_values[i], reverse=True)
        ranked_ldlj = [LDLJ_values[i] for i in ranked_indices]

        # Create a bar plot for ranked LDLJ values
        bars = ax.bar(range(1, len(ranked_ldlj) + 1), ranked_ldlj, color="blue", alpha=0.7)
        ax.set_xlabel("Reach Number")
        ax.set_ylabel("LDLJ Values")
        ax.set_title(f"Ranked Reach Segments by LDLJ Values\nFile: {os.path.basename(file_path)}")
        ax.set_xticks(range(1, len(ranked_ldlj) + 1))
        ax.set_xticklabels([f"Reach {i + 1}" for i in ranked_indices], rotation=45)
        ax.grid()

        # Add LDLJ value on top of each bar
        for bar, value in zip(bars, ranked_ldlj):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", 
                    ha="center", va="bottom", fontsize=10)

    plt.tight_layout()

    if save_path:
        unique_name = f"ranked_ldlj_visualization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        plt.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()


'''Dead code, kept for reference'''

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

def plot_aligned_segments_red_blue(time, Traj_Space_data, reach_speed_segments, marker_name, save_path):
    """
    Plots aligned position, speed, acceleration, and jerk segments for a marker in a single figure with 4 subplots.

    Args:
        time (pd.Series): Time for each frame.
        Traj_Space_data (dict): Dictionary containing trajectory space data (position, speed, acceleration, jerk).
        reach_speed_segments (list): List of (start, end) tuples for reach speed segments.
        marker_name (str): Marker name to plot (e.g., "RFIN").
        save_path (str): Directory to save the figure.
    """

    # Define color gradients for red (1-8) and blue (9-16)
    red_colors = list(mcolors.LinearSegmentedColormap.from_list("", ["lightcoral", "red"])(range(8)))
    blue_colors = list(mcolors.LinearSegmentedColormap.from_list("", ["lightblue", "blue"])(range(8)))

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    titles = ["Aligned Position Segments", "Aligned Speed Segments", 
                "Aligned Acceleration Segments", "Aligned Jerk Segments"]
    ylabels = ["Position (mm)", "Speed (mm/s)", "Acceleration (mm/s²)", "Jerk (mm/s³)"]

    # Iterate through the subplots and plot the data
    for i, (ax, title, ylabel) in enumerate(zip(axs, titles, ylabels)):
        for idx, (start, end) in enumerate(reach_speed_segments):
            mask = (time >= start) & (time <= end)
            color = red_colors[idx] if idx < 8 else blue_colors[idx - 8]
            ax.plot(time[mask] - start, Traj_Space_data[marker_name][i][mask], 
                    alpha=0.7, label=f"Reach {idx + 1}", color=color)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    axs[-1].set_xlabel("Time (s)")
    axs[0].legend(title="Reach Order", loc="upper right")
    plt.tight_layout()

    if save_path:
        unique_name = f"{marker_name}_aligned_segments_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        fig.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def calculate_ldlj_for_all_reaches_constant_speed(marker_name, time, reach_speed_segments, save_path=None): # for understand LDLJ only
    """
    Calculates the Log Dimensionless Jerk (LDLJ) for all reach segments of a given marker,
    assuming a constant speed of 500 mm/s, and plots speed and acceleration with LDLJ as the title.

    Args:
        marker_name (str): Marker name to calculate LDLJ for (e.g., "RFIN").
        time (pd.Series): Time for each frame.
        reach_speed_segments (list): List of (start_time, end_time) tuples for reach speed segments.
        save_path (str, optional): Directory to save the figure. If None, the figure is not saved.

    Returns:
        list: A list of LDLJ values for each reach segment.
    """
    speed = np.full(len(time), 500 + np.random.uniform(-6, 0.1, len(time)))  # Constant speed of 500 mm/s with ±2 variation
    acceleration = np.gradient(speed, time)  # Calculate acceleration as the gradient of speed
    jerk = np.gradient(acceleration, time)  # Calculate jerk as the gradient of acceleration

    LDLJ_values = []

    fig, axs = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)
    axs = axs.flatten()

    for idx, segment in enumerate(reach_speed_segments[:16]):
        # Convert Tstart and Tend from time values to indices
        Tstart_idx = time[time >= segment[0]].index[0]
        Tend_idx = time[time >= segment[1]].index[0]

        # Calculate the total movement duration
        movement_duration = time.iloc[Tend_idx] - time.iloc[Tstart_idx]

        # Calculate the integral of the squared jerk
        jerk_squared_integral = np.trapezoid(jerk[Tstart_idx:Tend_idx]**2, time.iloc[Tstart_idx:Tend_idx])

        # Find the peak velocity (constant in this case)
        v_peak = speed[Tstart_idx:Tend_idx].max()

        # Avoid division by zero
        if movement_duration == 0 or v_peak == 0:
            LDLJ_values.append(None)  # Append None if calculation is not possible
            continue

        # Calculate the dimensionless jerk
        dimensionless_jerk = (jerk_squared_integral * movement_duration**3) / (v_peak**2)

        # Calculate and append the Log Dimensionless Jerk (LDLJ)
        LDLJ = -np.log(dimensionless_jerk)
        LDLJ_values.append(LDLJ)

        # Plot speed and acceleration for the current segment
        axs[idx].plot(time.iloc[Tstart_idx:Tend_idx] - time.iloc[Tstart_idx], speed[Tstart_idx:Tend_idx], label="Speed", color="blue")
        axs[idx].plot(time.iloc[Tstart_idx:Tend_idx] - time.iloc[Tstart_idx], acceleration[Tstart_idx:Tend_idx], label="Acceleration", color="green")
        axs[idx].set_title(f"Reach {idx + 1}\nLDLJ: {LDLJ:.2f}" if LDLJ is not None else f"Reach {idx + 1}\nLDLJ: N/A")
        axs[idx].grid()
        axs[idx].legend()

    for ax in axs[len(reach_speed_segments):]:
        ax.axis("off")

    fig.suptitle(f"{marker_name} Speed and Acceleration with LDLJ", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        unique_name = f"{marker_name}_speed_acceleration_ldlj_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        fig.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

    return LDLJ_values

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

def plot_reach_acceleration_with_ldlj(time, Traj_Space_data, marker_name, reach_speed_segments, LDLJ_values, save_path):
    """
    Plots 16 reach segments in a 4x4 format, overlays position, speed, acceleration, and jerk, 
    and indicates their LDLJ as the title.

    Args:
        time (pd.Series): Time for each frame.
        Traj_Space_data (dict): Dictionary containing trajectory space data (position, speed, acceleration, jerk).
        marker_name (str): Marker name to plot (e.g., "RFIN").
        reach_speed_segments (list): List of (start, end) tuples for reach speed segments.
        LDLJ_values (list): List of LDLJ values for each reach segment.
        save_path (str): Directory to save the figure.
    """
    # Ensure LDLJ_values matches the number of reach_speed_segments
    if len(LDLJ_values) != len(reach_speed_segments):
        raise ValueError("Mismatch between the number of reach_speed_segments and LDLJ_values.")

    position = Traj_Space_data[marker_name][0]
    speed = Traj_Space_data[marker_name][1]
    acceleration = Traj_Space_data[marker_name][2]
    jerk = Traj_Space_data[marker_name][3]

    fig, axs = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True)
    axs = axs.flatten()

    for idx, (start, end) in enumerate(reach_speed_segments[:16]):
        mask = (time >= start) & (time <= end)
        axs[idx].plot(time[mask] - start, acceleration[mask], color="red", label="Acceleration")
        axs[idx].plot(time[mask] - start, jerk[mask], color="orange", linestyle="--", label="Jerk")
        axs[idx].plot(time[mask] - start, position[mask], color="blue", label="Position")
        axs[idx].plot(time[mask] - start, speed[mask], color="green", label="Speed")
        axs[idx].set_title(f"Reach {idx + 1}\nLDLJ: {LDLJ_values[idx]:.2f}" if LDLJ_values[idx] is not None else f"Reach {idx + 1}\nLDLJ: N/A")
        axs[idx].grid()
        axs[idx].legend()

    for ax in axs[len(reach_speed_segments):]:
        ax.axis("off")

    fig.suptitle(f"{marker_name} Reach Position, Speed, Acceleration, and Jerk with LDLJ", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        unique_name = f"{marker_name}_reach_position_speed_acceleration_jerk_ldlj_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        fig.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def plot_combined_correlations_past(reach_durations, LDLJ_values, LDLJ_correlation, LDLJ_p_value, 
                                reach_distances, reach_distance_correlation, reach_distance_p_value, 
                                path_distances, path_distances_correlation, path_distances_p_value, 
                                v_peaks, v_peaks_correlation, v_peaks_p_value, 
                                acc_peaks, acc_peaks_correlation, acc_peaks_p_value, 
                                save_path):
    """
    Plots combined correlation subplots for various metrics against reach durations.

    Args:
        reach_durations (list): List of reach durations.
        LDLJ_values (list): List of LDLJ values for each reach segment.
        LDLJ_correlation (float): Correlation coefficient for LDLJ vs reach durations.
        LDLJ_p_value (float): P-value for LDLJ vs reach durations.
        reach_distances (list): List of reach distances for each reach segment.
        reach_distance_correlation (float): Correlation coefficient for reach distances vs durations.
        reach_distance_p_value (float): P-value for reach distances vs durations.
        path_distances (list): List of path distances traveled for each reach segment.
        path_distances_correlation (float): Correlation coefficient for path distances vs durations.
        path_distances_p_value (float): P-value for path distances vs durations.
        v_peaks (list): List of peak velocities for each reach segment.
        v_peaks_correlation (float): Correlation coefficient for peak velocities vs durations.
        v_peaks_p_value (float): P-value for peak velocities vs durations.
        acc_peaks (list): List of peak accelerations for each reach segment.
        acc_peaks_correlation (float): Correlation coefficient for peak accelerations vs durations.
        acc_peaks_p_value (float): P-value for peak accelerations vs durations.
        save_path (str): Directory to save the combined plot.
    """
    fig, axs = plt.subplots(2, 3, figsize=(10, 10))
    axs = axs.flatten()

    # Plot LDLJ vs Reach Duration
    colors = cm.Reds(np.linspace(0.3, 1, len(reach_durations)))
    for i, (duration, ldlj, color) in enumerate(zip(reach_durations, LDLJ_values, colors)):
        axs[0].scatter(duration, ldlj, color=color, label=f'Reach {i + 1}', s=50)
    axs[0].set_title(f'LDLJ vs Reach Duration\nCorrelation: {LDLJ_correlation:.2f}, P-value: {LDLJ_p_value:.3f}')
    axs[0].set_xlabel('Reach Duration (s)')
    axs[0].set_ylabel('LDLJ Values')
    axs[0].grid(True)

    # Plot Reach Distance vs Reach Duration
    colors = cm.Blues(np.linspace(0.3, 1, len(reach_distances)))
    for i, (distance, duration, color) in enumerate(zip(reach_distances, reach_durations, colors)):
        axs[1].scatter(distance, duration, color=color, label=f'Reach {i + 1}', s=50)
    axs[1].set_title(f'Reach Distance vs Reach Duration\nCorrelation: {reach_distance_correlation:.2f}, P-value: {reach_distance_p_value:.3f}')
    axs[1].set_xlabel('Reach Distance (mm)')
    axs[1].set_ylabel('Reach Duration (s)')
    axs[1].grid(True)

    # Plot Path Distance vs Reach Duration
    colors = cm.Greens(np.linspace(0.3, 1, len(path_distances)))
    for i, (distance, duration, color) in enumerate(zip(path_distances, reach_durations, colors)):
        axs[2].scatter(distance, duration, color=color, label=f'Reach {i + 1}', s=50)
    axs[2].set_title(f'Path Distance vs Reach Duration\nCorrelation: {path_distances_correlation:.2f}, P-value: {path_distances_p_value:.3f}')
    axs[2].set_xlabel('Path Distance (mm)')
    axs[2].set_ylabel('Reach Duration (s)')
    axs[2].grid(True)

    # Plot Peak Velocity vs Reach Duration
    colors = cm.Purples(np.linspace(0.3, 1, len(v_peaks)))
    for i, (v_peak, duration, color) in enumerate(zip(v_peaks, reach_durations, colors)):
        axs[3].scatter(v_peak, duration, color=color, label=f'Reach {i + 1}', s=50)
    axs[3].set_title(f'Peak Velocity vs Reach Duration\nCorrelation: {v_peaks_correlation:.2f}, P-value: {v_peaks_p_value:.3f}')
    axs[3].set_xlabel('Peak Velocity (mm/s)')
    axs[3].set_ylabel('Reach Duration (s)')
    axs[3].grid(True)

    # Plot Peak Acceleration vs Reach Duration
    colors = cm.Oranges(np.linspace(0.3, 1, len(acc_peaks)))
    for i, (acc_peak, duration, color) in enumerate(zip(acc_peaks, reach_durations, colors)):
        axs[4].scatter(acc_peak, duration, color=color, label=f'Reach {i + 1}', s=50)
    axs[4].set_title(f'Peak Acceleration vs Reach Duration\nCorrelation: {acc_peaks_correlation:.2f}, P-value: {acc_peaks_p_value:.3f}')
    axs[4].set_xlabel('Peak Acceleration (mm/s²)')
    axs[4].set_ylabel('Reach Duration (s)')
    axs[4].grid(True)

    # Hide unused subplot
    axs[5].axis("off")

    plt.tight_layout()

    if save_path:
        unique_name = f"combined_correlations_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        plt.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def plot_combined_correlations_color(reach_durations, reach_distances, path_distances, v_peaks, LDLJ_values,
                                corr_dur_dist, p_dur_dist, corr_dur_path, p_dur_path,
                                corr_dist_vpeaks, p_dist_vpeaks, corr_dist_ldlj, p_dist_ldlj,
                                save_path):
    """
    Plots combined correlation subplots for various metrics.

    Args:
        reach_durations (list): List of reach durations.
        reach_distances (list): List of reach distances for each reach segment.
        path_distances (list): List of path distances traveled for each reach segment.
        v_peaks (list): List of peak velocities for each reach segment.
        LDLJ_values (list): List of LDLJ values for each reach segment.
        corr_dur_dist (float): Correlation coefficient for reach durations vs reach distances.
        p_dur_dist (float): P-value for reach durations vs reach distances.
        corr_dur_path (float): Correlation coefficient for reach durations vs path distances.
        p_dur_path (float): P-value for reach durations vs path distances.
        corr_dist_vpeaks (float): Correlation coefficient for reach distances vs peak velocities.
        p_dist_vpeaks (float): P-value for reach distances vs peak velocities.
        corr_dist_ldlj (float): Correlation coefficient for reach distances vs LDLJ values.
        p_dist_ldlj (float): P-value for reach distances vs LDLJ values.
        save_path (str): Directory to save the combined plot.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()

    # Plot Reach Duration vs Reach Distance
    colors = cm.Blues(np.linspace(0.3, 1, len(reach_durations)))
    for i, (duration, distance, color) in enumerate(zip(reach_durations, reach_distances, colors)):
        axs[0].scatter(duration, distance, color=color, label=f'Reach {i + 1}', s=50)
    axs[0].set_title(f'Reach Duration vs Reach Distance\nCorrelation: {corr_dur_dist:.2f}, P-value: {p_dur_dist:.3f}')
    axs[0].set_xlabel('Reach Duration (s)')
    axs[0].set_ylabel('Reach Distance (mm)')
    axs[0].grid(True)

    # Plot Reach Duration vs Path Distance
    colors = cm.Greens(np.linspace(0.3, 1, len(reach_durations)))
    for i, (duration, path, color) in enumerate(zip(reach_durations, path_distances, colors)):
        axs[1].scatter(duration, path, color=color, label=f'Reach {i + 1}', s=50)
    axs[1].set_title(f'Reach Duration vs Path Distance\nCorrelation: {corr_dur_path:.2f}, P-value: {p_dur_path:.3f}')
    axs[1].set_xlabel('Reach Duration (s)')
    axs[1].set_ylabel('Path Distance (mm)')
    axs[1].grid(True)

    # Plot Reach Distance vs Peak Velocity
    colors = cm.Purples(np.linspace(0.3, 1, len(reach_distances)))
    for i, (distance, v_peak, color) in enumerate(zip(reach_distances, v_peaks, colors)):
        axs[2].scatter(distance, v_peak, color=color, label=f'Reach {i + 1}', s=50)
    axs[2].set_title(f'Reach Distance vs Peak Velocity\nCorrelation: {corr_dist_vpeaks:.2f}, P-value: {p_dist_vpeaks:.3f}')
    axs[2].set_xlabel('Reach Distance (mm)')
    axs[2].set_ylabel('Peak Velocity (mm/s)')
    axs[2].grid(True)

    # Plot Reach Distance vs LDLJ
    colors = cm.Reds(np.linspace(0.3, 1, len(reach_distances)))
    for i, (distance, ldlj, color) in enumerate(zip(reach_distances, LDLJ_values, colors)):
        axs[3].scatter(distance, ldlj, color=color, label=f'Reach {i + 1}', s=50)
    axs[3].set_title(f'Reach Distance vs LDLJ\nCorrelation: {corr_dist_ldlj:.2f}, P-value: {p_dist_ldlj:.3f}')
    axs[3].set_xlabel('Reach Distance (mm)')
    axs[3].set_ylabel('LDLJ Values')
    axs[3].grid(True)

    plt.tight_layout()

    if save_path:
        unique_name = f"combined_correlations_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        plt.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def plot_reach_duration_vs_ldlj(reach_durations, LDLJ_values, correlation, p_value, save_path):
    """
    Plots the correlation between reach durations and LDLJ values.

    Args:
        reach_durations (list): List of reach durations.
        LDLJ_values (list): List of LDLJ values for each reach segment.
        correlation (float): Pearson correlation coefficient.
        p_value (float): P-value of the correlation.
        save_path (str): Directory to save the correlation plot.
    """
    # Generate a colormap for the reaches (light to dark red)
    colors = cm.Reds(np.linspace(0.3, 1, len(reach_durations)))

    # Plot the correlation with colored reaches
    plt.figure(figsize=(10, 6))
    for i, (duration, ldlj, color) in enumerate(zip(reach_durations, LDLJ_values, colors)):
        plt.scatter(duration, ldlj, color=color, label=f'Reach {i + 1}', s=50)

    plt.xlabel('Reach Duration (s)')
    plt.ylabel('LDLJ Values')
    plt.title(f'Correlation between Reach Duration and LDLJ Values\n'
              f'Correlation: {correlation:.2f}, P-value: {p_value:.3f}')
    plt.grid(True)
    plt.legend(title="Reaches", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        unique_name = f"reach_duration_vs_ldlj_correlation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        plt.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def plot_reach_distance_vs_duration(reach_distances, reach_durations, correlation, p_value, save_path):
    """
    Plots the correlation between reach distances and reach durations.

    Args:
        reach_distances (list): List of reach distances.
        reach_durations (list): List of reach durations.
        correlation (float): Pearson correlation coefficient.
        p_value (float): P-value of the correlation.
        save_path (str): Directory to save the correlation plot.
    """
    # Generate a colormap for the reaches (light to dark blue)
    colors = cm.Blues(np.linspace(0.3, 1, len(reach_distances)))

    # Plot the correlation with colored reaches
    plt.figure(figsize=(10, 6))
    for i, (distance, duration, color) in enumerate(zip(reach_distances, reach_durations, colors)):
        plt.scatter(distance, duration, color=color, label=f'Reach {i + 1}', s=50)

    plt.xlabel('Reach Distance (mm)')
    plt.ylabel('Reach Duration (s)')
    plt.title(f'Correlation between Reach Distance and Reach Duration\n'
                f'Correlation: {correlation:.2f}, P-value: {p_value:.3f}')
    plt.grid(True)
    plt.legend(title="Reaches", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        unique_name = f"reach_distance_vs_duration_correlation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        plt.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def plot_path_distances_vs_duration(path_distances, reach_durations, correlation, p_value, save_path):
    """
    Plots the correlation between path_distances and reach durations.

    Args:
        path_distances (list): List of path_distances traveled for each reach segment.
        reach_durations (list): List of reach durations.
        correlation (float): Pearson correlation coefficient.
        p_value (float): P-value of the correlation.
        save_path (str): Directory to save the correlation plot.
    """
    # Generate a colormap for the reaches (light to dark green)
    colors = cm.Greens(np.linspace(0.3, 1, len(path_distances)))

    # Plot the correlation with colored reaches
    plt.figure(figsize=(10, 6))
    for i, (distance, duration, color) in enumerate(zip(path_distances, reach_durations, colors)):
        plt.scatter(distance, duration, color=color, label=f'Reach {i + 1}', s=50)

    plt.xlabel('Path Distance (mm)')
    plt.ylabel('Reach Duration (s)')
    plt.title(f'Correlation between Path Distance and Reach Duration\n'
                f'Correlation: {correlation:.2f}, P-value: {p_value:.3f}')
    plt.grid(True)
    plt.legend(title="Reaches", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        unique_name = f"path_distances_vs_duration_correlation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        plt.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def plot_v_peak_vs_duration(v_peak, reach_durations, v_peak_correlation, v_peak_p_value, save_path):
    """
    Plots the correlation between peak velocities and reach durations.

    Args:
        v_peak (list): List of peak velocities for each reach segment.
        reach_durations (list): List of reach durations.
        v_peak_correlation (float): Pearson correlation coefficient.
        v_peak_p_value (float): P-value of the correlation.
        save_path (str): Directory to save the correlation plot.
    """
    # Generate a colormap for the reaches (light to dark purple)
    colors = cm.Purples(np.linspace(0.3, 1, len(v_peak)))

    # Plot the correlation with colored reaches
    plt.figure(figsize=(10, 6))
    for i, (v_peak, duration, color) in enumerate(zip(v_peak, reach_durations, colors)):
        plt.scatter(v_peak, duration, color=color, label=f'Reach {i + 1}', s=50)

    plt.xlabel('Peak Velocity (mm/s)')
    plt.ylabel('Reach Duration (s)')
    plt.title(f'Correlation between Peak Velocity and Reach Duration\n'
                f'Correlation: {v_peak_correlation:.2f}, P-value: {v_peak_p_value:.3f}')
    plt.grid(True)
    plt.legend(title="Reaches", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        unique_name = f"v_peak_vs_duration_correlation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        plt.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()

def plot_acc_peak_vs_duration(acc_peaks, reach_durations, acc_peaks_correlation, acc_peaks_p_value, save_path):
    """
    Plots the correlation between peak accelerations and reach durations.

    Args:
        acc_peaks (list): List of peak accelerations for each reach segment.
        reach_durations (list): List of reach durations.
        acc_peaks_correlation (float): Pearson correlation coefficient.
        acc_peaks_p_value (float): P-value of the correlation.
        save_path (str): Directory to save the correlation plot.
    """
    # Generate a colormap for the reaches (light to dark orange)
    colors = cm.Oranges(np.linspace(0.3, 1, len(acc_peaks)))

    # Plot the correlation with colored reaches
    plt.figure(figsize=(10, 6))
    for i, (acc_peak, duration, color) in enumerate(zip(acc_peaks, reach_durations, colors)):
        plt.scatter(acc_peak, duration, color=color, label=f'Reach {i + 1}', s=50)

    plt.xlabel('Peak Acceleration (mm/s²)')
    plt.ylabel('Reach Duration (s)')
    plt.title(f'Correlation between Peak Acceleration and Reach Duration\n'
                f'Correlation: {acc_peaks_correlation:.2f}, P-value: {acc_peaks_p_value:.3f}')
    plt.grid(True)
    plt.legend(title="Reaches", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        unique_name = f"acc_peak_vs_duration_correlation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        plt.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.show()
