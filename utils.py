import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from scipy.signal import find_peaks
import math
from scipy.stats import zscore
from scipy.stats import ttest_1samp
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# # --- CSV_TO_TRAJ_DATA ---
# def CSV_To_traj_data(file_path):
#     """
#     Reads a CSV file containing trajectory data, processes it, and extracts specific columns based on predefined indices.

#     Args:
#         file_path (str): Path to the CSV file.

#     Returns:
#         dict: A dictionary containing extracted data for each prefix.
#     """
#     # Read the file, skipping the first 5 rows
#     df = pd.read_csv(
#         file_path,
#         skiprows=4,
#         sep=r"\s+|,",  # Split on whitespace or commas
#         engine="python"
#     )

#     # Extract Frame from column 1 (second column)
#     Frame = df.iloc[:, 0]

#     # Calculate the time for each frame based on a 200 Hz frame capture rate
#     time = Frame / 200  # Time in seconds

#     # Define a function to extract X, Y, Z, VX, VY, VZ, AX, AY, AZ, MX, MVX, MAX for a given prefix and column indices
#     def extract_columns(prefix, indices):
#         return {
#             f"{prefix}_X": df.iloc[:, indices[0]],
#             f"{prefix}_Y": df.iloc[:, indices[1]],
#             f"{prefix}_Z": df.iloc[:, indices[2]],
#             f"{prefix}_VX": df.iloc[:, indices[3]],
#             f"{prefix}_VY": df.iloc[:, indices[4]],
#             f"{prefix}_VZ": df.iloc[:, indices[5]],
#             f"{prefix}_AX": df.iloc[:, indices[6]],
#             f"{prefix}_AY": df.iloc[:, indices[7]],
#             f"{prefix}_AZ": df.iloc[:, indices[8]],
#             f"{prefix}_radial_pos": df.iloc[:, indices[9]], # radial position; the distance from the origin 
#             f"{prefix}_radial_vel": df.iloc[:, indices[10]], # radial velocity; How fast is the distance from the origin changing over time
#             f"{prefix}_radial_acc": df.iloc[:, indices[11]],# radial acceleration
#         }

#     # Define the column indices for each prefix
#     column_indices = {
#         "C7": [2, 3, 4, 74, 75, 76, 146, 147, 148, 218, 242, 266],
#         "T10": [5, 6, 7, 77, 78, 79, 149, 150, 151, 219, 243, 267],
#         "CLAV": [8, 9, 10, 80, 81, 82, 152, 153, 154, 220, 244, 268],
#         "STRN": [11, 12, 13, 83, 84, 85, 155, 156, 157, 221, 245, 269],
#         "LSHO": [14, 15, 16, 86, 87, 88, 158, 159, 160, 222, 246, 270],
#         "LUPA": [17, 18, 19, 89, 90, 91, 161, 162, 163, 223, 247, 271],
#         "LUPB": [20, 21, 22, 92, 93, 94, 164, 165, 166, 224, 248, 272],
#         "LUPC": [23, 24, 25, 95, 96, 97, 167, 168, 169, 225, 249, 273],
#         "LELB": [26, 27, 28, 98, 99, 100, 170, 171, 172, 226, 250, 274],
#         "LMEP": [29, 30, 31, 101, 102, 103, 173, 174, 175, 227, 251, 275],
#         "LWRA": [32, 33, 34, 104, 105, 106, 176, 177, 178, 228, 252, 276],
#         "LWRB": [35, 36, 37, 107, 108, 109, 179, 180, 181, 229, 253, 277],
#         "LFRA": [38, 39, 40, 110, 111, 112, 182, 183, 184, 230, 254, 278],
#         "LFIN": [41, 42, 43, 113, 114, 115, 185, 186, 187, 231, 255, 279],
#         "RSHO": [44, 45, 46, 116, 117, 118, 188, 189, 190, 232, 256, 280],
#         "RUPA": [47, 48, 49, 119, 120, 121, 191, 192, 193, 233, 257, 281],
#         "RUPB": [50, 51, 52, 122, 123, 124, 194, 195, 196, 234, 258, 282],
#         "RUPC": [53, 54, 55, 125, 126, 127, 197, 198, 199, 235, 259, 283],
#         "RELB": [56, 57, 58, 128, 129, 130, 200, 201, 202, 236, 260, 284],
#         "RMEP": [59, 60, 61, 131, 132, 133, 203, 204, 205, 237, 261, 285],
#         "RWRA": [62, 63, 64, 134, 135, 136, 206, 207, 208, 238, 262, 286],
#         "RWRB": [65, 66, 67, 137, 138, 139, 209, 210, 211, 239, 263, 287],
#         "RFRA": [68, 69, 70, 140, 141, 142, 212, 213, 214, 240, 264, 288],
#         "RFIN": [71, 72, 73, 143, 144, 145, 215, 216, 217, 241, 265, 289],
#     }

#     # Extract the traj_data for each prefix
#     traj_data = {}
#     for prefix, indices in column_indices.items():
#         traj_data.update(extract_columns(prefix, indices))


#     return traj_data, Frame, time

# # --- CSV_TO_TRAJ_DATA ---
# def CSV_To_traj_data(file_path, marker_name):
#     """
#     Reads a CSV file containing trajectory data, processes it, and extracts specific columns based on predefined indices.

#     Args:
#         file_path (str): Path to the CSV file.
#         marker_name (str): The marker name to extract data for.

#     Returns:
#         dict: A dictionary containing extracted data for the specified marker.
#     """
#     # Define the column indices for each prefix
#     column_indices = {
#         "C7": [2, 3, 4, 74, 75, 76, 146, 147, 148, 218, 242, 266],
#         "T10": [5, 6, 7, 77, 78, 79, 149, 150, 151, 219, 243, 267],
#         "CLAV": [8, 9, 10, 80, 81, 82, 152, 153, 154, 220, 244, 268],
#         "STRN": [11, 12, 13, 83, 84, 85, 155, 156, 157, 221, 245, 269],
#         "LSHO": [14, 15, 16, 86, 87, 88, 158, 159, 160, 222, 246, 270],
#         "LUPA": [17, 18, 19, 89, 90, 91, 161, 162, 163, 223, 247, 271],
#         "LUPB": [20, 21, 22, 92, 93, 94, 164, 165, 166, 224, 248, 272],
#         "LUPC": [23, 24, 25, 95, 96, 97, 167, 168, 169, 225, 249, 273],
#         "LELB": [26, 27, 28, 98, 99, 100, 170, 171, 172, 226, 250, 274],
#         "LMEP": [29, 30, 31, 101, 102, 103, 173, 174, 175, 227, 251, 275],
#         "LWRA": [32, 33, 34, 104, 105, 106, 176, 177, 178, 228, 252, 276],
#         "LWRB": [35, 36, 37, 107, 108, 109, 179, 180, 181, 229, 253, 277],
#         "LFRA": [38, 39, 40, 110, 111, 112, 182, 183, 184, 230, 254, 278],
#         "LFIN": [41, 42, 43, 113, 114, 115, 185, 186, 187, 231, 255, 279],
#         "RSHO": [44, 45, 46, 116, 117, 118, 188, 189, 190, 232, 256, 280],
#         "RUPA": [47, 48, 49, 119, 120, 121, 191, 192, 193, 233, 257, 281],
#         "RUPB": [50, 51, 52, 122, 123, 124, 194, 195, 196, 234, 258, 282],
#         "RUPC": [53, 54, 55, 125, 126, 127, 197, 198, 199, 235, 259, 283],
#         "RELB": [56, 57, 58, 128, 129, 130, 200, 201, 202, 236, 260, 284],
#         "RMEP": [59, 60, 61, 131, 132, 133, 203, 204, 205, 237, 261, 285],
#         "RWRA": [62, 63, 64, 134, 135, 136, 206, 207, 208, 238, 262, 286],
#         "RWRB": [65, 66, 67, 137, 138, 139, 209, 210, 211, 239, 263, 287],
#         "RFRA": [68, 69, 70, 140, 141, 142, 212, 213, 214, 240, 264, 288],
#         "RFIN": [71, 72, 73, 143, 144, 145, 215, 216, 217, 241, 265, 289],
#     }

#     if marker_name not in column_indices:
#         raise ValueError(f"Marker name '{marker_name}' not found in column indices.")

#     # Read only the required columns for the specified marker
#     indices = column_indices[marker_name]
#     cols_to_read = [0] + indices  # Include the Frame column (index 0)
#     df = pd.read_csv(
#         file_path,
#         skiprows=4,
#         usecols=cols_to_read,
#         sep=r"\s+|,",  # Split on whitespace or commas
#         engine="python",
#     )

#     # Extract Frame and calculate time
#     Frame = df.iloc[:, 0]
#     time = Frame / 200  # Time in seconds

#     # Extract the marker data
#     traj_data = {
#         f"{marker_name}_X": df.iloc[:, 1],
#         f"{marker_name}_Y": df.iloc[:, 2],
#         f"{marker_name}_Z": df.iloc[:, 3],
#         f"{marker_name}_VX": df.iloc[:, 4],
#         f"{marker_name}_VY": df.iloc[:, 5],
#         f"{marker_name}_VZ": df.iloc[:, 6],
#         f"{marker_name}_AX": df.iloc[:, 7],
#         f"{marker_name}_AY": df.iloc[:, 8],
#         f"{marker_name}_AZ": df.iloc[:, 9],
#         f"{marker_name}_radial_pos": df.iloc[:, 10],
#         f"{marker_name}_radial_vel": df.iloc[:, 11],
#         f"{marker_name}_radial_acc": df.iloc[:, 12],
#     }

#     return traj_data, Frame, time

# --- CSV_TO_TRAJ_DATA ---
def CSV_To_traj_data_filter(file_path, marker_name):
    """
    Reads a CSV file containing trajectory data, processes it, and extracts specific columns based on predefined indices.

    Args:
        file_path (str): Path to the CSV file.
        marker_name (str): The marker name to extract data for.

    Returns:
        dict: A dictionary containing extracted data for the specified marker.
    """
    # Sampling frequency and filter settings
    fs = 200       # Hz
    cutoff = 10     # Hz
    order = 4

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

    if marker_name not in column_indices:
        raise ValueError(f"Marker name '{marker_name}' not found in column indices.")

    # Read only the required columns for the specified marker
    indices = column_indices[marker_name]
    cols_to_read = [0] + indices  # Include the Frame column (index 0)
    df = pd.read_csv(
        file_path,
        skiprows=4,
        usecols=cols_to_read,
        sep=r"\s+|,",  # Split on whitespace or commas
        engine="python",
    )

    # Extract Frame and calculate time
    Frame = df.iloc[:, 0]
    time = Frame / 200  # Time in seconds

    # Extract and filter data
    traj_data = {}
    data_labels = [
        f"{marker_name}_X", f"{marker_name}_Y", f"{marker_name}_Z",
        f"{marker_name}_VX", f"{marker_name}_VY", f"{marker_name}_VZ",
        f"{marker_name}_AX", f"{marker_name}_AY", f"{marker_name}_AZ",
        f"{marker_name}_radial_pos", f"{marker_name}_radial_vel", f"{marker_name}_radial_acc"
    ]

    for i, label in enumerate(data_labels, start=1):
        raw_data = df.iloc[:, i]
        skip = 1 if any(k in label for k in ["VX", "VY", "VZ", "radial_vel"]) else \
            2 if any(k in label for k in ["AX", "AY", "AZ", "radial_acc"]) else 0
        raw_data = raw_data.iloc[skip:].reset_index(drop=True)
        
        filtered_data = pd.Series(butter_lowpass_filter(raw_data, cutoff, fs, order))
        if skip:
            filtered_data = pd.concat([pd.Series([np.nan] * skip), filtered_data]).reset_index(drop=True)
        
        traj_data[label] = filtered_data


    return traj_data, Frame, time

# --- TRAJ_SPACE_DATA --
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
        # Jerk = Acceleration.diff() * 200  # Assuming the data is sampled at 200 Hz, adjust if necessary
        # Calculate the change in acceleration over time (jerk) for each axis
        Jerk_X = traj_data[f"{marker}_AX"].diff() * 200  # Assuming the data is sampled at 200 Hz
        Jerk_Y = traj_data[f"{marker}_AY"].diff() * 200
        Jerk_Z = traj_data[f"{marker}_AZ"].diff() * 200

        # Combine the jerk components to calculate the overall jerk magnitude
        Jerk = (Jerk_X**2 + Jerk_Y**2 + Jerk_Z**2)**0.5

        # Store the results for the current marker
        Traj_Space_data[marker] = (Position, Speed, Acceleration, Jerk)

    return Traj_Space_data

# --- FIND LOCAL MINIMA AND PEAKS ---
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

# --- GET SUBFOLDERS WITH DEPTH ---
def get_subfolders_with_depth(Traj_folder, depth=3):
    """
    Get subfolders with a specific depth relative to the Traj_folder.

    Args:
        Traj_folder (str): The root folder to search within.
        depth (int): The depth of subfolders to retrieve.

    Returns:
        list: A list of subfolder paths with the specified depth.
    """
    return [
        os.path.join(*os.path.relpath(root, Traj_folder).split(os.sep)[-depth:])
        for root, _, _ in os.walk(Traj_folder)
        if len(os.path.relpath(root, Traj_folder).split(os.sep)) == depth
    ]

# --- FIND BBT FILES ---
def find_bbt_files(Traj_folder, date, file_type):
    """
    Find all files containing the specified file type ('tBBT', 'sBBT', or 'iBBT') in their name within the specified date folder and sort them by name.

    Args:
        date (str): The date in the format 'month/day' (e.g., '6/13').
        file_type (str): The file type to filter by ('tBBT', 'sBBT', 'iBBT').

    Returns:
        list: A sorted list of full file paths containing the specified file type.
    """
    target_folder = os.path.join(Traj_folder, date.replace("/", os.sep))
    
    print(f"Searching in folder: {target_folder}")
    if not os.path.exists(target_folder):
        raise FileNotFoundError(f"The folder for date {date} does not exist.")
    
    bbt_files = [os.path.join(target_folder, f) for f in os.listdir(target_folder) if file_type in f]
    
    # Filter out files that start with '._'
    bbt_files = [f for f in bbt_files if not os.path.basename(f).startswith('._')]
    
    return sorted(bbt_files, key=lambda x: os.path.basename(x))

# --- FILTER FILES BY HAND ---
def filter_files_by_hand(file_list, hand):
    if hand not in ["right", "left"]:
        raise ValueError("hand must be 'right' or 'left'")
    return [
        file for file in file_list
        if (int(os.path.splitext(os.path.basename(file))[0][-2:]) % 2 == 1 if hand == "right" else int(os.path.splitext(os.path.basename(file))[0][-2:]) % 2 == 0)
    ]

# --- PROCESS TRAJECTORY DATA ---
def process_trajectory_data(file_paths, BoxRange, prominence_threshold_speed, prominence_threshold_position, hand):

    # --- CHECK HAND ---
    if hand == "right":
        marker_name = "RFIN"
    elif hand == "left":
        marker_name = "LFIN"

    # --- CACHE TRAJECTORY DATA ---
    cached_data = {}
    for file_path in file_paths:
        traj_data = CSV_To_traj_data_filter(file_path, marker_name)[0]
        # traj_data = CSV_To_traj_data(file_path, marker_name)[0]
        # traj_data = CSV_To_traj_data(file_path)[0]
        traj_space = Traj_Space_data(traj_data)
        cached_data[file_path] = {
            "traj_data": traj_data,
            "traj_space": traj_space
        }
        
    # --- DETECT MINIMA ---
    candidate_data = {
        file_path: {
            "min_idx_speed": find_local_minima_peaks(cached_data[file_path]["traj_space"][marker_name][1], prominence_threshold_speed)[2],
            "min_idx_position": find_local_minima_peaks(cached_data[file_path]["traj_space"][marker_name][0], prominence_threshold_position)[2]
        }
        for file_path in cached_data
    }
    for file_path, data in candidate_data.items():
        traj_space = cached_data[file_path]["traj_space"]
        data.update({
            "minima_speed_values": traj_space[marker_name][1][data["min_idx_speed"]],
            "minima_position_values": traj_space[marker_name][0][data["min_idx_position"]]
        })

    # --- AGGREGATE STATS ---
    all_minima_positions = np.concatenate([data["minima_position_values"] for data in candidate_data.values()])
    all_minima_positions = np.sort(all_minima_positions)[30:-30]
    if len(all_minima_positions) == 0:
        raise ValueError("No valid minima_position values found for statistics.")
    ava_minima_position, ava_std = np.mean(all_minima_positions), np.std(all_minima_positions)


    # --- FILTER BY POSITION RANGE ---
    filtered_indices = {
        file_path: [
            i for i in data["min_idx_speed"]
            if ava_minima_position - 5 * ava_std <= cached_data[file_path]["traj_space"][marker_name][0][i] <= ava_minima_position + 13 * ava_std
        ]
        for file_path, data in candidate_data.items()
    }

    # --- FILTER BY BOX RANGE ---
    filtered_indices_final = {
        file_path: [
            idx for idx in indices
            if (BoxRange[1] > cached_data[file_path]["traj_data"][f"{marker_name}_X"][idx] > BoxRange[2]) or
               (BoxRange[0] > cached_data[file_path]["traj_data"][f"{marker_name}_X"][idx] > BoxRange[1])
        ]
        for file_path, indices in filtered_indices.items()
    }

    # --- CLASSIFY AND GROUP NON-ALTERNATING SEQUENCES ---
    def classify(x, hand):
        if hand == "right":
            if BoxRange[1] > x > BoxRange[2]: return 'start'
            if BoxRange[0] > x > BoxRange[1]: return 'end'
        elif hand == "left":
            if BoxRange[0] > x > BoxRange[1]: return 'start'
            if BoxRange[1] > x > BoxRange[2]: return 'end'
        return None

    filtered_indices_classified = {
        file_path: [
            (idx, classify(cached_data[file_path]["traj_data"][f"{marker_name}_X"][idx], hand))
            for idx in indices
            if classify(cached_data[file_path]["traj_data"][f"{marker_name}_X"][idx], hand)
        ]
        for file_path, indices in filtered_indices_final.items()
    }
    print("✅ Classified indices with start/end labels.")


    # --- FIND NON-ALTERNATING GROUPS AND SELECT BASED ON ACCELERATION ---
    final_selected_indices = {}
    for file_path, classified_indices in filtered_indices_classified.items():
        traj_space = cached_data[file_path]["traj_space"]
        acceleration = traj_space[marker_name][2]

        grouped_indices, current_group, last_label = [], [], None
        for idx, label in classified_indices:
            if label == last_label:
                current_group.append(idx)
            else:
                if current_group:
                    grouped_indices.append((last_label, current_group))
                current_group, last_label = [idx], label
        if current_group:
            grouped_indices.append((last_label, current_group))

        # Pick the index with the highest acceleration for "start" and the lowest for "end"
        selected_indices = [
            (max(group, key=lambda i: acceleration[i]) if label == "start" else min(group, key=lambda i: acceleration[i]))
            for label, group in grouped_indices
        ]
        final_selected_indices[file_path] = selected_indices


    # --- MATCH START AND END INDICES ---
    for file_path, indices in final_selected_indices.items():
        traj_data = cached_data[file_path]["traj_data"]
        starts = [idx for idx in indices if classify(traj_data[f"{marker_name}_X"][idx], hand) == "start"]
        ends = [idx for idx in indices if classify(traj_data[f"{marker_name}_X"][idx], hand) == "end"]

        matched_indices = []
        while starts and ends:
            start, end = starts.pop(0), ends.pop(0)
            if start < end:
                matched_indices.extend([start, end])
        final_selected_indices[file_path] = matched_indices

    # --- CHECK LENGTH AFTER BOX FILTER ---
    for file_path, indices in final_selected_indices.items():
        if len(indices) < 32:
            print(f"Too few indices after box filter in file {file_path}: {len(indices)}")
        if len(indices) > 32:
            raise ValueError(f"Too many indices after box filter in file {file_path}: {len(indices)}")

    print("✅ All files passed filtering with = 32 valid events.")

    return final_selected_indices, cached_data

# --- CALCULATE RANGES BY BOX TRAJ ---
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

# --- SAVE RESULTS TO A SINGLE CSV ---
def save_indices_to_single_csv(right_indices, left_indices, DataProcess_folder,tBBT_files):
    combined_file_path = os.path.join(DataProcess_folder, f"{os.path.basename(tBBT_files[0])[:-6]}_ReachStartAndEnd.csv")
    with open(combined_file_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        all_file_paths = sorted(set(right_indices.keys()) | set(left_indices.keys()))
        writer.writerow(all_file_paths)
        writer.writerow(["right" if fp in right_indices else "left" for fp in all_file_paths])
        max_len = max(max(len(right_indices.get(fp, [])), len(left_indices.get(fp, []))) for fp in all_file_paths)
        for i in range(max_len):
            writer.writerow([right_indices.get(fp, [""])[i] if fp in right_indices else 
                                left_indices.get(fp, [""])[i] if fp in left_indices else "" 
                                for fp in all_file_paths])
    print(f"✅ Saved combined indices to {combined_file_path}")

# --- PLOT RESULTS ---
def plot_results(file_paths, cached_data, final_selected_indices, BoxRange, plot_save_path, hand):
    # --- CHECK HAND ---
    if hand == "right":
        marker_name = "RFIN"
    elif hand == "left":
        marker_name = "LFIN"

    for file_path in file_paths:
        traj_data = cached_data[file_path]["traj_data"]
        traj_space = cached_data[file_path]["traj_space"]
        filtered_indices = final_selected_indices[file_path]

        plot_info = [
            (traj_data[f"{marker_name}_X"], "X"),
            (traj_space[marker_name][0], "Position"),
            (traj_space[marker_name][1], "Speed"),
            (traj_space[marker_name][2], "Acceleration")
        ]

        fig, axes = plt.subplots(len(plot_info), 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Filtered Results for {os.path.basename(file_path)}")

        for ax, (data, label) in zip(axes, plot_info):
            ax.plot(data, label=label)
            ax.set_ylabel(label)
            ax.legend()

            x_vals = np.array([traj_data[f"{marker_name}_X"][i] for i in filtered_indices])
            for idx, x in zip(filtered_indices, x_vals):
                if BoxRange[1] > x > BoxRange[2]:
                    ax.scatter(idx, data[idx], color="red", alpha=0.7)
                elif BoxRange[0] > x > BoxRange[1]:
                    ax.scatter(idx, data[idx], color="green", alpha=0.7)

        axes[-1].set_xlabel("Frame")
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(plot_save_path, f"{os.path.basename(file_path)}_filtered_plot.png"))
        plt.close()

    print("✅ Processing and plotting completed successfully.")

# --- PROCESS DATE FUNCTION ---
def process_date(date, traj_folder, Box_Traj_folder, figure_folder, DataProcess_folder, 
                 prominence_threshold_speed, prominence_threshold_position):
    print(f"Processing date: {date}")
    
    # --- FIND FILES ---
    tBBT_files = find_bbt_files(traj_folder, date, file_type="tBBT")
    # if len(tBBT_files) != 64:
    #     raise ValueError(f"Expected 64 tBBT files, but found {len(tBBT_files)}")

    box_traj_file = find_bbt_files(Box_Traj_folder, date, file_type="BOX_Cali")
    if len(box_traj_file) != 1:
        raise ValueError(f"Expected 1 Box Trajectory file, but found {len(box_traj_file)}")

    # --- BOX RANGE ---
    box_range = calculate_rangesByBoxTraj(box_traj_file)

    # --- CREATE SAVE PATHS ---
    plot_save_path = os.path.join(figure_folder, os.path.basename(tBBT_files[0])[:-6])
    os.makedirs(plot_save_path, exist_ok=True)

    # --- PROCESS TRAJECTORY DATA ---
    processed_data = {}
    for hand in ["right", "left"]:
        file_paths = filter_files_by_hand(tBBT_files, hand=hand)
        print(f"Number of files for {hand} hand: {len(file_paths)}")
        final_selected_indices, cached_data = process_trajectory_data(
            file_paths, box_range, prominence_threshold_speed, prominence_threshold_position, hand=hand
        )
        processed_data[hand] = (final_selected_indices, cached_data)

    # --- PLOT RESULTS ---    
        plot_results(file_paths, cached_data, final_selected_indices, box_range, plot_save_path, hand=hand)

    # --- Save indices to CSV ---
    save_indices_to_single_csv(
        processed_data["right"][0],
        processed_data["left"][0],
        DataProcess_folder,
        tBBT_files
    )

    return processed_data












# --- PROCESS RESULTS TO GET REACH SPEED SEGMENTS ---
def get_reach_speed_segments(results):
    reach_speed_segments = {}

    for date, hands_data in results.items():
        reach_speed_segments[date] = {}
        for hand, (indices, _) in hands_data.items():
            reach_speed_segments[date][hand] = {}
            for trial, trial_indices in indices.items():
                start_indices = [trial_indices[i] for i in range(len(trial_indices)) if i % 2 == 0]
                end_indices = [trial_indices[i] for i in range(len(trial_indices)) if i % 2 == 1]
                reach_speed_segments[date][hand][trial] = list(zip(start_indices, end_indices))

    return reach_speed_segments

# --- CALCULATE REACH METRICS ---
def calculate_reach_metrics(reach_speed_segments, results):
    reach_durations = {}
    reach_cartesian_distances = {}
    reach_path_distances = {}
    reach_v_peaks = {}
    reach_v_peak_indices = {}

    for date, hands_data in reach_speed_segments.items():
        reach_durations[date] = {}
        reach_cartesian_distances[date] = {}
        reach_path_distances[date] = {}
        reach_v_peaks[date] = {}
        reach_v_peak_indices[date] = {}

        for hand, trials_data in hands_data.items():
            reach_durations[date][hand] = {}
            reach_cartesian_distances[date][hand] = {}
            reach_path_distances[date][hand] = {}
            reach_v_peaks[date][hand] = {}
            reach_v_peak_indices[date][hand] = {}

            for trial, segments in trials_data.items():
                # --- Calculate durations for each segment ---
                durations = [(end - start) * (1 / 200) for start, end in segments]
                reach_durations[date][hand][trial] = durations

                # --- Calculate distances for each segment ---
                position = results[date][hand][1][trial]['traj_space']['RFIN'][0] if hand == "right" else results[date][hand][1][trial]['traj_space']['LFIN'][0]
                cartesian_distances = [abs(position[end] - position[start]) for start, end in segments] 
                reach_cartesian_distances[date][hand][trial] = cartesian_distances

                # --- Calculate path distances for each segment ---
                path_distances = [
                    position[start:end].diff().abs().sum()
                    for start, end in segments
                ]
                reach_path_distances[date][hand][trial] = path_distances

                # --- Calculate peak velocities for each segment ---
                speed = results[date][hand][1][trial]['traj_space']['RFIN'][1] if hand == "right" else results[date][hand][1][trial]['traj_space']['LFIN'][1]
                v_peak = [speed[start:end].max() for start, end in segments]
                reach_v_peaks[date][hand][trial] = v_peak
                v_peak_indices = [
                    speed[start:end].idxmax() for start, end in segments
                ]
                reach_v_peak_indices[date][hand][trial] = v_peak_indices

    reach_metrics = {
        "reach_durations": reach_durations,
        "reach_cartesian_distances": reach_cartesian_distances,
        "reach_path_distances": reach_path_distances,
        "reach_v_peaks": reach_v_peaks,
        "reach_v_peak_indices": reach_v_peak_indices,
    }

    return reach_metrics

# --- DEFINE TIME WINDOWS ---
def define_time_windows(time_window_method, reach_speed_segments, reach_metrics, frame_rate, window_size):
    if time_window_method == 1:
        return reach_speed_segments
    elif time_window_method == 2:
        return {
            date: {
                hand: {
                    trial: [
                        (segment[0], reach_metrics['reach_v_peak_indices'][date][hand][trial][i]) 
                        for i, segment in enumerate(reach_speed_segments[date][hand][trial])
                    ]
                    for trial in reach_speed_segments[date][hand]
                }
                for hand in reach_speed_segments[date]
            }
            for date in reach_speed_segments
        }
    elif time_window_method == 3:
        window_frames = int(frame_rate * window_size)
        return {
            date: {
                hand: {
                    trial: [
                        (reach_metrics['reach_v_peak_indices'][date][hand][trial][i] - window_frames,
                         reach_metrics['reach_v_peak_indices'][date][hand][trial][i] + window_frames)
                        for i in range(len(reach_metrics['reach_v_peak_indices'][date][hand][trial]))
                    ]
                    for trial in reach_speed_segments[date][hand]
                }
                for hand in reach_speed_segments[date]
            }
            for date in reach_speed_segments
        }
    elif time_window_method == 4:
        window_frames = int(frame_rate * 0.1)  # 100 ms before the peak
        return {
            date: {
                hand: {
                    trial: [
                        (reach_metrics['reach_v_peak_indices'][date][hand][trial][i] - window_frames,
                         reach_metrics['reach_v_peak_indices'][date][hand][trial][i])
                        for i in range(len(reach_metrics['reach_v_peak_indices'][date][hand][trial]))
                    ]
                    for trial in reach_speed_segments[date][hand]
                }
                for hand in reach_speed_segments[date]
            }
            for date in reach_speed_segments
        }
    elif time_window_method == 5:
        window_frames = int(frame_rate * 0.1)  # 100 ms after the peak
        return {
            date: {
                hand: {
                    trial: [
                        (reach_metrics['reach_v_peak_indices'][date][hand][trial][i],
                            reach_metrics['reach_v_peak_indices'][date][hand][trial][i]+ window_frames)
                        for i in range(len(reach_metrics['reach_v_peak_indices'][date][hand][trial]))
                    ]
                    for trial in reach_speed_segments[date][hand]
                }
                for hand in reach_speed_segments[date]
            }
            for date in reach_speed_segments
        }
    raise ValueError("Invalid time_window_method. Choose 1, 2, 3, 4, 5.")

# --- CALCULATE REACH METRICS SPECIFIC TO TIME WINDOW ---
def calculate_reach_metrics_for_time_windows(test_windows, results):
    reach_acc_peaks = {}
    reach_jerk_peaks = {}
    reach_LDLJ = {}

    for date, hands_data in test_windows.items():
        reach_acc_peaks[date] = {}
        reach_jerk_peaks[date] = {}
        reach_LDLJ[date] = {}

        for hand, trials_data in hands_data.items():
            reach_acc_peaks[date][hand] = {}
            reach_jerk_peaks[date][hand] = {}
            reach_LDLJ[date][hand] = {}

            for trial, segments in trials_data.items():
                marker = "RFIN" if hand == "right" else "LFIN"
                position, speed, acceleration, jerk = results[date][hand][1][trial]['traj_space'][marker]

                # Find the peak acceleration
                acc_peak = [acceleration[start:end].max() for start, end in segments]
                reach_acc_peaks[date][hand][trial] = acc_peak

                # Find the peak jerk
                jerk_peak = [jerk[start:end].max() for start, end in segments]
                reach_jerk_peaks[date][hand][trial] = jerk_peak

                LDLJ = []
                for start, end in segments:
                    jerk_segment = jerk[start:end] # Get the jerk segment for the current time window
                    duration = (end - start) / 200 # Calculate the duration of the segment in seconds
                    t = np.linspace(0, duration, len(jerk_segment)) # Create a time vector for the segment
                    jerk_squared_integral = np.trapezoid(jerk_segment**2, t) # Calculate the integral of the squared jerk
                    vpeak = speed[start:end].max() # Get the peak speed for the segment
                    dimensionless_jerk = (duration**3 / vpeak**2) * jerk_squared_integral
                    LDLJ.append(-math.log(abs(dimensionless_jerk), math.e))

                reach_LDLJ[date][hand][trial] = LDLJ

    reach_TW_metrics = {
        "reach_acc_peaks": reach_acc_peaks,
        "reach_jerk_peaks": reach_jerk_peaks,
        "reach_LDLJ": reach_LDLJ
    }
    return reach_TW_metrics

# --- SAVE LDLJ VALUES ---
def save_ldlj_values(reach_TW_metrics, DataProcess_folder):
    """
    Save all LDLJ values by subject, hand, and trial to a CSV file.

    Parameters:
        reach_TW_metrics (dict): Metrics for time windows.
        DataProcess_folder (str): Folder path to save the CSV file.
    """
    ldlj_table = []

    for date in reach_TW_metrics['reach_LDLJ']:
        for hand in ['right', 'left']:
            for trial in reach_TW_metrics['reach_LDLJ'][date][hand]:
                ldlj_values = reach_TW_metrics['reach_LDLJ'][date][hand][trial]
                ldlj_table.append({
                    "Date": date,
                    "Hand": hand,
                    "Trial": trial,
                    **{f"Reach {i + 1}": ldlj_value for i, ldlj_value in enumerate(ldlj_values)}
                })

    ldlj_df = pd.DataFrame(ldlj_table)

    # Save as CSV file
    csv_save_path = os.path.join(DataProcess_folder, "ldlj_values.csv")
    ldlj_df.to_csv(csv_save_path, index=False)
    print(f"LDLJ values saved to {csv_save_path}")

# --- SPARC FUNCTION ---
def sparc(movement, fs, padlevel=4, fc=10.0, amp_th=0.05):
    """
    Calcualtes the smoothness of the given speed profile using the modified
    spectral arc length metric.

    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    padlevel : integer, optional
               Indicates the amount of zero padding to be done to the movement
               data for estimating the spectral arc length. [default = 4]
    fc       : float, optional
               The max. cut off frequency for calculating the spectral arc
               length metric. [default = 10.]
    amp_th   : float, optional
               The amplitude threshold to used for determing the cut off
               frequency upto which the spectral arc length is to be estimated.
               [default = 0.05]

    Returns
    -------
    sal      : float
               The spectral arc length estimate of the given movement's
               smoothness.
    (f, Mf)  : tuple of two np.arrays
               This is the frequency(f) and the magntiude spectrum(Mf) of the
               given movement data. This spectral is from 0. to fs/2.
    (f_sel, Mf_sel) : tuple of two np.arrays
                      This is the portion of the spectrum that is selected for
                      calculating the spectral arc length.

    Notes
    -----
    This is the modfieid spectral arc length metric, which has been tested only
    for discrete movements.

    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> sal, _, _ = sparc(move, fs=100.)
    >>> '%.5f' % sal
    '-1.41403'

    """
    # Number of zeros to be padded.
    nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

    # Frequency
    f = np.arange(0, fs, fs / nfft)
    # Normalized magnitude spectrum
    Mf = abs(np.fft.fft(movement, nfft))
    Mf = Mf / max(Mf)

    # Indices to choose only the spectrum within the given cut off frequency
    # Fc.
    # NOTE: This is a low pass filtering operation to get rid of high frequency
    # noise from affecting the next step (amplitude threshold based cut off for
    # arc length calculation).
    fc_inx = ((f <= fc) * 1).nonzero()
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]

    # Choose the amplitude threshold based cut off frequency.
    # Index of the last point on the magnitude spectrum that is greater than
    # or equal to the amplitude threshold.
    inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
    fc_inx = range(inx[0], inx[-1] + 1)
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]

    # Calculate arc length
    new_sal = -sum(np.sqrt(pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) +
                           pow(np.diff(Mf_sel), 2)))
    return new_sal, (f, Mf), (f_sel, Mf_sel)

# --- SPARC WITH PLOTS ---
def sparc_with_plots(movement, fs=200, padlevel=4, fc=10.0, amp_th=0.05):
    # 1. Time domain plot (original movement)
    t = np.arange(len(movement)) / fs
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 2, 1)
    plt.plot(t, movement)
    plt.title("1. Speed Profile (Time Domain)")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed")

    # 2. FFT and Normalization
    nfft = int(2 ** (np.ceil(np.log2(len(movement))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    Mf = np.abs(np.fft.fft(movement, nfft))
    Mf = Mf / np.max(Mf)

    plt.subplot(3, 2, 2)
    plt.plot(f[:nfft // 2], Mf[:nfft // 2])
    plt.title("2. Full Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Magnitude")

    # 3. Cutoff Frequency Filtering
    fc_idx = np.where(f <= fc)[0]
    f_sel = f[fc_idx]
    Mf_sel = Mf[fc_idx]

    plt.subplot(3, 2, 3)
    plt.plot(f_sel, Mf_sel)
    plt.title("3. Spectrum Below Cutoff (fc = 10 Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    # 4. Amplitude Thresholding with Continuous Range
    inx = np.where(Mf_sel >= amp_th)[0]
    fc_inx = np.arange(inx[0], inx[-1] + 1)
    f_cut = f_sel[fc_inx]
    Mf_cut = Mf_sel[fc_inx]

    plt.subplot(3, 2, 4)
    plt.plot(f_sel, Mf_sel, color='lightgray', label='All under 10Hz')
    plt.plot(f_cut, Mf_cut, color='blue', label='Above threshold')
    plt.axhline(y=amp_th, color='red', linestyle='--', label='Threshold')
    plt.title("4. After Amplitude Thresholding")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()

    # 5. Spectral Arc Length Calculation (matching sparc())
    df = np.diff(f_cut) / (f_cut[-1] - f_cut[0])
    dM = np.diff(Mf_cut)
    arc_length = -np.sum(np.sqrt(df ** 2 + dM ** 2))

    plt.subplot(3, 2, 5)
    plt.plot(f_cut, Mf_cut, marker='o')
    for i in range(len(df)):
        plt.plot([f_cut[i], f_cut[i+1]], [Mf_cut[i], Mf_cut[i+1]], 'k--')
    plt.title("5. Arc Segments Used in SPARC")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    # 6. Display SPARC Value
    plt.subplot(3, 2, 6)
    plt.text(0.1, 0.5, f"SPARC Value:\n{arc_length:.4f}", fontsize=20)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return arc_length

# --- CALCULATE REACH SPARC ---
def calculate_reach_sparc(test_windows, results):
    """
    Calculate SPARC for each test window for all dates, hands, and trials.

    Args:
        test_windows (dict): Dictionary containing test windows for each date, hand, and trial.
        results (dict): Dictionary containing processed results for each date, hand, and trial.

    Returns:
        dict: SPARC values for each date, hand, and trial.
    """
    reach_sparc = {}

    for date, hands_data in test_windows.items():
        reach_sparc[date] = {}

        for hand, trials_data in hands_data.items():
            reach_sparc[date][hand] = {}

            for trial, segments in trials_data.items():
                marker_name = "RFIN" if hand == "right" else "LFIN"
                speed_data = results[date][hand][1][trial]["traj_space"][marker_name][1]

                sparc_values = []
                for start_frame, end_frame in segments:
                    window_speed = speed_data[start_frame:end_frame]
                    sal, _, _ = sparc(window_speed, fs=200.)
                    sparc_values.append(sal)

                reach_sparc[date][hand][trial] = sparc_values

    return reach_sparc

# --- SAVE SPARC VALUES ---
def save_sparc_values(reach_sparc, DataProcess_folder):
    """
    Save all SPARC values by subject, hand, and trial into a CSV file.

    Parameters:
        reach_sparc (dict): Dictionary containing SPARC values for each date, hand, and trial.
        DataProcess_folder (str): Path to the folder where the CSV file will be saved.
    """
    sparc_table = []

    for date in reach_sparc:
        for hand in reach_sparc[date]:
            for trial in reach_sparc[date][hand]:
                sparc_values = reach_sparc[date][hand][trial]
                sparc_table.append({
                    "Date": date,
                    "Hand": hand,
                    "Trial": trial,
                    **{f"Window {i + 1}": sparc_value for i, sparc_value in enumerate(sparc_values)}
                })

    sparc_df = pd.DataFrame(sparc_table)

    # Save as CSV file
    csv_save_path = os.path.join(DataProcess_folder, "sparc_values.csv")
    sparc_df.to_csv(csv_save_path, index=False)
    print(f"SPARC values saved to {csv_save_path}")


# --- PLOT EACH SEGMENT SPEED AS SUBPLOT WITH LDLJ AND SPARC VALUES ---
def plot_segments_with_ldlj_and_sparc(date, hand, trial, test_windows, results, reach_TW_metrics, reach_sparc, save_path):
    """
    Plot speed segments for a specific trial with LDLJ and SPARC values as titles.

    Parameters:
        date (str): The date of the trial.
        hand (str): The hand ('right' or 'left').
        trial (str): The trial file path.
        test_windows (dict): The time window segments.
        results (dict): The processed results data.
        reach_TW_metrics (dict): The time window metrics containing LDLJ values.
        reach_sparc (dict): The SPARC values for each segment.
        save_path (str): The directory to save the plots.
    """
    # Get the data for the selected trial
    segments = test_windows[date][hand][trial]
    marker = "RFIN" if hand == "right" else "LFIN"
    speed = results[date][hand][1][trial]['traj_space'][marker][1]
    ldlj_values = reach_TW_metrics['reach_LDLJ'][date][hand][trial]
    sparc_values = reach_sparc[date][hand][trial]

    # Plot the speed for each segment
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    fig.suptitle(f"Date: {date}, Hand: {hand}, Trial: {os.path.basename(trial)}", fontsize=20, fontweight='bold')
    axes = axes.flatten()

    for i, (start, end) in enumerate(segments):
        if i >= 16:  # Limit to 16 subplots
            break
        ax = axes[i]
        ax.plot(speed[start:end], color='blue', label='Speed (m/s)')
        ax.set_title(
            f"LDLJ: {ldlj_values[i]:.2f}, SPARC: {sparc_values[i]:.2f}" if ldlj_values[i] is not None and sparc_values[i] is not None else "LDLJ/SPARC: None"
        )
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Speed (m/s)")
        ax.set_ylim(0, 3000)  # Set y-axis range from 0 to 3000
        ax.grid(True)

    # Hide unused subplots
    for j in range(i + 1, 16):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{hand}_{os.path.basename(trial)}_speed_segments.png")
    plt.savefig(save_file)
    plt.close(fig)

# # --- CALCULATE REACH METRICS SPECIFIC TO TIME WINDOW ---
# def calculate_LDLJ_for_time_windows(test_windows, results):
#     reach_LDLJ_X = {}
#     reach_LDLJ_V_peak = {}

#     for date, hands_data in test_windows.items():

#         reach_LDLJ_X[date] = {}
#         reach_LDLJ_V_peak[date] = {}

#         for hand, trials_data in hands_data.items():

#             reach_LDLJ_X[date][hand] = {}
#             reach_LDLJ_V_peak[date][hand] = {}

#             for trial, segments in trials_data.items():
#                 marker = "RFIN" if hand == "right" else "LFIN"

#                 position, speed, _, _ = results[date][hand][1][trial]['traj_space'][marker]

#                 traj_data = results[date][hand][1][trial]['traj_data']

#                 LDLJ_X = []
#                 LDLJ_V_peak = []

#                 for start, end in segments:
#                     position_segment = position[start:end] # Get the position segment for the current time window
#                     speed_segment = speed[start:end] # Get the speed segment for the current time window

#                     duration = (end - start) / 200 # Calculate the duration of the segment in seconds

#                     position_t = np.linspace(0, duration, len(position_segment)) # Create a time vector for the segment
#                     # speed_t = np.linspace(0, duration, len(speed_segment)) # Create a time vector for the segment

#                     # position based
#                     d3x_dt3 = np.gradient(np.gradient(np.gradient(position_segment, position_t), position_t), position_t)
#                     jerk_squared_integral_p = np.trapezoid(d3x_dt3**2, position_t)
#                     # movement_amplitude = abs(position_segment.iloc[-1] - position_segment.iloc[0])
#                     movement_amplitude = position_segment.diff().abs().sum()
#                     dimensionless_jerk_p = ((duration**5 / movement_amplitude**2) * jerk_squared_integral_p)
#                     LDLJ_X.append ( -math.log(dimensionless_jerk_p, math.e))

#                     # # speed based
#                     # d2v_dt2 = np.gradient(np.gradient(speed_segment, speed_t), speed_t)
#                     # jerk_squared_integral_v = np.trapezoid(d2v_dt2**2, speed_t) # Calculate the integral of the squared jerk
#                     # vpeak = speed_segment.max() # Get the peak speed for the segment                 
#                     # dimensionless_jerk_v = (duration**3 / vpeak**2) * jerk_squared_integral_v
#                     # LDLJ_V_peak.append(-math.log(dimensionless_jerk_v, math.e))


#                     # # Alternative speed based calculation
#                     # duration = (end - start) / 200 # Calculate the duration of the segment in seconds

#                     # position_X = traj_data[f"{marker}_X"][start:end]
#                     # position_Y = traj_data[f"{marker}_Y"][start:end]
#                     # position_Z = traj_data[f"{marker}_Z"][start:end]

#                     # # Calculate jerk from position (third derivative) Use gradient
#                     # Jerk_X = np.gradient(np.gradient(np.gradient(position_X, edge_order=2), edge_order=2), edge_order=2) * (200 ** 3)
#                     # Jerk_Y = np.gradient(np.gradient(np.gradient(position_Y, edge_order=2), edge_order=2), edge_order=2) * (200 ** 3)
#                     # Jerk_Z = np.gradient(np.gradient(np.gradient(position_Z, edge_order=2), edge_order=2), edge_order=2) * (200 ** 3)

#                     # jerk = (Jerk_X**2 + Jerk_Y**2 + Jerk_Z**2)**0.5


#                     # position_t = np.linspace(0, duration, len(jerk))  # Create a time vector for the segment

#                     # # position based
#                     # jerk_squared_integral_p = np.trapezoid(jerk**2, position_t)

#                     # movement_amplitude = position_segment.diff().abs().sum()

#                     # dimensionless_jerk_p = ((duration**5 / movement_amplitude**2) * jerk_squared_integral_p)
#                     # LDLJ_X.append ( -math.log(dimensionless_jerk_p, math.e))

#                     # speed based
#                     speed_X = traj_data[f"{marker}_VX"][start:end]
#                     speed_Y = traj_data[f"{marker}_VY"][start:end]
#                     speed_Z = traj_data[f"{marker}_VZ"][start:end]

#                     # Calculate jerk from speed (second derivative)
#                     Jerk_X = np.gradient(np.gradient(speed_X, edge_order=2), edge_order=2) * (200 ** 2)
#                     Jerk_Y = np.gradient(np.gradient(speed_Y, edge_order=2), edge_order=2) * (200 ** 2)
#                     Jerk_Z = np.gradient(np.gradient(speed_Z, edge_order=2), edge_order=2) * (200 ** 2)

#                     jerk = (Jerk_X**2 + Jerk_Y**2 + Jerk_Z**2)**0.5

#                     speed_t = np.linspace(0, duration, len(jerk))  # Create a time vector for the segment
                    
#                     jerk_squared_integral_v = np.trapezoid(jerk**2, speed_t) # Calculate the integral of the squared jerk
#                     vpeak = speed_segment.max() # Get the peak speed for the segment                 
#                     dimensionless_jerk_v = (duration**3 / vpeak**2) * jerk_squared_integral_v
#                     LDLJ_V_peak.append(-math.log(dimensionless_jerk_v, math.e))

#                 reach_LDLJ_X[date][hand][trial] = LDLJ_X
#                 reach_LDLJ_V_peak[date][hand][trial] = LDLJ_V_peak

#     All_LDLJ = {
#         "reach_LDLJ_X": reach_LDLJ_X,
#         "reach_LDLJ_V_peak": reach_LDLJ_V_peak,

#     }
#     return All_LDLJ

# --- PLOT JERK SEGMENTS ---
def plot_jerk_segments(date, hand, trial, test_windows, results, reach_TW_metrics, save_path):
    """
    Plot jerk segments for a specific trial with LDLJ values as titles, overlaying speed data.

    Parameters:
        date (str): The date of the trial.
        hand (str): The hand ('right' or 'left').
        trial (str): The trial file path.
        test_windows (dict): The time window segments.
        results (dict): The processed results data.
        reach_TW_metrics (dict): The time window metrics containing LDLJ values.
    """
    # Get the data for the selected trial
    segments = test_windows[date][hand][trial]
    jerk = results[date][hand][1][trial]['traj_space']['RFIN'][3] if hand == "right" else results[date][hand][1][trial]['traj_space']['LFIN'][3]
    speed = results[date][hand][1][trial]['traj_space']['RFIN'][1] if hand == "right" else results[date][hand][1][trial]['traj_space']['LFIN'][1]
    ldlj_values = reach_TW_metrics['reach_LDLJ'][date][hand][trial]

    # Calculate global min and max values for jerk and speed across all segments
    min_jerk = min(jerk[start:end].min() for start, end in segments)
    max_jerk = max(jerk[start:end].max() for start, end in segments)
    min_speed = min(speed[start:end].min() for start, end in segments)
    max_speed = max(speed[start:end].max() for start, end in segments)

    # Plot the jerk and overlay speed for each segment
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    fig.suptitle(f"Date: {date}, Hand: {hand}, Trial: {os.path.basename(trial)}", fontsize=20, fontweight='bold')  # Add date and hand as big title
    axes = axes.flatten()

    for i, (start, end) in enumerate(segments):
        if i >= 16:  # Limit to 16 subplots
            break
        ax1 = axes[i]
        ax1.plot(jerk[start:end], color='blue', label='Jerk (m/s³)')
        ax1.set_title(f"LDLJ: {ldlj_values[i]:.2f}" if ldlj_values[i] is not None else "LDLJ: None")
        ax1.set_xlabel("Time (samples)")
        ax1.set_ylabel("Jerk (m/s³)", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_yticks(np.linspace(min_jerk, max_jerk, 5))
        ax1.set_ylim(min_jerk - 1000, max_jerk + 1000)  # Set consistent y-axis limits for jerk

        # Create a secondary y-axis for speed
        ax2 = ax1.twinx()
        ax2.plot(speed[start:end], color='red', label='Speed (m/s)', alpha=0.7)
        ax2.set_ylabel("Speed (m/s)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_yticks(np.linspace(min_speed, max_speed, 5))
        ax2.set_ylim(min_speed - 50, max_speed + 50)  # Set consistent y-axis limits for speed

    # Hide unused subplots
    for j in range(i + 1, 16):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{hand}_{os.path.basename(trial)}_jerk_segments.png")
    plt.savefig(save_file)
    plt.close(fig)


# --- PLOT SEGMENTS ---
def plot_segments(date, hand, trial, test_windows, results, reach_TW_metrics, save_path):
    """
    Plot jerk segments for a specific trial with LDLJ values as titles, overlaying speed, position, and acceleration data.

    Parameters:
        date (str): The date of the trial.
        hand (str): The hand ('right' or 'left').
        trial (str): The trial file path.
        test_windows (dict): The time window segments.
        results (dict): The processed results data.
        reach_TW_metrics (dict): The time window metrics containing LDLJ values.
    """
    # Get the data for the selected trial
    segments = test_windows[date][hand][trial]
    marker = "RFIN" if hand == "right" else "LFIN"
    jerk = results[date][hand][1][trial]['traj_space'][marker][3]
    speed = results[date][hand][1][trial]['traj_space'][marker][1]
    position = results[date][hand][1][trial]['traj_space'][marker][0]
    acceleration = results[date][hand][1][trial]['traj_space'][marker][2]
    ldlj_values = reach_TW_metrics['reach_LDLJ'][date][hand][trial]

    # Plot the jerk and overlay speed, position, and acceleration for each segment
    fig, axes = plt.subplots(4, 4, figsize=(30, 15))
    fig.suptitle(f"Date: {date}, Hand: {hand}, Trial: {os.path.basename(trial)}", fontsize=20, fontweight='bold')  # Add date and hand as big title
    axes = axes.flatten()

    for i, (start, end) in enumerate(segments):
        if i >= 16:  # Limit to 16 subplots
            break
        ax1 = axes[i]
        ax1.plot(jerk[start:end], color='blue', label='Jerk (m/s³)')
        ax1.set_title(f"LDLJ: {ldlj_values[i]:.2f}" if ldlj_values[i] is not None else "LDLJ: None")
        ax1.set_xlabel("Time (samples)")
        ax1.set_ylabel("Jerk (m/s³)", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Create secondary y-axes for speed, position, and acceleration
        ax2 = ax1.twinx()
        ax2.plot(speed[start:end], color='red', label='Speed (m/s)', alpha=0.7)
        ax2.set_ylabel("Speed (m/s)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))  # Offset the third axis
        ax3.plot(position[start:end], color='green', label='Position (m)', alpha=0.7)
        ax3.set_ylabel("Position (m)", color='green')
        ax3.tick_params(axis='y', labelcolor='green')

        ax4 = ax1.twinx()
        ax4.spines["right"].set_position(("outward", 120))  # Offset the fourth axis
        ax4.plot(acceleration[start:end], color='purple', label='Acceleration (m/s²)', alpha=0.7)
        ax4.set_ylabel("Acceleration (m/s²)", color='purple')
        ax4.tick_params(axis='y', labelcolor='purple')

    # Hide unused subplots
    for j in range(i + 1, 16):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{hand}_{os.path.basename(trial)}_jerk_segments.png")
    plt.savefig(save_file)
    plt.close(fig)

# --- PLOT REACH LDLJ VALUES OVER TRIALS ---
def plot_reach_ldlj_over_trials(reach_TW_metrics, date, hand, save_path):
    """
    Save scatter plots for LDLJ values across trials for each reach as subplots.

    Parameters:
        reach_TW_metrics (dict): The reach time window metrics containing LDLJ values.
        date (str): The date of the trials.
        hand (str): The hand ('right' or 'left').
        save_path (str): The directory to save the plots.
    """
    # Determine global y-axis limits across all subplots
    all_ldlj_values = [
        ldlj
        for trial in reach_TW_metrics['reach_LDLJ'][date][hand]
        for reach_index, ldlj in enumerate(reach_TW_metrics['reach_LDLJ'][date][hand][trial])
    ]
    y_min, y_max = min(all_ldlj_values), max(all_ldlj_values)

    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    fig.suptitle(f"Date: {date}, Hand: {hand}", fontsize=20, fontweight='bold')  # Add date and hand as big title
    axes = axes.flatten()

    for reach_index in range(16):
        ldlj_values = [
            reach_TW_metrics['reach_LDLJ'][date][hand][trial][reach_index]
            for trial in reach_TW_metrics['reach_LDLJ'][date][hand]
            if reach_index < len(reach_TW_metrics['reach_LDLJ'][date][hand][trial])
        ]

        ax = axes[reach_index]
        ax.scatter(range(len(ldlj_values)), ldlj_values, color='blue', label='LDLJ Values')
        ax.set_xlabel('Trial Index')
        ax.set_ylabel('LDLJ Value')
        ax.set_title(f'Reach {reach_index + 1}')
        ax.legend()
        ax.grid(True)

        # Set consistent y-axis limits
        ax.set_ylim(y_min, y_max)

    # Hide unused subplots if any
    for i in range(reach_index + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{hand}_reach_ldlj_over_trials.png")
    plt.savefig(save_file)
    plt.close(fig)

# --- PLOT REACH LDLJ VALUES AGAINST REACH DURATIONS ---
def plot_reach_ldlj_vs_duration(reach_TW_metrics, reach_metrics, date, hand, save_path):
    """
    Generates scatter plots of LDLJ values against reach durations for multiple reaches 
    and saves the plots as an image file.

    Parameters:
        reach_TW_metrics (dict): A dictionary containing LDLJ metrics data. 
                                 Expected structure:
                                 reach_TW_metrics['reach_LDLJ'][date][hand][trial][reach_index].
        reach_metrics (dict): A dictionary containing reach duration metrics data. 
                              Expected structure:
                              reach_metrics['reach_durations'][date][hand][trial][reach_index].
        date (str): The date key used to access the metrics data.
        hand (str): The hand key ('left' or 'right') used to access the metrics data.
        save_path (str): The directory path where the scatter plot image will be saved.

    Returns:
        None: The function saves the scatter plot image to the specified path and does not return anything.

    Notes:
        - The function creates a 4x4 grid of subplots, one for each reach (up to 16 reaches).
        - If there are fewer than 16 reaches, unused subplots are hidden.
        - Each subplot displays a scatter plot of reach durations (x-axis) against LDLJ values (y-axis).
        - The image is saved as "{hand}_LDLJ_scatter_plots.png" in the specified save_path directory.
    """
    # Determine global x and y axis limits across all subplots
    all_ldlj_values = [
        ldlj
        for trial in reach_TW_metrics['reach_LDLJ'][date][hand]
        for reach_index, ldlj in enumerate(reach_TW_metrics['reach_LDLJ'][date][hand][trial])
    ]
    all_reach_durations = [
        duration
        for trial in reach_metrics['reach_durations'][date][hand]
        for reach_index, duration in enumerate(reach_metrics['reach_durations'][date][hand][trial])
    ]
    x_min, x_max = min(all_reach_durations), max(all_reach_durations)
    y_min, y_max = min(all_ldlj_values), max(all_ldlj_values)

    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    fig.suptitle(f"Date: {date}, Hand: {hand}", fontsize=20, fontweight='bold')# Add date as big title
    axes = axes.flatten()

    for reach_index in range(16):
        ldlj_values = [
            reach_TW_metrics['reach_LDLJ'][date][hand][trial][reach_index]
            for trial in reach_TW_metrics['reach_LDLJ'][date][hand]
            if reach_index < len(reach_TW_metrics['reach_LDLJ'][date][hand][trial])
        ]

        reach_durations = [
            reach_metrics['reach_durations'][date][hand][trial][reach_index]
            for trial in reach_metrics['reach_durations'][date][hand]
            if reach_index < len(reach_metrics['reach_durations'][date][hand][trial])
        ]
        ax = axes[reach_index]
        ax.scatter(reach_durations, ldlj_values, color='blue', label='LDLJ Values')
        ax.set_xlabel('Reach Duration (s)')
        ax.set_ylabel('LDLJ Value')
        ax.set_title(f'Reach {reach_index + 1}')
        ax.grid(True)

        # Set consistent axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Hide unused subplots if any
    for i in range(reach_index + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"{hand}_reach_ldlj_vs_duration.png")
    plt.savefig(save_file)
    plt.close(fig)

# --- CALCULATE REACH LDLJ VALUES AGAINST REACH DURATIONS AND CORRELATIONS ---
def calculate_ldlj_vs_duration_correlation(reach_TW_metrics, reach_metrics, date, hand):
    """
    Calculates the correlation between z-scores of LDLJ values and reach durations for multiple reaches.

    Parameters:
        reach_TW_metrics (dict): A dictionary containing LDLJ metrics data. 
                                 Expected structure:
                                 reach_TW_metrics['reach_LDLJ'][date][hand][trial][reach_index].
        reach_metrics (dict): A dictionary containing reach duration metrics data. 
                              Expected structure:
                              reach_metrics['reach_durations'][date][hand][trial][reach_index].
        date (str): The date key used to access the metrics data.
        hand (str): The hand key ('left' or 'right') used to access the metrics data.

    Returns:
        dict: A dictionary containing correlations for each reach index.

    Notes:
        - The function calculates z-scores for LDLJ values and reach durations.
        - Correlation is calculated for each reach index across trials.
        - If there are fewer than 2 data points for a reach index, correlation is set to None.
    """
    correlations = {}

    for reach_index in range(16):
        ldlj_values = [
            reach_TW_metrics['reach_LDLJ'][date][hand][trial][reach_index]
            for trial in reach_TW_metrics['reach_LDLJ'][date][hand]
            if reach_index < len(reach_TW_metrics['reach_LDLJ'][date][hand][trial])
        ]

        reach_durations = [
            reach_metrics['reach_durations'][date][hand][trial][reach_index]
            for trial in reach_metrics['reach_durations'][date][hand]
            if reach_index < len(reach_metrics['reach_durations'][date][hand][trial])
        ]

        if len(ldlj_values) > 1 and len(reach_durations) > 1:
            # Calculate z-scores
            ldlj_z = zscore(ldlj_values)
            durations_z = zscore(reach_durations)

            # Calculate correlation
            correlation = np.corrcoef(ldlj_z, durations_z)[0, 1]
            correlations[reach_index] = correlation
        else:
            correlations[reach_index] = None

    return correlations

# --- PLOT REACH LDLJ DURATION CORRELATIONS AS HISTOGRAM ---
def plot_ldlj_duration_correlations_histogram_by_hand(reach_ldlj_duration_correlations, save_path):
    """
    Plots histograms of LDLJ duration correlations for each date and hand in one figure as subplots.

    Parameters:
        reach_ldlj_duration_correlations (dict): A dictionary containing LDLJ duration correlations.
        save_path (str): The directory to save the histogram plots.

    Returns:
        None: The function saves the histogram plots to the specified path.
    """
    fig, axes = plt.subplots(len(reach_ldlj_duration_correlations), 2, figsize=(15, 5 * len(reach_ldlj_duration_correlations)))
    fig.suptitle("LDLJ Duration Correlations Histograms", fontsize=20, fontweight='bold')

    for i, date in enumerate(reach_ldlj_duration_correlations):
        for j, hand in enumerate(['right', 'left']):
            correlations = [
                corr for corr in reach_ldlj_duration_correlations[date][hand].values() if corr is not None
            ]

            ax = axes[i][j] if len(reach_ldlj_duration_correlations) > 1 else axes[j]
            ax.hist(correlations, bins=10, color='blue', alpha=0.7)
            ax.set_title(f"Date: {date}, Hand: {hand}", fontsize=14)
            ax.set_xlabel("Correlation Coefficient")
            ax.set_ylabel("Frequency")
            ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, "ldlj_duration_correlations_histograms.png")
    plt.savefig(save_file)
    plt.close()

# --- PLOT REACH LDLJ DURATION CORRELATIONS AS OVERLAPPED HISTOGRAM ---
def plot_ldlj_duration_correlations_histogram(reach_ldlj_duration_correlations, save_path):
    """
    Plots a single histogram with overlapped data for LDLJ duration correlations for each date, using different colors.
    Also calculates if the correlations are significantly shifted from 0 and labels the p-value.

    Parameters:
        reach_ldlj_duration_correlations (dict): A dictionary containing LDLJ duration correlations.
        save_path (str): The directory to save the histogram plot.

    Returns:
        None: The function saves the histogram plot to the specified path.
    """
    plt.figure(figsize=(15, 10))
    plt.title("LDLJ Duration Correlations Histogram", fontsize=20, fontweight='bold')
    colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

    for i, date in enumerate(reach_ldlj_duration_correlations):
        correlations = [
            corr for hand in reach_ldlj_duration_correlations[date]
            for corr in reach_ldlj_duration_correlations[date][hand].values() if corr is not None
        ]
        
        # Perform a one-sample t-test to check if correlations are significantly different from 0
        t_stat, p_value = ttest_1samp(correlations, 0)
        
        # Plot histogram
        plt.hist(correlations, bins=10, color=colors[i % len(colors)], alpha=0.7, 
                 label=f"Date: {date} (n={len(correlations)}, p={p_value:.3f})")

    # Add a vertical line at x = 0
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1, label="x = 0")

    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right")
    plt.grid(True)

    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, "ldlj_duration_correlations_histogram.png")
    plt.savefig(save_file)
    plt.close()

