import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
import pickle


cutoff=10 # Cutoff frequency for low-pass filter in Hz
fs=200 # Frame rate in Hz


# PART 1
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


# PART 2
# --- BUTTER LOWPASS FILTER ---
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# --- CSV_TO_TRAJ_DATA ---
def CSV_To_traj_data_filter(file_path, marker_name, cutoff, fs):
    """
    Reads a CSV file containing trajectory data, processes it, and extracts specific columns based on predefined indices.

    Args:
        file_path (str): Path to the CSV file.
        marker_name (str): The marker name to extract data for.

    Returns:
        dict: A dictionary containing extracted data for the specified marker.
    """

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
    time = Frame / fs  # Time in seconds

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
        
        filtered_data = pd.Series(butter_lowpass_filter(raw_data, cutoff, fs, order=4))
        if skip:
            filtered_data = pd.concat([pd.Series([np.nan] * skip), filtered_data]).reset_index(drop=True)
        
        traj_data[label] = filtered_data


    return traj_data, Frame, time

# --- TRAJ_SPACE_DATA --
def Traj_Space_data(traj_data, fs):

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
        Jerk_X = traj_data[f"{marker}_AX"].diff() * fs  # Assuming the data is sampled at 200 Hz
        Jerk_Y = traj_data[f"{marker}_AY"].diff() * fs
        Jerk_Z = traj_data[f"{marker}_AZ"].diff() * fs

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

    return minima, peaks, minima_indices, peaks_indices

# Part 3
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

# --- FILTER FILES BY HAND ---
def filter_files_by_hand(file_list, hand):
    if hand not in ["right", "left"]:
        raise ValueError("hand must be 'right' or 'left'")
    return [
        file for file in file_list
        if (int(os.path.splitext(os.path.basename(file))[0][-2:]) % 2 == 1 if hand == "right" else int(os.path.splitext(os.path.basename(file))[0][-2:]) % 2 == 0)
    ]

# --- PROCESS TRAJECTORY DATA ---
def get_reach_segements(file_paths, BoxRange, prominence_threshold_speed, prominence_threshold_position, hand, cutoff, fs):

    # --- CHECK HAND ---
    if hand == "right":
        marker_name = "RFIN"
    elif hand == "left":
        marker_name = "LFIN"

    # --- CACHE TRAJECTORY DATA ---
    cached_data = {}
    for file_path in file_paths:
        traj_data = CSV_To_traj_data_filter(file_path, marker_name, cutoff, fs)[0]
        # traj_data = CSV_To_traj_data(file_path, marker_name)[0]
        # traj_data = CSV_To_traj_data(file_path)[0]
        traj_space = Traj_Space_data(traj_data, fs)
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
               (BoxRange[0]+10 > cached_data[file_path]["traj_data"][f"{marker_name}_X"][idx] > BoxRange[1])
        ]
        for file_path, indices in filtered_indices.items()
    }

    # --- CLASSIFY AND GROUP NON-ALTERNATING SEQUENCES ---
    def classify(x, hand):
        if hand == "right":
            if BoxRange[1] > x > BoxRange[2]: return 'start'
            if BoxRange[0]+10 > x > BoxRange[1]: return 'end'
        elif hand == "left":
            if BoxRange[0]+10 > x > BoxRange[1]: return 'start'
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

# --- PLOT RESULTS ---
def plot_Filtered_Trajectory_Components(file_paths, cached_data, final_selected_indices, BoxRange, plot_save_path, hand):
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
                elif BoxRange[0]+10 > x > BoxRange[1]:
                    ax.scatter(idx, data[idx], color="green", alpha=0.7)

        axes[-1].set_xlabel("Frame")

        plt.tight_layout()
        plt.title(f"Marker: {marker_name}")
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(plot_save_path, f"{os.path.basename(file_path)}_Filtered_Trajectory_Components.png"))
        plt.close()

    print("✅ Processing and plotting [Filtered Trajectory Components] completed successfully.")

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

# PART 4
# --- PROCESS DATE FUNCTION ---
def process_single_subject(date, traj_folder, Box_Traj_folder, figure_folder, DataProcess_folder, 
                 prominence_threshold_speed, prominence_threshold_position):
    print(f"Processing subject: {date}")
    
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
    processed_single_subject_data = {}
    for hand in ["right", "left"]:
        file_paths = filter_files_by_hand(tBBT_files, hand=hand)
        print(f"Number of files for {hand} hand: {len(file_paths)}")
        final_selected_indices, cached_data = get_reach_segements(
            file_paths, box_range, prominence_threshold_speed, prominence_threshold_position, hand=hand, cutoff=cutoff, fs=fs
        )
        processed_single_subject_data[hand] = (final_selected_indices, cached_data)

    # --- PLOT RESULTS ---    
        # plot_Filtered_Trajectory_Components(file_paths, cached_data, final_selected_indices, box_range, plot_save_path, hand=hand)

    # --- Save indices to CSV ---
    save_indices_to_single_csv(
        processed_single_subject_data["right"][0],
        processed_single_subject_data["left"][0],
        DataProcess_folder,
        tBBT_files
    )

    return processed_single_subject_data

# PART 5
# --- PROCESS EACH DATE FUNCTION ---
# def process_all_dates_ALL_IN_ONE(All_dates, Traj_folder, Box_Traj_folder, Figure_folder, DataProcess_folder, 
#                       prominence_threshold_speed, prominence_threshold_position):
#     """
#     Processes all dates and saves the results to a file.

#     Args:
#         All_dates (list): List of dates to process.
#         utils (module): Utility module containing the process_date function.
#         Traj_folder (str): Path to the trajectory folder.
#         Box_Traj_folder (str): Path to the box trajectory folder.
#         Figure_folder (str): Path to the figure folder.
#         DataProcess_folder (str): Path to the data processing folder.
#         prominence_threshold_speed (float): Threshold for speed prominence.
#         prominence_threshold_position (float): Threshold for position prominence.

#     Returns:
#         None
#     """
#     results = {}
#     for date in All_dates:
#         results[date] = process_single_subject(date, Traj_folder, Box_Traj_folder, Figure_folder, DataProcess_folder, 
#                                            prominence_threshold_speed, prominence_threshold_position)
#     # Save results to a file
#     save_results(results, DataProcess_folder)

def process_all_dates_separate(All_dates, Traj_folder, Box_Traj_folder, Figure_folder, DataProcess_folder, 
                      prominence_threshold_speed, prominence_threshold_position):
    """
    Processes all dates and saves the results for each subject to a separate file.

    Args:
        All_dates (list): List of dates to process.
        Traj_folder (str): Path to the trajectory folder.
        Box_Traj_folder (str): Path to the box trajectory folder.
        Figure_folder (str): Path to the figure folder.
        DataProcess_folder (str): Path to the data processing folder.
        prominence_threshold_speed (float): Threshold for speed prominence.
        prominence_threshold_position (float): Threshold for position prominence.

    Returns:
        None
    """
    for date in All_dates:
        subject_data = process_single_subject(date, Traj_folder, Box_Traj_folder, Figure_folder, DataProcess_folder, 
                                              prominence_threshold_speed, prominence_threshold_position)
        # Save each subject's data as a separate file
        subject_filename = f"{date.replace('/', '_')}_processed_data.pkl"
        save_results(subject_data, DataProcess_folder, filename=subject_filename)


# PART 6
# --- SAVE RESULTS FUNCTION ---
def save_results(results, DataProcess_folder, filename=None):
    """
    Saves the processed results to a file.

    Args:
        results (dict): The processed results to save.
        DataProcess_folder (str): Path to the data processing folder.
        filename (str, optional): Name of the file to save the results. Defaults to "processed_results.pkl".

    Returns:
        None
    """
    if filename is None:
        filename = "processed_results.pkl"  # Set default filename if not provided
    os.makedirs(DataProcess_folder, exist_ok=True)  # Ensure the folder exists
    results_file = os.path.join(DataProcess_folder, filename)
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    print(f"✅ Results saved to {results_file}")


# --- LOAD RESULTS FUNCTION ---
def load_results(DataProcess_folder, filename=None):
    if filename is None:
        filename = "processed_results.pkl"  # Set default filename if not provided
    
    results_file = os.path.join(DataProcess_folder, filename)
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, "rb") as f:
        results = pickle.load(f)
    
    return results

def load_selected_subject_results(All_dates, DataProcess_folder):
    """
    Loads the processed results for selected subjects and aggregates them into a single dictionary.

    Args:
        All_dates (list): List of dates to load results for.
        DataProcess_folder (str): Path to the data processing folder.

    Returns:
        dict: A dictionary containing the aggregated results for all selected subjects.
    """
    results = {}
    for date in All_dates:
        subject_filename = f"{date.replace('/', '_')}_processed_data.pkl"
        try:
            results[date] = load_results(DataProcess_folder, filename=subject_filename)
        except FileNotFoundError:
            print(f"Warning: Results file for {date} not found. Skipping.")
    return results

