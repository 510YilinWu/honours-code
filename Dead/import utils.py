import utils
import traj_utils
import os
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt


date = "06/26/2"
time_window_method = 1  # Choose 1, 2, or 3
window_size = 0.05  # 50 ms
prominence_threshold_speed = 600  # Adjust as needed
prominence_threshold_position = 80  # Adjust as needed
speed_threshold = 500  # Adjust as needed

Traj_folder = "/Volumes/MNHS-MoCap/Yilin-Honours/Subject/Traj/2025/"
Box_Traj_folder = "/Volumes/MNHS-MoCap/Yilin-Honours/Box/Traj/2025/"
save_path = "/Users/yilinwu/Desktop/honours/Thesis/figure"  # Adjust the save path as needed


tBBT_files = utils.find_bbt_files(Traj_folder, date, file_type = "tBBT")
if len(tBBT_files) != 64:
    raise ValueError(f"Expected 64 tBBT files, but found {len(tBBT_files)}")

Box_Traj_file = utils.find_bbt_files(Box_Traj_folder, date, file_type = "BOX_Cali")
if len(Box_Traj_file) != 1:
    raise ValueError(f"Expected 1 Box Trajectory file, but found {len(Box_Traj_file)}")


# Select only files at odd indices (right hand)
file_paths = [file for i, file in enumerate(tBBT_files) if i % 2 == 0] 
# file_paths = file_paths[14:15]
marker_name = "RFIN"  # Adjust marker name as needed

# # Select only files at even indices (left hand)
# file_paths = [file for i, file in enumerate(tBBT_files) if i % 2 == 1]  
# marker_name = "LFIN"  # Adjust marker name as needed


file_save_path = save_path
candidate_data = {}
for file_path in file_paths:
    # print(f"Processing file: {file_path}")

    # # Create a directory for saving results based on the file name
    file_name = os.path.basename(file_path).split('.')[0]
    # file_save_path = os.path.join(save_path, file_name)
    # os.makedirs(file_save_path, exist_ok=True)


    traj_data, Frame, time = traj_utils.CSV_To_traj_data(file_path)
    Traj_Space_data = traj_utils.Traj_Space_data(traj_data)
    speed_data = Traj_Space_data[marker_name][1]

    BoxRange = traj_utils.calculate_rangesByBoxTraj(Box_Traj_file)

    # minima, peaks, min_idx, peak_idx = traj_utils.find_local_minima_peaks(Traj_Space_data[marker_name][1], prominence_threshold_speed) # speed_data
    # minima, peaks, min_idx, peak_idx = traj_utils.find_local_minima_peaks(Traj_Space_data[marker_name][0], prominence_threshold_position) # position_data

    # Step 1: Find Local Minima for Each Signal
    minima_speed, peaks, min_idx_speed, peak_idx = traj_utils.find_local_minima_peaks(Traj_Space_data[marker_name][1], prominence_threshold_speed) # speed_data
    minima_position, peaks, min_idx_position, peak_idx = traj_utils.find_local_minima_peaks(Traj_Space_data[marker_name][0], prominence_threshold_position) # position_data
    # print(f"File: {file_name}, Minima Speed: {len(min_idx_speed)}, Minima Position: {len(min_idx_position)}")

    # Save minima and indices for each file
    candidate_data[file_path] = {
        "file": file_path,
        "minima_speed": minima_speed,
        "min_idx_speed": min_idx_speed,
        "minima_position": minima_position,
        "min_idx_position": min_idx_position
    }

# Calculate average and standard deviation of sorted minima_position and minima_speed across all files
def calculate_stats(data_key):
    sorted_data = sorted(
        [val for data in candidate_data.values() for val in data[data_key]]
    )[30:-30]
    avg = np.mean(sorted_data) if sorted_data else None
    std = np.std(sorted_data) if sorted_data else None
    print(f"Average {data_key} (excluding first 100 and last 100 indices): {avg}")
    print(f"Standard deviation of {data_key} (excluding first 100 and last 100 indices): {std}")
    return avg, std

ava_minima_position, ava_std = calculate_stats("minima_position")
# ava_minima_speed, ava_std = calculate_stats("minima_speed")

filtered_indices_list = {}
for file_path in file_paths:
    traj_data, Frame, time = traj_utils.CSV_To_traj_data(file_path)
    Traj_Space_data = traj_utils.Traj_Space_data(traj_data)

    position_data = Traj_Space_data[marker_name][0]

    min_idx_speed = candidate_data[file_path]["min_idx_speed"]

    filtered_indices = [
        idx for idx in min_idx_speed
        if ava_minima_position - 5 * ava_std <= position_data[idx] <= ava_minima_position + 5 * ava_std
    ]

    filtered_indices_list[file_path] = filtered_indices

# Check if the minimal length in each filtered_indices_list[file_path] is 32
for file_path, indices in filtered_indices_list.items():
    if len(indices) < 32:
        raise ValueError(f"Filtered indices for file {file_path} have less than 32 elements. Found: {len(indices)}")


filtered_indices_list_update = {}
for file_path, indices in filtered_indices_list.items():
    traj_data, Frame, time = traj_utils.CSV_To_traj_data(file_path)
    filtered_indices_list_update[file_path] = [
        idx for idx in indices
        if (BoxRange[1] > traj_data[f"{marker_name}_X"][idx] > BoxRange[2]) or 
            (BoxRange[0] > traj_data[f"{marker_name}_X"][idx] > BoxRange[1])
    ]

# Check if the minimal length in each filtered_indices_list_update[file_path] is 32
for file_path, indices in filtered_indices_list_update.items():
    if len(indices) < 32:
        raise ValueError(f"Filtered indices for file {file_path} have less than 32 elements. Found: {len(indices)}")




import utils
import traj_utils
import os
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt


date = "06/26/2"
time_window_method = 1  # Choose 1, 2, or 3
window_size = 0.05  # 50 ms
prominence_threshold_speed = 600  # Adjust as needed
prominence_threshold_position = 80  # Adjust as needed
speed_threshold = 500  # Adjust as needed

Traj_folder = "/Volumes/MNHS-MoCap/Yilin-Honours/Subject/Traj/2025/"
Box_Traj_folder = "/Volumes/MNHS-MoCap/Yilin-Honours/Box/Traj/2025/"
save_path = "/Users/yilinwu/Desktop/honours/Thesis/figure"  # Adjust the save path as needed


tBBT_files = utils.find_bbt_files(Traj_folder, date, file_type = "tBBT")
if len(tBBT_files) != 64:
    raise ValueError(f"Expected 64 tBBT files, but found {len(tBBT_files)}")

Box_Traj_file = utils.find_bbt_files(Box_Traj_folder, date, file_type = "BOX_Cali")
if len(Box_Traj_file) != 1:
    raise ValueError(f"Expected 1 Box Trajectory file, but found {len(Box_Traj_file)}")


# Select only files at odd indices (right hand)
file_paths = [file for i, file in enumerate(tBBT_files) if i % 2 == 0] 
# file_paths = file_paths[14:15]
marker_name = "RFIN"  # Adjust marker name as needed

# # Select only files at even indices (left hand)
# file_paths = [file for i, file in enumerate(tBBT_files) if i % 2 == 1]  
# marker_name = "LFIN"  # Adjust marker name as needed

def categorize_indices(indices, traj_data, marker_name, ranges):
    categorized = {g: [] for g in range(4)}
    for idx in indices:
        x_val = traj_data[f"{marker_name}_X"][idx]
        for g, (x_max, x_min) in ranges.items():
            if x_min < x_val < x_max:
                categorized[g].append(idx)
                break
    return categorized


file_save_path = save_path
candidate_data = {}
for file_path in file_paths:
    # print(f"Processing file: {file_path}")

    # # Create a directory for saving results based on the file name
    file_name = os.path.basename(file_path).split('.')[0]
    # file_save_path = os.path.join(save_path, file_name)
    # os.makedirs(file_save_path, exist_ok=True)


    traj_data, Frame, time = traj_utils.CSV_To_traj_data(file_path)
    Traj_Space_data = traj_utils.Traj_Space_data(traj_data)
    speed_data = Traj_Space_data[marker_name][1]

    BoxRange = traj_utils.calculate_rangesByBoxTraj(Box_Traj_file)

    # minima, peaks, min_idx, peak_idx = traj_utils.find_local_minima_peaks(Traj_Space_data[marker_name][1], prominence_threshold_speed) # speed_data
    # minima, peaks, min_idx, peak_idx = traj_utils.find_local_minima_peaks(Traj_Space_data[marker_name][0], prominence_threshold_position) # position_data

    # Step 1: Find Local Minima for Each Signal
    minima_speed, peaks, min_idx_speed, peak_idx = traj_utils.find_local_minima_peaks(Traj_Space_data[marker_name][1], prominence_threshold_speed) # speed_data
    minima_position, peaks, min_idx_position, peak_idx = traj_utils.find_local_minima_peaks(Traj_Space_data[marker_name][0], prominence_threshold_position) # position_data
    # print(f"File: {file_name}, Minima Speed: {len(min_idx_speed)}, Minima Position: {len(min_idx_position)}")

    # Save minima and indices for each file
    candidate_data[file_path] = {
        "file": file_path,
        "minima_speed": minima_speed,
        "min_idx_speed": min_idx_speed,
        "minima_position": minima_position,
        "min_idx_position": min_idx_position
    }

# # Check if the minimal length in each filtered_indices_list_update[file_path] is 32
# for file_path in file_paths:

#     # Load trajectory data for the current file
#     traj_data, Frame, time = traj_utils.CSV_To_traj_data(file_path)
#     Traj_Space_data = traj_utils.Traj_Space_data(traj_data)

#     # Create a figure for the current file
#     fig, axs = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
#     axs = axs.ravel()

#     # Data and labels for plotting
#     plot_info = [
#         (f"{marker_name}_X", "X"),
#         (Traj_Space_data[marker_name][0], "Position"),
#         (Traj_Space_data[marker_name][1], "Speed"),
#         (Traj_Space_data[marker_name][2], "Acceleration")
#     ]
#     for i, (data_key, label) in enumerate(plot_info):
#         y_data = traj_data[data_key] if isinstance(data_key, str) else data_key
#         axs[i].plot(Frame, y_data, label=label)

#         # Overlay min_idx_speed
#         axs[i].scatter([Frame[idx] for idx in min_idx_speed], [y_data[idx] for idx in min_idx_speed], color='red', label="Minima Speed")

#         axs[i].set_ylabel(label)
#         axs[i].set_title(f"{label} over Time")
#         axs[i].legend()

#     axs[-1].set_xlabel("Frame")  # Only set x-axis label on the bottom subplot

#     plt.tight_layout()
#     plt.show()
#     break











# Calculate average and standard deviation of sorted minima_position and minima_speed across all files
def calculate_stats(data_key):
    sorted_data = sorted(
        [val for data in candidate_data.values() for val in data[data_key]]
    )[30:-30]
    avg = np.mean(sorted_data) if sorted_data else None
    std = np.std(sorted_data) if sorted_data else None
    print(f"Average {data_key} (excluding first 100 and last 100 indices): {avg}")
    print(f"Standard deviation of {data_key} (excluding first 100 and last 100 indices): {std}")
    return avg, std

ava_minima_position, ava_std = calculate_stats("minima_position")
# ava_minima_speed, ava_std = calculate_stats("minima_speed")

filtered_indices_list = {}
for file_path in file_paths:
    traj_data, Frame, time = traj_utils.CSV_To_traj_data(file_path)
    Traj_Space_data = traj_utils.Traj_Space_data(traj_data)

    position_data = Traj_Space_data[marker_name][0]

    min_idx_speed = candidate_data[file_path]["min_idx_speed"]

    filtered_indices = [
        idx for idx in min_idx_speed
        if ava_minima_position - 5 * ava_std <= position_data[idx] <= ava_minima_position + 5 * ava_std
    ]

    filtered_indices_list[file_path] = filtered_indices

# Check if the minimal length in each filtered_indices_list[file_path] is 32
for file_path, indices in filtered_indices_list.items():
    if len(indices) < 32:
        raise ValueError(f"Filtered indices for file {file_path} have less than 32 elements. Found: {len(indices)}")


filtered_indices_list_update = {}
for file_path, indices in filtered_indices_list.items():
    traj_data, Frame, time = traj_utils.CSV_To_traj_data(file_path)
    filtered_indices_list_update[file_path] = [
        idx for idx in indices
        if (BoxRange[1] > traj_data[f"{marker_name}_X"][idx] > BoxRange[2]) or 
            (BoxRange[0] > traj_data[f"{marker_name}_X"][idx] > BoxRange[1])
    ]

# Check if the minimal length in each filtered_indices_list_update[file_path] is 32
for file_path, indices in filtered_indices_list_update.items():
    if len(indices) < 32:
        raise ValueError(f"Filtered indices for file {file_path} have less than 32 elements. Found: {len(indices)}")





    # # Load trajectory data for the current file
    # traj_data, Frame, time = traj_utils.CSV_To_traj_data(file_path)
    # Traj_Space_data = traj_utils.Traj_Space_data(traj_data)

    # # Create a figure for the current file
    # fig, axs = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    # axs = axs.ravel()

    # # Data and labels for plotting
    # plot_info = [
    #     (f"{marker_name}_X", "X"),
    #     (Traj_Space_data[marker_name][0], "Position"),
    #     (Traj_Space_data[marker_name][1], "Speed"),
    #     (Traj_Space_data[marker_name][2], "Acceleration")
    # ]

    # for i, (data_key, label) in enumerate(plot_info):
    #     y_data = traj_data[data_key] if isinstance(data_key, str) else data_key
    #     axs[i].plot(Frame, y_data, label=label)

    #     # Overlay filtered indices
    #     for idx in filtered_indices_list_update[file_path]:
    #         if BoxRange[1] > traj_data[f"{marker_name}_X"][idx] > BoxRange[2]:
    #             axs[i].scatter(Frame[idx], y_data[idx], color='red', label="Filtered (Red)")
    #         elif BoxRange[0] > traj_data[f"{marker_name}_X"][idx] > BoxRange[1]:
    #             axs[i].scatter(Frame[idx], y_data[idx], color='green', label="Filtered (Green)")

    #     axs[i].set_ylabel(label)
    #     axs[i].set_title(f"{label} over Time")
    #     # axs[i].legend()

    # axs[-1].set_xlabel("Frame")  # Only set x-axis label on the bottom subplot

    # plt.tight_layout()
    # plt.show()
























# tStart_indices = {}
# tEnd_indices = {}
# for file_path in file_paths:
#     traj_data, Frame, time = traj_utils.CSV_To_traj_data(file_path)  # Load traj_data for the current file
#     tStart_indices[file_path] = [idx for idx in filtered_indices_list[file_path] if BoxRange[1] > traj_data[f"{marker_name}_X"][idx] > BoxRange[2]]
#     tEnd_indices[file_path] = [idx for idx in filtered_indices_list[file_path] if BoxRange[0] > traj_data[f"{marker_name}_X"][idx] > BoxRange[1]]
#     print(len(tStart_indices[file_path]), len(tEnd_indices[file_path]))











    # # Create a figure with two subplots
    # fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # # Plot minima_speed and minima_position overlay on speed data
    # axs[0].plot(Frame, Traj_Space_data[marker_name][1], label="Speed Data", color='blue')
    # axs[0].scatter([Frame[idx] for idx in min_idx_speed], [Traj_Space_data[marker_name][1][idx] for idx in min_idx_speed], color='red', label="Minima Speed")
    # axs[0].scatter([Frame[idx] for idx in min_idx_position], [Traj_Space_data[marker_name][1][idx] for idx in min_idx_position], color='green', label="Minima Position")
    # axs[0].set_ylabel("Speed")
    # axs[0].set_title(f"Speed Data with Minima for {marker_name}")
    # axs[0].legend()
    # axs[0].grid()

    # # Add file information to the plot
    # axs[0].text(0.5, 0.9, f"File: {file_name}, Minima Speed: {len(min_idx_speed)}, Minima Position: {len(min_idx_position)}", 
    #             transform=axs[0].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='center')

    # # Plot traj_data[f"{marker_name}_X"]
    # axs[1].plot(Frame, traj_data[f"{marker_name}_X"], label=f"{marker_name}_X Data", color='purple')
    # axs[1].scatter([Frame[idx] for idx in min_idx_speed], [traj_data[f"{marker_name}_X"][idx] for idx in min_idx_speed], color='red', label="Minima Speed")
    # axs[1].scatter([Frame[idx] for idx in min_idx_position], [traj_data[f"{marker_name}_X"][idx] for idx in min_idx_position], color='green', label="Minima Position")
    # axs[1].set_xlabel("Frame")
    # axs[1].set_ylabel(f"{marker_name}_X")
    # axs[1].set_title(f"{marker_name}_X Data over Frames")
    # axs[1].legend()
    # axs[1].grid()

    # # Adjust layout and save the figure
    # plt.tight_layout()
    # plt.show()
    # # plt.savefig(os.path.join(file_save_path, f"{file_name}_speed_and_X_overlay.png"))
    # # plt.close()






    # fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    # axs = axs.ravel()

    # # Data and labels for plotting
    # plot_info = [
    #     (Traj_Space_data[marker_name][0], "Position"),
    #     (Traj_Space_data[marker_name][1], "Speed"),
    # ]

    # for i, (data_key, label) in enumerate(plot_info):
    #     for file_path in file_paths:
    #         traj_data, Frame, time = traj_utils.CSV_To_traj_data(file_path)
    #         Traj_Space_data = traj_utils.Traj_Space_data(traj_data)
    #         y_data = data_key
    #         axs[i].plot(Frame, y_data, label=f"{label} ({os.path.basename(file_path)})")

    #         # Plot min_idx_speed and min_idx_position in different colors
    #         if label == "Speed":
    #             axs[i].scatter([Frame[idx] for idx in min_idx_speed], [y_data[idx] for idx in min_idx_speed], color='blue', label=f'Minima Speed ({os.path.basename(file_path)})')
    #         elif label == "Position":
    #             axs[i].scatter([Frame[idx] for idx in min_idx_position], [y_data[idx] for idx in min_idx_position], color='orange', label=f'Minima Position ({os.path.basename(file_path)})')
        
    #     axs[i].set_ylabel(label)
    #     axs[i].set_title(f"{label} over Frame")
    #     axs[i].legend()

    # axs[-1].set_xlabel("Frame")  # Only set x-axis label on the bottom subplot

    # plt.tight_layout()
    # plt.show()




    # # Create a single figure with two subplots sharing the x-axis
    # fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # # Plot Position Data with Local Minima
    # axs[0].plot(Frame, Traj_Space_data[marker_name][0], label="Position")
    # axs[0].scatter([Frame[idx] for idx in min_idx], [Traj_Space_data[marker_name][0][idx] for idx in min_idx], color='orange', label='Local Minima')
    # axs[0].set_ylabel("Position")
    # axs[0].set_title(f"Position Data with Local Minima for {marker_name}")
    # axs[0].legend()
    # axs[0].grid()

    # # Plot Speed Data with Local Minima
    # axs[1].plot(Frame, Traj_Space_data[marker_name][1], label="Speed")
    # axs[1].scatter([Frame[idx] for idx in min_idx], [Traj_Space_data[marker_name][1][idx] for idx in min_idx], color='orange', label='Local Minima')
    # axs[1].set_ylabel("Speed")
    # axs[1].set_title(f"Speed Data with Local Minima for {marker_name}")
    # axs[1].legend()
    # axs[1].grid()

    # # Set x-axis label only on the bottom subplot
    # axs[1].set_xlabel("Frame")

    # # Adjust layout and show the figure
    # plt.tight_layout()
    # plt.show()






    # minima, peaks, min_idx, peak_idx = traj_utils.find_local_minima_peaks(Traj_Space_data[marker_name][0], prominence_threshold_speed) # position_data


    # minima, peaks, min_idx, peak_idx = traj_utils.find_local_minima_peaks(Traj_Space_data[marker_name][1], prominence_threshold_speed) # speed_data
    # minima, peaks, min_idx, peak_idx = traj_utils.find_local_minima_peaks(Traj_Space_data[marker_name][2], prominence_threshold_speed) # acceleration_data

    # minima, peaks, min_idx, peak_idx = traj_utils.find_local_minima_peaks(speed_data, prominence_threshold_speed)





    # tStart_indices = [idx for idx in min_idx if BoxRange[1] > traj_data[f"{marker_name}_X"][idx] > BoxRange[2]]
    # tEnd_indices = [idx for idx in min_idx if BoxRange[0] > traj_data[f"{marker_name}_X"][idx] > BoxRange[1]]

    # print(len(tStart_indices), len(tEnd_indices))



    # filter tStart_indices: 
    # traj_data[f"{marker_name}_Z"][tStart_indices] should be local minimal
    # Traj_Space_data[marker_name][0][tStart_indices] should be local minimal
    # Traj_Space_data[marker_name][1][tStart_indices] should be local minimal
    # Traj_Space_data[marker_name][2][tStart_indices] should not be local minimal


    # filter tEnd_indices: 
    # traj_data[f"{marker_name}_Z"][tStart_indices] should be local minimal
    # Traj_Space_data[marker_name][0][tStart_indices] should be local minimal
    # Traj_Space_data[marker_name][1][tStart_indices] should be local minimal
    # Traj_Space_data[marker_name][2][tStart_indices] should be local minimal













    # # print(traj_data[f"{marker_name}_X"][tStart_indices])
    # # print(traj_data[f"{marker_name}_X"][tEnd_indices])

    # fig, axs = plt.subplots(6, 1, figsize=(15, 10), sharex=True)
    # axs = axs.ravel()

    # # Data and labels for plotting
    # plot_info = [
    #     (f"{marker_name}_X", "X"),
    #     (f"{marker_name}_Y", "Y"),
    #     (f"{marker_name}_Z", "Z"),
    #     (Traj_Space_data[marker_name][0], "Position"),
    #     (Traj_Space_data[marker_name][1], "Speed"),
    #     (Traj_Space_data[marker_name][2], "Acceleration")
    # ]

    # for i, (data_key, label) in enumerate(plot_info):
    #     y_data = traj_data[data_key] if isinstance(data_key, str) else data_key
    #     axs[i].plot(Frame, y_data, label=label)
    #     # axs[i].scatter([Frame[idx] for idx in tStart_indices], [y_data[idx] for idx in tStart_indices], color='g', label='tStart')
    #     # axs[i].scatter([Frame[idx] for idx in tEnd_indices], [y_data[idx] for idx in tEnd_indices], color='r', label='tEnd')
    #     axs[i].scatter([Frame[idx] for idx in min_idx_speed], [y_data[idx] for idx in min_idx_speed], color='red', label="Minima Speed")
    #     axs[i].scatter([Frame[idx] for idx in min_idx_position], [y_data[idx] for idx in min_idx_position], color='green', label="Minima Position")
    
    #     axs[i].set_ylabel(label)
    #     axs[i].set_title(f"{label} over Frame")
    #     axs[i].legend()

    # axs[-1].set_xlabel("Frame")  # Only set x-axis label on the bottom subplot

    # plt.tight_layout()
    # plt.show()
    # # plt.savefig(os.path.join(file_save_path, f"{file_name}_subplots.png"))
    # # plt.close()

