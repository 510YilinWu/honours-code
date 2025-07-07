import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import utils
import traj_utils

# --- PARAMETERS ---
date = "06/24/1"
time_window_method = 1
window_size = 0.05
prominence_threshold_speed = 600
prominence_threshold_position = 80
speed_threshold = 500

Traj_folder = "/Volumes/MNHS-MoCap/Yilin-Honours/Subject/Traj/2025/"
Box_Traj_folder = "/Volumes/MNHS-MoCap/Yilin-Honours/Box/Traj/2025/"
save_path = "/Users/yilinwu/Desktop/honours/Thesis/figure"

marker_name = "RFIN"  # Use "LFIN" for left hand

# --- FIND FILES ---
tBBT_files = utils.find_bbt_files(Traj_folder, date, file_type="tBBT")
if len(tBBT_files) != 64:
    raise ValueError(f"Expected 64 tBBT files, but found {len(tBBT_files)}")

Box_Traj_file = utils.find_bbt_files(Box_Traj_folder, date, file_type="BOX_Cali")
if len(Box_Traj_file) != 1:
    raise ValueError(f"Expected 1 Box Trajectory file, but found {len(Box_Traj_file)}")

# # Select only odd-indexed (right-hand) or even-indexed (left-hand) files
# file_paths = [file for i, file in enumerate(tBBT_files) if i % 2 == 0]


# Filter odd-numbered (right-hand) or even-numbered (left-hand) files
def filter_files_by_hand(file_list, hand):
    if hand not in ["right", "left"]:
        raise ValueError("hand must be 'right' or 'left'")
    return [
        file for file in file_list
        if (int(os.path.splitext(os.path.basename(file))[0][-2:]) % 2 == 1 if hand == "right" else int(os.path.splitext(os.path.basename(file))[0][-2:]) % 2 == 0)
    ]

# Example usage: filter odd-numbered files
file_paths = filter_files_by_hand(tBBT_files, hand="right")


# --- CACHE TRAJECTORY DATA ---
cached_data = {}
for file_path in file_paths:
    traj_data, Frame, time = traj_utils.CSV_To_traj_data(file_path)
    traj_space = traj_utils.Traj_Space_data(traj_data)
    cached_data[file_path] = {
        "traj_data": traj_data,
        "traj_space": traj_space,
        "position": traj_space[marker_name][0],
        "speed": traj_space[marker_name][1]
    }

# --- BOX RANGE ---
BoxRange = traj_utils.calculate_rangesByBoxTraj(Box_Traj_file)

# --- DETECT MINIMA ---
candidate_data = {}
for file_path in file_paths:
    speed = cached_data[file_path]["speed"]
    position = cached_data[file_path]["position"]

    # Detect local minima
    _, _, min_idx_speed, _ = traj_utils.find_local_minima_peaks(speed, prominence_threshold_speed)
    _, _, min_idx_position, _ = traj_utils.find_local_minima_peaks(position, prominence_threshold_position)

    candidate_data[file_path] = {
        "min_idx_speed": min_idx_speed,
        "min_idx_position": min_idx_position,
        "minima_speed_values": speed[min_idx_speed],
        "minima_position_values": position[min_idx_position]
    }


# --- AGGREGATE STATS ---
all_minima_positions = np.concatenate([data["minima_position_values"] for data in candidate_data.values()])
all_minima_positions = np.sort(all_minima_positions)[30:-30]  # exclude outliers

if len(all_minima_positions) == 0:
    raise ValueError("No valid minima_position values found for statistics.")

ava_minima_position = np.mean(all_minima_positions)
ava_std = np.std(all_minima_positions)

print(f"Average minima position (excl. 30 from each end): {ava_minima_position}")
print(f"Standard deviation: {ava_std}")

# --- FILTER BY POSITION RANGE ---
filtered_indices = {}
for file_path in file_paths:
    pos = cached_data[file_path]["position"]
    idx_speed = candidate_data[file_path]["min_idx_speed"]

    # Filter within ±5 std of average position
    valid_idx = [i for i in idx_speed if ava_minima_position - 5*ava_std <= pos[i] <= ava_minima_position + 12*ava_std]
    filtered_indices[file_path] = valid_idx

# --- CHECK LENGTH AFTER POSITION FILTER ---
for file_path, indices in filtered_indices.items():
    if len(indices) < 32:
        raise ValueError(f"Too few indices after position filter in file {file_path}: {len(indices)}")

# --- FILTER BY BOX RANGE ---
filtered_indices_final = {}
for file_path, indices in filtered_indices.items():
    traj_data = cached_data[file_path]["traj_data"]
    x_vals = np.array([traj_data[f"{marker_name}_X"][i] for i in indices])
    valid_idx = [
        idx for idx, x in zip(indices, x_vals)
        if (BoxRange[1] > x > BoxRange[2]) or (BoxRange[0] > x > BoxRange[1])
    ]
    filtered_indices_final[file_path] = valid_idx

# --- CHECK LENGTH AFTER BOX FILTER ---
for file_path, indices in filtered_indices_final.items():
    if len(indices) < 32:
        raise ValueError(f"Too few indices after box filter in file {file_path}: {len(indices)}")

print("✅ All files passed filtering with >= 32 valid events.")


# --- FIND NON-ALTERNATING SEQUENCES ---
def classify(x):
    if BoxRange[1] > x > BoxRange[2]: return 'start'
    if BoxRange[0] > x > BoxRange[1]: return 'end'
    return None


# --- CLASSIFY AND GROUP NON-ALTERNATING SEQUENCES ---
filtered_indices_classified = {}

for file_path, indices in filtered_indices_final.items():
    traj_data = cached_data[file_path]["traj_data"]
    x_vals = np.array([traj_data[f"{marker_name}_X"][i] for i in indices])
    
    classified_indices = []
    for idx, x in zip(indices, x_vals):
        classification = classify(x)
        if classification:
            classified_indices.append((idx, classification))
    
    filtered_indices_classified[file_path] = classified_indices

print("✅ Classified indices with start/end labels.")


# --- FIND NON-ALTERNATING GROUPS AND SELECT BASED ON ACCELERATION ---
# Groups consecutive labels that are the same.
# Selects representative indices from each group based on their acceleration.
# Outputs a cleaned set of "start" and "end" indices that are more meaningful than just using all labeled points.

final_selected_indices = {}

for file_path, classified_indices in filtered_indices_classified.items():
    traj_data = cached_data[file_path]["traj_data"]
    traj_space = cached_data[file_path]["traj_space"]

    acceleration = traj_space[marker_name][2]

    # Group non-alternating sequences
    grouped_indices = []
    current_group = []
    last_label = None

    for idx, label in classified_indices:
        if label == last_label:
            current_group.append(idx)
        else:
            if current_group:
                grouped_indices.append((last_label, current_group))
            current_group = [idx]
            last_label = label

    if current_group:
        grouped_indices.append((last_label, current_group))

    # Select indices based on acceleration
    selected_indices = []
    for label, group in grouped_indices:
        if label == "start":
            # Pick the index with the highest acceleration
            max_acc_idx = max(group, key=lambda i: acceleration[i])
            selected_indices.append(max_acc_idx)
        elif label == "end":
            # Pick the index with the lowest acceleration
            min_acc_idx = min(group, key=lambda i: acceleration[i])
            selected_indices.append(min_acc_idx)

    final_selected_indices[file_path] = selected_indices

print("✅ Final selected indices based on acceleration.")

# --- CHECK LENGTH AFTER BOX FILTER ---
for file_path, indices in final_selected_indices.items():
    if len(indices) < 32:
        # raise ValueError(f"Too few indices after box filter in file {file_path}: {len(indices)}")
        print(f"Too few indices after box filter in file {file_path}: {len(indices)}")
    if len(indices) > 32:
        # raise ValueError(f"Too many indices after box filter in file {file_path}: {len(indices)}")
        print(f"Too many indices after box filter in file {file_path}: {len(indices)}")

print("✅ All files passed filtering with = 32 valid events.")



for file_path, indices in final_selected_indices.items():
    traj_data = cached_data[file_path]["traj_data"]
    traj_space = cached_data[file_path]["traj_space"]

    # Separate indices into "start" and "end"
    starts = [idx for idx in indices if classify(traj_data[f"{marker_name}_X"][idx]) == "start"]
    ends = [idx for idx in indices if classify(traj_data[f"{marker_name}_X"][idx]) == "end"]

    # Ensure each "start" has a corresponding "end"
    matched_indices = []
    while starts and ends:
        start = starts.pop(0)
        end = ends.pop(0)
        if start < end:  # Ensure logical order
            matched_indices.extend([start, end])

    final_selected_indices[file_path] = matched_indices

# --- CHECK LENGTH AFTER BOX FILTER ---
for file_path, indices in final_selected_indices.items():
    if len(indices) < 32:
        raise ValueError(f"Too few indices after box filter in file {file_path}: {len(indices)}")
    if len(indices) > 32:
        raise ValueError(f"Too many indices after box filter in file {file_path}: {len(indices)}")

print("✅ All files passed filtering with = 32 valid events.")



# --- PLOT RESULTS ---
for file_path in file_paths:
    traj_data = cached_data[file_path]["traj_data"]
    traj_space = cached_data[file_path]["traj_space"]
    filtered_indices = final_selected_indices[file_path]

    # Data and labels for plotting
    plot_info = [
        (traj_data[f"{marker_name}_X"], "X"),
        (traj_space[marker_name][0], "Position"),
        (traj_space[marker_name][1], "Speed"),
        (traj_space[marker_name][2], "Acceleration")
    ]

    # Create subplots
    fig, axes = plt.subplots(len(plot_info), 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Filtered Results for {os.path.basename(file_path)}")

    for ax, (data, label) in zip(axes, plot_info):
        ax.plot(data, label=label)
        ax.set_ylabel(label)
        ax.legend()

        # Overlay filtered indices with color coding
        x_vals = np.array([traj_data[f"{marker_name}_X"][i] for i in filtered_indices])
        # for idx, x in zip(filtered_indices, x_vals):
        #     if BoxRange[1] > x > BoxRange[2]:
        #         ax.axvline(idx, color="red", linestyle="--", alpha=0.7, label="BoxRange[1] > x > BoxRange[2]")
        #     elif BoxRange[0] > x > BoxRange[1]:
        #         ax.axvline(idx, color="green", linestyle="--", alpha=0.7, label="BoxRange[0] > x > BoxRange[1]")

        for idx, x in zip(filtered_indices, x_vals):
            if BoxRange[1] > x > BoxRange[2]:
                ax.scatter(idx, data[idx], color="red", label="BoxRange[1] > x > BoxRange[2]", alpha=0.7)
            elif BoxRange[0] > x > BoxRange[1]:
                ax.scatter(idx, data[idx], color="green", label="BoxRange[0] > x > BoxRange[1]", alpha=0.7)

    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the plot
    save_file = os.path.join(save_path, f"{os.path.basename(file_path)}_filtered_plot.png")
    plt.savefig(save_file)
    plt.close()

print("✅ Plots saved successfully.")
