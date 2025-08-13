import utils

date = "06/26/2"
time_window_method = 1  # Choose 1, 2, or 3
window_size = 0.05  # 50 ms
prominence_threshold_speed = 600  # Adjust as needed
speed_threshold = 500  # Adjust as needed

Traj_folder = "/Volumes/MNHS-MoCap/Yilin-Honours/Subject/Traj/2025/"
Box_Traj_folder = "/Volumes/MNHS-MoCap/Yilin-Honours/Box/Traj/2025/"
save_path = "/Users/yilinwu/Desktop/honours/Thesis/figure"  # Adjust the save path as needed

tBBT_files = utils.find_bbt_files(Traj_folder, date, file_type = "tBBT")
# print(f"tBBT files: {tBBT_files}\nNumber of tBBT files: {len(tBBT_files)}")
if len(tBBT_files) != 64:
    raise ValueError(f"Expected 64 tBBT files, but found {len(tBBT_files)}")

Box_Traj_file = utils.find_bbt_files(Box_Traj_folder, date, file_type = "BOX_Cali")
# print("Box Trajectory files:", Box_Traj_file)
# print("Number of Box Trajectory files:", len(Box_Traj_file))
if len(Box_Traj_file) != 1:
    raise ValueError(f"Expected 1 Box Trajectory file, but found {len(Box_Traj_file)}")

# Select only files at odd indices (right hand)
file_paths = [file for i, file in enumerate(tBBT_files) if i % 2 == 0] 


file_paths = file_paths[14:15]
marker_name = "RFIN"  # Adjust marker name as needed

# # Select only files at even indices (left hand)
# file_paths = [file for i, file in enumerate(tBBT_files) if i % 2 == 1]  
# marker_name = "LFIN"  # Adjust marker name as needed

# results = utils.process_files(file_paths, Box_Traj_file, marker_name, prominence_threshold_speed, speed_threshold, window_size, time_window_method, save_path)

import os
import traj_utils

def group_candidates_by_time(indices, threshold=700):
    if not indices:
        return []
    indices = sorted(indices)
    groups, current_group = [], [indices[0]]
    for idx in indices[1:]:
        if all(abs(idx - existing) <= threshold for existing in current_group):
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]
    groups.append(current_group)
    return groups

def select_best_candidate(group, traj_data, speed_data, marker_name, target_range):
    x_vals = [traj_data[f"{marker_name}_X"][idx] for idx in group]
    center = sum(target_range) / 2
    distances = [abs(x - center) for x in x_vals]
    min_dist = min(distances)

    candidates = [i for i, d in enumerate(distances) if d == min_dist]

    if len(candidates) == 1:
        return group[candidates[0]]
    speeds = [speed_data[group[i]] for i in candidates]
    return group[candidates[speeds.index(min(speeds))]]

def filter_grouped_indices(grouped, traj_data, speed_data, marker_name, ranges, forward=True):
    result = {}
    previous_group_selected = {}

    for group_id in sorted(grouped.keys()):
        indices = grouped[group_id]
        subgroups = group_candidates_by_time(indices)
        selected_indices = []

        for i, subgroup in enumerate(subgroups):
            # Select the best candidate from the subgroup
            candidate = select_best_candidate(subgroup, traj_data, speed_data, marker_name, ranges[group_id])

            # If forward checking is enabled, ensure the candidate is valid compared to the previous group
            if forward and group_id > 0:
                previous_group = previous_group_selected.get(group_id - 1, [])
                if i < len(previous_group):
                    valid_candidates = [idx for idx in subgroup if idx >= previous_group[i] + 200]
                    if valid_candidates:
                        candidate = select_best_candidate(valid_candidates, traj_data, speed_data, marker_name, ranges[group_id])

            selected_indices.append(candidate)

        # Store the selected indices for the current group
        result[group_id] = sorted(selected_indices)
        previous_group_selected[group_id] = selected_indices

    return result

def categorize_indices(indices, traj_data, marker_name, ranges):
    categorized = {g: [] for g in range(4)}
    for idx in indices:
        x_val = traj_data[f"{marker_name}_X"][idx]
        for g, (x_max, x_min) in ranges.items():
            if x_min < x_val < x_max:
                categorized[g].append(idx)
                break
    return categorized

for file_path in file_paths:
    print(f"Processing file: {file_path}")
    file_name = os.path.basename(file_path).split('.')[0]
    file_save_path = os.path.join(save_path, file_name)
    os.makedirs(file_save_path, exist_ok=True)

    traj_data, Frame, time = traj_utils.CSV_To_traj_data(file_path)
    Traj_Space_data = traj_utils.Traj_Space_data(traj_data)
    speed_data = Traj_Space_data[marker_name][1]
    BoxRange = traj_utils.calculate_rangesByBoxTraj(Box_Traj_file)

    minima, peaks, min_idx, peak_idx = traj_utils.find_local_minima_peaks(speed_data, prominence_threshold_speed)

    tEnd_indices = [idx for idx in min_idx if BoxRange[0] > traj_data[f"{marker_name}_X"][idx] > BoxRange[1]]
    tStart_indices = [idx for idx in min_idx if BoxRange[1] > traj_data[f"{marker_name}_X"][idx] > BoxRange[2]]

    print(f"tStart_indices: {sorted(tStart_indices)}")
    print(f"tEnd_indices: {sorted(tEnd_indices)}")

    print(traj_data[f"{marker_name}_X"][tStart_indices])
    print(traj_data[f"{marker_name}_X"][tEnd_indices])

    
    print(len(tStart_indices), len(tEnd_indices))

    start_ranges = {0: (-35, -79), 1: (-80, -115), 2: (-120, -165), 3: (-170, -225)}
    end_ranges   = {0: (110, 75),  1: (145, 115),  2: (188, 150),  3: (235, 180)}

    if len(tStart_indices) == 16:
        sorted_starts = sorted(tStart_indices)
        final_tStart_filtered = {i: sorted_starts[i*4:(i+1)*4] for i in range(4)}
    else:
        start_grouped = categorize_indices(tStart_indices, traj_data, marker_name, start_ranges)
        final_tStart_filtered = filter_grouped_indices(start_grouped, traj_data, speed_data, marker_name, start_ranges)
    
    print(f"\nCategorized tStart indices by group: {final_tStart_filtered}")

    if len(tEnd_indices) == 16:
        sorted_ends = sorted(tEnd_indices)
        final_tEnd_filtered = {i: sorted_ends[i*4:(i+1)*4] for i in range(4)}
    else:
        end_grouped = categorize_indices(tEnd_indices, traj_data, marker_name, end_ranges)
        final_tEnd_filtered = filter_grouped_indices(end_grouped, traj_data, speed_data, marker_name, end_ranges)
    
    print(f"\nCategorized tEnd indices by group: {final_tEnd_filtered}")

    # tStart_indices = [idx for group in final_tStart_filtered.values() for idx in group]
    # tStart_indices = sorted([idx for group in final_tStart_filtered.values() for idx in group])

    # tEnd_indices = sorted([idx for group in final_tEnd_filtered.values() for idx in group])


    # check if group 0 0 > group 1 0 > group 2 0 > group 3 0 > group 0 1 > group 1 1 > group 2 1 > group 3 1 > group 0 2 > group 1 2 > group 2 2 > group 3 2 > group 0 3 > group 1 3 > group 2 3 > group 3 3

    # Check if the indices follow the required order
    required_order = []
    for i in range(4):
        for j in range(4):
            required_order.append((j, i))

    flattened_starts = [final_tStart_filtered[group][i] for group, i in required_order]
    # Re-filter flattened_starts[i] using its subgroup (excluding itself)
    for idx, (group_id, sub_id) in enumerate(required_order):
        current_index = flattened_starts[idx]
        subgroup = final_tStart_filtered[group_id].copy()

        if current_index not in subgroup:
            continue  # Skip if value was not in original group

        # Remove current index from subgroup to re-evaluate
        subgroup.remove(current_index)

        # Only re-select if subgroup has alternatives
        if len(subgroup) > 0:
            new_candidate = select_best_candidate(subgroup, traj_data, speed_data, marker_name, start_ranges[group_id])
            flattened_starts[idx] = new_candidate





    flattened_ends = [final_tEnd_filtered[group][i] for group, i in required_order]

    print(f"Flattened tStart_indices: {flattened_starts}")
    print(f"Flattened tEnd_indices: {flattened_ends}")

    for i in range(len(flattened_starts) - 1):
        if not flattened_starts[i] < flattened_starts[i + 1]:
            print(f"Problematic tStart_indices: {flattened_starts[i]} (index {i}) and {flattened_starts[i + 1]} (index {i + 1})")
    for i in range(len(flattened_ends) - 1):
        if not flattened_ends[i] < flattened_ends[i + 1]:
            print(f"Problematic tEnd_indices: {flattened_ends[i]} (index {i}) and {flattened_ends[i + 1]} (index {i + 1})")
            raise ValueError("tEnd_indices do not follow the required order.")
    if not all(flattened_ends[i] < flattened_ends[i + 1] for i in range(len(flattened_ends) - 1)):
        raise ValueError("tEnd_indices do not follow the required order.")

    print("tStart_indices and tEnd_indices follow the required order.")



    # print(f"Filtered tStart_indices: {sorted(tStart_indices)}")
    # print(f"Filtered tEnd_indices: {sorted(tEnd_indices)}")



    

    # Ensure each start is followed by an end
    if len(tStart_indices) != len(tEnd_indices) or any(start >= end for start, end in zip(tStart_indices, tEnd_indices)):
        traj_utils.plot_x_speed_one_extrema_space(
            time, Traj_Space_data, traj_data,
            traj_data[f"{marker_name}_X"][tStart_indices],
            traj_data[f"{marker_name}_X"][tEnd_indices],
            marker_name, file_save_path
        )
        raise ValueError("Mismatch between tStart_indices and tEnd_indices: Each start must be followed by an end.")

    if len(tStart_indices) != 16 or len(tEnd_indices) != 16 or len(tStart_indices) != len(tEnd_indices):
        traj_utils.plot_x_speed_one_extrema_space(
            time, Traj_Space_data, traj_data,
            traj_data[f"{marker_name}_X"][tStart_indices],
            traj_data[f"{marker_name}_X"][tEnd_indices],
            marker_name, file_save_path
        )
        raise ValueError(f"Expected 16 tStart_indices and 16 tEnd_indices with equal lengths, but got {len(tStart_indices)} tStart_indices and {len(tEnd_indices)} tEnd_indices")



