import os
import numpy as np



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
    
    return sorted(bbt_files)

Traj_folder = "/Users/yilinwu/Desktop/honours data/Yilin-Honours/Subject/Traj/"
date = "06/13"
# sBBT_files = find_bbt_files(Traj_folder, date, file_type = "sBBT")
# print("sBBT files:", sBBT_files)

# iBBT_files = find_bbt_files(Traj_folder, date, file_type = "iBBT")
# print("iBBT files:", iBBT_files)

tBBT_files = find_bbt_files(Traj_folder, date, file_type = "tBBT")
print("tBBT files:", tBBT_files)




import traj_utils
from scipy.stats import pearsonr
import numpy as np

def process_files(file_paths, BoxTrajfile_path, marker_name, prominence_threshold_speed, speed_threshold, window_size, time_window_method, save_path):
    """
    Processes multiple trajectory files to analyze motion data and calculate metrics.
    """
    results = {}
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        
        # Extract trajectory data
        traj_data, Frame, time = traj_utils.CSV_To_traj_data(file_path)

        # traj_data = apply_butterworth_filter(traj_data, cutoff=55, fs=200, order=4)

        Traj_Space_data = traj_utils.Traj_Space_data(traj_data)


        

        # Find local minima and peaks in speed data
        speed_minima, speed_peaks = traj_utils.find_local_minima_peaks(Traj_Space_data[marker_name][1], prominence_threshold_speed)

        # Define segments based on speed thresholds
        speed_segments = traj_utils.find_speed_segments(marker_name, Traj_Space_data, time, speed_threshold, speed_peaks)
                
        # Classify the speed segments into reach and return segments
        # Note: Adjust the rfin_x_range and lfin_x_threshold based on specific requirements
        lfin_x_range, rfin_x_range = traj_utils.calculate_rangesByBoxTraj(BoxTrajfile_path)
        reach_speed_segments, return_speed_segments = traj_utils.classify_speed_segments(speed_segments, traj_data, marker_name, time, lfin_x_range, rfin_x_range)

        print(f"Number of reach segments found: {len(reach_speed_segments)}")

        # Calculate reach durations and distances
        reach_durations = traj_utils.calculate_reach_durations(reach_speed_segments)
        reach_distances = traj_utils.calculate_reach_distances(Traj_Space_data, marker_name, reach_speed_segments, time)
        path_distances = traj_utils.calculate_reach_path_distances(Traj_Space_data, marker_name, reach_speed_segments, time)
        v_peaks, v_peak_indices = traj_utils.calculate_v_peaks(Traj_Space_data, marker_name, time, reach_speed_segments)
        
        # Define time windows based on the selected method
        if time_window_method == 1:
            test_windows = reach_speed_segments  # Entire reach segments
        elif time_window_method == 2:
            start_to_peak_segments, peak_to_end_segments = traj_utils.split_segments_at_peak_speed(reach_speed_segments, Traj_Space_data, time, v_peaks, marker_name)  # Split at peak speed
            test_windows = start_to_peak_segments  # Use the first part (start to peak speed)
            # test_windows = peak_to_end_segments  # Use the second part (peak to end)
        elif time_window_method == 3:
            test_windows = [
                (time[max(0, v_peak_index - int(window_size * 200))], time[min(len(time) - 1, v_peak_index + int(window_size * 200))])
                for v_peak_index in v_peak_indices
            ]  # 50 ms before and after peak speed
        else:
            raise ValueError("Invalid time window method selected. Choose 1, 2, or 3.")
        
        # Calculate LDLJ and other metrics for the selected method
        print(f"Processing time windows (Method {time_window_method})")
        LDLJ_values, acc_peaks, jerk_peaks = traj_utils.calculate_ldlj_for_all_reaches(Traj_Space_data, marker_name, time, test_windows)
        
        # Correlation analysis
        '''
            # Hypotheses:
            # 1. Reach duration increases with reach distance.
            # 2. Reach duration increases with path distance.
            # 3. Peak speed increases with reach distance.
            # 4. LDLJ values increase (closer to 0) with a decrease in reach distance, indicating smoother motion for shorter reaches.
        '''
        corr_dur_dist, p_dur_dist = pearsonr(reach_durations, reach_distances) # important
        corr_dur_path, p_dur_path = pearsonr(reach_durations, path_distances) # important
        corr_dist_vpeaks, p_dist_vpeaks = pearsonr(reach_distances, v_peaks) # important
        corr_dist_ldlj, p_dist_ldlj = pearsonr(reach_distances, LDLJ_values) # important
        
        print(f"Method {time_window_method} - Correlation (Duration vs Distance): {corr_dur_dist}, P-value: {p_dur_dist}")
        print(f"Method {time_window_method} - Correlation (Duration vs Path Distance): {corr_dur_path}, P-value: {p_dur_path}")
        print(f"Method {time_window_method} - Correlation (Peak Speed vs Distance): {corr_dist_vpeaks}, P-value: {p_dist_vpeaks}")
        print(f"Method {time_window_method} - Correlation (LDLJ vs Distance): {corr_dist_ldlj}, P-value: {p_dist_ldlj}")


        # traj_utils.plot_x_position_and_speed_with_segments(time, traj_data, Traj_Space_data, marker_name, reach_speed_segments, save_path, file_path) 
        # traj_utils.plot_aligned_segments(time, Traj_Space_data, reach_speed_segments, marker_name, save_path, file_path)
        # traj_utils.plot_aligned_segments_xyz(time, traj_data, reach_speed_segments, marker_name, save_path, file_path)

        # Store results for the current file
        results[file_path] = {
            'parameters': {
            'reach_durations': reach_durations,
            'reach_distances': reach_distances,
            'path_distances': path_distances,
            'v_peaks': v_peaks,
            'LDLJ_values': LDLJ_values,
            'acc_peaks': acc_peaks,
            'jerk_peaks': jerk_peaks,
            'correlations': {
                'duration_distance': (corr_dur_dist, p_dur_dist),
                'duration_path_distance': (corr_dur_path, p_dur_path),
                'distance_peak_speed': (corr_dist_vpeaks, p_dist_vpeaks),
                'distance_ldlj': (corr_dist_ldlj, p_dist_ldlj)
            }
            },
            'test_windows': test_windows,
            'time': time
        }

    return results


file_paths = [file for i, file in enumerate(tBBT_files) if i % 2 == 0]  # Select only files at odd indices
# file_paths = [file for i, file in enumerate(tBBT_files) if i % 2 == 1]  # Select only files at even indices



BoxTrajfile_path = "/Users/yilinwu/Desktop/honours data/Yilin-Honours/Box/Traj/06/13/tBBT01.csv"
marker_name = "RFIN"  # Adjust marker name as needed
# marker_name = "LFIN"  # Adjust marker name as needed

prominence_threshold_speed = 350  # Adjust as needed
speed_threshold = 300  # Adjust as needed
window_size = 0.05  # 50 ms
time_window_method = 1  # Choose 1, 2, or 3
save_path = "/Users/yilinwu/Desktop/honours/Thesis/figure"  # Adjust the save path as needed



import datetime
import matplotlib.pyplot as plt
import os
# def plot_overlayed_reaches_xyz(file_paths, marker_name, save_path):
#     """
#     Plots overlay of RFIN_X, RFIN_Y, and RFIN_Z trajectory data for all files in file_paths as subplots.
#     """
#     fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
#     fig.suptitle(f"Overlayed {marker_name} Trajectories (X, Y, Z) ({', '.join([os.path.basename(fp) for fp in file_paths])})")

#     axes[0].set_title(f"{marker_name}_X Coordinate")
#     axes[0].set_ylabel("X Value")
#     axes[1].set_title(f"{marker_name}_Y Coordinate")
#     axes[1].set_ylabel("Y Value")
#     axes[2].set_title(f"{marker_name}_Z Coordinate")
#     axes[2].set_ylabel("Z Value")
#     axes[2].set_xlabel("Time (s)")

#     for file_path in file_paths:
#         traj_data, _, time = traj_utils.CSV_To_traj_data(file_path)

#         axes[0].plot(time, traj_data[f"{marker_name}_X"], label=os.path.basename(file_path))
#         axes[1].plot(time, traj_data[f"{marker_name}_Y"], label=os.path.basename(file_path))
#         axes[2].plot(time, traj_data[f"{marker_name}_Z"], label=os.path.basename(file_path))

#     for ax in axes:
#         ax.legend(loc="upper right", fontsize=8, frameon=False)
#         ax.grid(True)

#     if save_path:
#         unique_name = f"overlayed_{marker_name}_XYZ_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
#         full_path = os.path.join(save_path, unique_name)
#         plt.savefig(full_path)
#         print(f"Figure saved to {full_path}")

#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.show()

# plot_overlayed_reaches_xyz(file_paths, marker_name, save_path)

results = process_files(file_paths, BoxTrajfile_path, marker_name, prominence_threshold_speed, speed_threshold, window_size, time_window_method, save_path)


# def plot_reaches_xyz(results, marker_name, save_path):
#     """
#     Plots the X, Y, Z coordinates of each reach and overlays all reaches for visualization.
#     """
#     for file_path, data in results.items():
#         test_windows = data['test_windows']
#         time = data['time']
#         traj_data = traj_utils.CSV_To_traj_data(file_path)[0]

#         fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
#         fig.suptitle(f"Overlayed {marker_name} Trajectories (X, Y, Z) for {os.path.basename(file_path)}")

#         axes[0].set_title(f"{marker_name}_X Coordinate")
#         axes[0].set_ylabel("X Value")
#         axes[1].set_title(f"{marker_name}_Y Coordinate")
#         axes[1].set_ylabel("Y Value")
#         axes[2].set_title(f"{marker_name}_Z Coordinate")
#         axes[2].set_ylabel("Z Value")
#         axes[2].set_xlabel("Time (s)")

#         for start, end in test_windows:
#             start_idx = np.searchsorted(time, start)
#             end_idx = np.searchsorted(time, end)

#             axes[0].plot(time[start_idx:end_idx], traj_data[f"{marker_name}_X"][start_idx:end_idx], label=f"Reach {start}-{end}")
#             axes[1].plot(time[start_idx:end_idx], traj_data[f"{marker_name}_Y"][start_idx:end_idx], label=f"Reach {start}-{end}")
#             axes[2].plot(time[start_idx:end_idx], traj_data[f"{marker_name}_Z"][start_idx:end_idx], label=f"Reach {start}-{end}")

#         for ax in axes:
#             ax.legend(loc="upper right", fontsize=8, frameon=False)
#             ax.grid(True)

#         if save_path:
#             unique_name = f"reaches_{marker_name}_XYZ_{os.path.basename(file_path)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
#             full_path = os.path.join(save_path, unique_name)
#             plt.savefig(full_path)
#             print(f"Figure saved to {full_path}")

#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         plt.show()

# plot_reaches_xyz(results, marker_name, save_path)


# def plot_overlayed_reaches_xyz_across_files(results, marker_name, save_path):
#     """
#     Plots overlay of RFIN_X, RFIN_Y, and RFIN_Z trajectory data across all files in results as subplots.
#     """
#     fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
#     fig.suptitle(f"Overlayed {marker_name} Trajectories (X, Y, Z) Across Files")

#     axes[0].set_title(f"{marker_name}_X Coordinate")
#     axes[0].set_ylabel("X Value")
#     axes[1].set_title(f"{marker_name}_Y Coordinate")
#     axes[1].set_ylabel("Y Value")
#     axes[2].set_title(f"{marker_name}_Z Coordinate")
#     axes[2].set_ylabel("Z Value")
#     axes[2].set_xlabel("Time (s)")

#     for file_path, data in results.items():
#         test_windows = data['test_windows']
#         time = data['time']
#         traj_data = traj_utils.CSV_To_traj_data(file_path)[0]

#         for start, end in test_windows:
#             start_idx = np.searchsorted(time, start)
#             end_idx = np.searchsorted(time, end)

#             axes[0].plot(time[start_idx:end_idx], traj_data[f"{marker_name}_X"][start_idx:end_idx], label=os.path.basename(file_path))
#             axes[1].plot(time[start_idx:end_idx], traj_data[f"{marker_name}_Y"][start_idx:end_idx], label=os.path.basename(file_path))
#             axes[2].plot(time[start_idx:end_idx], traj_data[f"{marker_name}_Z"][start_idx:end_idx], label=os.path.basename(file_path))

#     for ax in axes:
#         ax.legend(loc="upper right", fontsize=8, frameon=False)
#         ax.grid(True)

#     if save_path:
#         unique_name = f"overlayed_reaches_{marker_name}_XYZ_across_files_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
#         full_path = os.path.join(save_path, unique_name)
#         plt.savefig(full_path)
#         print(f"Figure saved to {full_path}")

#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.show()

# plot_overlayed_reaches_xyz_across_files(results, marker_name, save_path)


def plot_overlayed_reaches_xyz_aligned(results, marker_name, save_path):
    """
    Plots overlay of RFIN_X, RFIN_Y, and RFIN_Z trajectory data across all files in results as subplots,
    aligning all reaches to the same start time for comparison.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"Aligned Overlayed {marker_name} Trajectories (X, Y, Z) Across Files")

    axes[0].set_title(f"{marker_name}_X Coordinate")
    axes[0].set_ylabel("X Value")
    axes[1].set_title(f"{marker_name}_Y Coordinate")
    axes[1].set_ylabel("Y Value")
    axes[2].set_title(f"{marker_name}_Z Coordinate")
    axes[2].set_ylabel("Z Value")
    axes[2].set_xlabel("Aligned Time (s)")

    for file_path, data in results.items():
        test_windows = data['test_windows']
        time = data['time']
        traj_data = traj_utils.CSV_To_traj_data(file_path)[0]

        for start, end in test_windows:
            start_idx = np.searchsorted(time, start)
            end_idx = np.searchsorted(time, end)

            aligned_time = time[start_idx:end_idx] - time[start_idx]  # Align time to start at 0
            axes[0].plot(aligned_time, traj_data[f"{marker_name}_X"][start_idx:end_idx], label=os.path.basename(file_path))
            axes[1].plot(aligned_time, traj_data[f"{marker_name}_Y"][start_idx:end_idx], label=os.path.basename(file_path))
            axes[2].plot(aligned_time, traj_data[f"{marker_name}_Z"][start_idx:end_idx], label=os.path.basename(file_path))

    for ax in axes:
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.grid(True)

    if save_path:
        unique_name = f"aligned_overlayed_reaches_{marker_name}_XYZ_across_files_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        full_path = os.path.join(save_path, unique_name)
        plt.savefig(full_path)
        print(f"Figure saved to {full_path}")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

plot_overlayed_reaches_xyz_aligned(results, marker_name, save_path)
