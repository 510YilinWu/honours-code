import utils
date = "06/26/2"
time_window_method = 1  # Choose 1, 2, or 3
window_size = 0.05  # 50 ms
prominence_threshold_speed = 350  # Adjust as needed
speed_threshold = 300  # Adjust as needed


Traj_folder = "/Volumes/MNHS-MoCap/Yilin-Honours/Subject/Traj/2025/"
Box_Traj_folder = "/Volumes/MNHS-MoCap/Yilin-Honours/Box/Traj/2025/"
save_path = "/Users/yilinwu/Desktop/honours/Thesis/figure"  # Adjust the save path as needed

tBBT_files = utils.find_bbt_files(Traj_folder, date, file_type = "tBBT")
print(f"tBBT files: {tBBT_files}\nNumber of tBBT files: {len(tBBT_files)}")
if len(tBBT_files) != 64:
    raise ValueError(f"Expected 64 tBBT files, but found {len(tBBT_files)}")

Box_Traj_file = utils.find_bbt_files(Box_Traj_folder, date, file_type = "BOX_Cali")
print("Box Trajectory files:", Box_Traj_file)
print("Number of Box Trajectory files:", len(Box_Traj_file))
if len(Box_Traj_file) != 1:
    raise ValueError(f"Expected 1 Box Trajectory file, but found {len(Box_Traj_file)}")

file_paths = [file for i, file in enumerate(tBBT_files) if i % 2 == 0]  # Select only files at odd indices
marker_name = "RFIN"  # Adjust marker name as needed

# file_paths = [file for i, file in enumerate(tBBT_files) if i % 2 == 1]  # Select only files at even indices
# marker_name = "LFIN"  # Adjust marker name as needed

print(len(file_paths))

results = process_files(file_paths, Box_Traj_file, marker_name, prominence_threshold_speed, speed_threshold, window_size, time_window_method)



def process_files(file_paths, Box_Traj_file, marker_name, prominence_threshold_speed, speed_threshold, window_size, time_window_method):
    """
    Processes multiple trajectory files to analyze motion data and calculate metrics.
    """
    results = {}
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        
        # Extract trajectory data
        traj_data, Frame, time = traj_utils.CSV_To_traj_data(file_path)
        Traj_Space_data = traj_utils.Traj_Space_data(traj_data)

        # Find local minima and peaks in speed data
        speed_minima, speed_peaks = traj_utils.find_local_minima_peaks(Traj_Space_data[marker_name][1], prominence_threshold_speed)

        # Define segments based on speed thresholds
        speed_segments = traj_utils.find_speed_segments(marker_name, Traj_Space_data, time, speed_threshold, speed_peaks)

                
        # Classify the speed segments into reach and return segments
        # Note: Adjust the rfin_x_range and lfin_x_threshold based on specific requirements
        RightBoxRange, LeftBoxRange = traj_utils.calculate_rangesByBoxTraj(Box_Traj_file)
        reach_speed_segments, return_speed_segments = traj_utils.classify_speed_segments(speed_segments, traj_data, marker_name, time, RightBoxRange, LeftBoxRange)

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
            'time': time,
            'traj_data': traj_data,
            'Traj_Space_data': Traj_Space_data,
            'reach_speed_segments': reach_speed_segments,
            'speed_segments': speed_segments,
            'speed_minima': speed_minima,
            'speed_peaks': speed_peaks
        }

        # Include start_to_peak_segments and peak_to_end_segments only if method 2 is selected
        if time_window_method == 2:
            results[file_path]['start_to_peak_segments'] = start_to_peak_segments
            results[file_path]['peak_to_end_segments'] = peak_to_end_segments

    return results