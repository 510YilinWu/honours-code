import os
import traj_utils
from scipy.stats import pearsonr

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


def process_files(file_paths, Box_Traj_file, marker_name, prominence_threshold_speed, speed_threshold, window_size, time_window_method, save_path):
    """
    Processes multiple trajectory files to analyze motion data and calculate metrics.
    """
    results = {}
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        
        # Create subfolder within save_path using the file name
        file_name = os.path.basename(file_path).split('.')[0]
        file_save_path = os.path.join(save_path, file_name)
        os.makedirs(file_save_path, exist_ok=True)

        # Extract trajectory data
        traj_data, Frame, time = traj_utils.CSV_To_traj_data(file_path)
        Traj_Space_data = traj_utils.Traj_Space_data(traj_data)

        # Find local minima and peaks in speed data
        speed_minima, speed_peaks, speed_minima_indices, speed_peaks_indices= traj_utils.find_local_minima_peaks(Traj_Space_data[marker_name][1], prominence_threshold_speed)

        print(f"Number of speed minima found: {len(speed_minima)}")
        print(f"Number of speed peaks found: {len(speed_peaks)}")

        traj_utils.plot_x_speed_one_extrema_space(time, Traj_Space_data, traj_data, speed_minima, speed_peaks, marker_name, file_save_path)

        # Define segments based on speed thresholds
        speed_segments = traj_utils.find_speed_segments(marker_name, Traj_Space_data, time, speed_threshold, speed_peaks)

        # Classify the speed segments into reach and return segments
        # Note: Adjust the rfin_x_range and lfin_x_threshold based on specific requirements
        BoxRange = traj_utils.calculate_rangesByBoxTraj(Box_Traj_file)
        reach_speed_segments, return_speed_segments = traj_utils.classify_speed_segments(speed_segments, traj_data, marker_name, time, BoxRange)

        # print(f"Number of reach segments found: {len(reach_speed_segments)}")
        if len(reach_speed_segments) != 16:
            print(f"File {file_path} does not have 16 reach segments.")

        traj_utils.plot_x_position_and_speed_with_segments(time, traj_data, Traj_Space_data, marker_name, reach_speed_segments, file_save_path, file_path) 

    return 


















































def x_process_files(file_paths, Box_Traj_file, marker_name, prominence_threshold_speed, speed_threshold, window_size, time_window_method, save_path):
    """
    Processes multiple trajectory files to analyze motion data and calculate metrics.
    """
    results = {}
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        
        # Create subfolder within save_path using the file name
        file_name = os.path.basename(file_path).split('.')[0]
        file_save_path = os.path.join(save_path, file_name)
        os.makedirs(file_save_path, exist_ok=True)

        # Extract trajectory data
        traj_data, Frame, time = traj_utils.CSV_To_traj_data(file_path)
        Traj_Space_data = traj_utils.Traj_Space_data(traj_data)

        # Find local minima and peaks in speed data
        speed_minima, speed_peaks = traj_utils.find_local_minima_peaks(Traj_Space_data[marker_name][1], prominence_threshold_speed)

        # traj_utils.plot_x_speed_one_extrema_space(time, Traj_Space_data, traj_data, speed_minima, speed_peaks, marker_name, file_save_path)

        # Define segments based on speed thresholds
        speed_segments = traj_utils.find_speed_segments(marker_name, Traj_Space_data, time, speed_threshold, speed_peaks)

        # Classify the speed segments into reach and return segments
        # Note: Adjust the rfin_x_range and lfin_x_threshold based on specific requirements
        BoxRange = traj_utils.calculate_rangesByBoxTraj(Box_Traj_file)
        reach_speed_segments, return_speed_segments = traj_utils.classify_speed_segments(speed_segments, traj_data, marker_name, time, BoxRange)

        # print(f"Number of reach segments found: {len(reach_speed_segments)}")
        if len(reach_speed_segments) != 16:
            print(f"File {file_path} does not have 16 reach segments.")

        # traj_utils.plot_x_position_and_speed_with_segments(time, traj_data, Traj_Space_data, marker_name, reach_speed_segments, file_save_path, file_path) 

        # # Calculate reach durations and distances
        # reach_durations = traj_utils.calculate_reach_durations(reach_speed_segments)
        # reach_distances = traj_utils.calculate_reach_distances(Traj_Space_data, marker_name, reach_speed_segments, time)
        # path_distances = traj_utils.calculate_reach_path_distances(Traj_Space_data, marker_name, reach_speed_segments, time)
        # v_peaks, v_peak_indices = traj_utils.calculate_v_peaks(Traj_Space_data, marker_name, time, reach_speed_segments)
        
        # # Define time windows based on the selected method
        # if time_window_method == 1:
        #     test_windows = reach_speed_segments  # Entire reach segments
        # elif time_window_method == 2:
        #     start_to_peak_segments, peak_to_end_segments = traj_utils.split_segments_at_peak_speed(reach_speed_segments, Traj_Space_data, time, v_peaks, marker_name)  # Split at peak speed
        #     test_windows = start_to_peak_segments  # Use the first part (start to peak speed)
        #     # test_windows = peak_to_end_segments  # Use the second part (peak to end)
        # elif time_window_method == 3:
        #     test_windows = [
        #         (time[max(0, v_peak_index - int(window_size * 200))], time[min(len(time) - 1, v_peak_index + int(window_size * 200))])
        #         for v_peak_index in v_peak_indices
        #     ]  # 50 ms before and after peak speed
        # else:
        #     raise ValueError("Invalid time window method selected. Choose 1, 2, or 3.")
        
        # # Calculate LDLJ and other metrics for the selected method
        # print(f"Processing time windows (Method {time_window_method})")
        # LDLJ_values, acc_peaks, jerk_peaks = traj_utils.calculate_ldlj_for_all_reaches(Traj_Space_data, marker_name, time, test_windows)
        
        # # Correlation analysis
        # '''
        #     # Hypotheses:
        #     # 1. Reach duration increases with reach distance.
        #     # 2. Reach duration increases with path distance.
        #     # 3. Peak speed increases with reach distance.
        #     # 4. LDLJ values increase (closer to 0) with a decrease in reach distance, indicating smoother motion for shorter reaches.
        # '''
        # corr_dur_dist, p_dur_dist = pearsonr(reach_durations, reach_distances) # important
        # corr_dur_path, p_dur_path = pearsonr(reach_durations, path_distances) # important
        # corr_dist_vpeaks, p_dist_vpeaks = pearsonr(reach_distances, v_peaks) # important
        # corr_dist_ldlj, p_dist_ldlj = pearsonr(reach_distances, LDLJ_values) # important

        # # Store results for the current file
        # results[file_path] = {
        #     'parameters': {
        #     'reach_durations': reach_durations,
        #     'reach_distances': reach_distances,
        #     'path_distances': path_distances,
        #     'v_peaks': v_peaks,
        #     'LDLJ_values': LDLJ_values,
        #     'acc_peaks': acc_peaks,
        #     'jerk_peaks': jerk_peaks,
        #     'correlations': {
        #         'duration_distance': (corr_dur_dist, p_dur_dist),
        #         'duration_path_distance': (corr_dur_path, p_dur_path),
        #         'distance_peak_speed': (corr_dist_vpeaks, p_dist_vpeaks),
        #         'distance_ldlj': (corr_dist_ldlj, p_dist_ldlj)
        #     }
        #     },
        #     'test_windows': test_windows,
        #     'time': time,
        #     'traj_data': traj_data,
        #     'Traj_Space_data': Traj_Space_data,
        #     'reach_speed_segments': reach_speed_segments,
        #     'speed_segments': speed_segments,
        #     'speed_minima': speed_minima,
        #     'speed_peaks': speed_peaks
        # }

        # # Include start_to_peak_segments and peak_to_end_segments only if method 2 is selected
        # if time_window_method == 2:
        #     results[file_path]['start_to_peak_segments'] = start_to_peak_segments
        #     results[file_path]['peak_to_end_segments'] = peak_to_end_segments

    return 

def plot_files(results, file_paths, marker_name, prominence_threshold_speed, speed_threshold, save_path):

    for file_path in file_paths:
        print(f"Processing file: {file_path}")

        # Create subfolder within save_path using the file name
        file_name = os.path.basename(file_path).split('.')[0]
        file_save_path = os.path.join(save_path, file_name)
        os.makedirs(file_save_path, exist_ok=True)

        ''' Accessing variables from results '''
        data = results[file_path]
        parameters = data['parameters']
        reach_durations = parameters['reach_durations']
        reach_distances = parameters['reach_distances']
        path_distances = parameters['path_distances']
        v_peaks = parameters['v_peaks']
        LDLJ_values = parameters['LDLJ_values']
        acc_peaks = parameters['acc_peaks']
        jerk_peaks = parameters['jerk_peaks']
        correlations = parameters['correlations']
        duration_distance_corr = correlations['duration_distance']
        duration_path_distance_corr = correlations['duration_path_distance']
        distance_peak_speed_corr = correlations['distance_peak_speed']
        distance_ldlj_corr = correlations['distance_ldlj']

        test_windows = data['test_windows']
        time = data['time']
        traj_data = data['traj_data']
        Traj_Space_data = data['Traj_Space_data']
        speed_minima = data['speed_minima']
        speed_peaks = data['speed_peaks']
        reach_speed_segments = data['reach_speed_segments']
        speed_segments = data['speed_segments']
        if 'start_to_peak_segments' in data:
            start_to_peak_segments = data['start_to_peak_segments']
        if 'peak_to_end_segments' in data:
            peak_to_end_segments = data['peak_to_end_segments']

        # ''' Plotting '''
        # traj_utils.plot_marker_trajectory_components(time, traj_data, Traj_Space_data, marker_name, file_save_path)
        # traj_utils.plot_single_marker_space(Traj_Space_data, time, marker_name, file_save_path)
        # traj_utils.plot_pos_speed_one_extrema_space(time, Traj_Space_data, speed_minima, speed_peaks, marker_name, file_save_path)
        # traj_utils.plot_x_speed_one_extrema_space(time, Traj_Space_data, traj_data, speed_minima, speed_peaks, marker_name, file_save_path)
        # traj_utils.plot_speed_x_segmentsByspeed_space(marker_name, time, Traj_Space_data, traj_data, speed_segments, speed_minima, speed_peaks, speed_threshold, prominence_threshold_speed, file_save_path)
        # traj_utils.plot_speed_position_segmentsByspeed_space(marker_name, time, Traj_Space_data, speed_segments, speed_minima, speed_threshold, prominence_threshold_speed, file_save_path)
        # traj_utils.plot_marker_3d_trajectory_traj(traj_data, marker_name, 0, 1001, file_save_path) # start_frame, end_frame
        # traj_utils.plot_marker_radial_components_space(time, traj_data, marker_name, file_save_path)
        # traj_utils.plot_reach_acceleration_with_ldlj_normalised(time, Traj_Space_data, marker_name, reach_speed_segments, LDLJ_values, file_save_path)
        # if 'start_to_peak_segments' in data and 'peak_to_end_segments' in data:
        #         traj_utils.plot_split_segments_speed(time, Traj_Space_data, marker_name, start_to_peak_segments, peak_to_end_segments, file_save_path)

        ''' important plots '''
        # traj_utils.plot_x_position_and_speed_with_segments(time, traj_data, Traj_Space_data, marker_name, reach_speed_segments, file_save_path, file_path) 
        # traj_utils.plot_aligned_segments(time, Traj_Space_data, reach_speed_segments, marker_name, file_save_path, file_path)
        # traj_utils.plot_aligned_segments_xyz(time, traj_data, reach_speed_segments, marker_name, file_save_path, file_path)
        # traj_utils.plot_reach_speed_and_jerk(time, Traj_Space_data, marker_name, reach_speed_segments, LDLJ_values, file_save_path, file_path)

        # ''' Plotting Time Windows Specific Metrics '''
        # traj_utils.rank_and_visualize_ldlj(reach_speed_segments, LDLJ_values, file_save_path, file_path)
        # traj_utils.plot_correlation_matrix(LDLJ_values, reach_distances, path_distances, v_peaks, acc_peaks, reach_durations, jerk_peaks, speed_threshold, file_save_path, file_path)

        # ''' Plotting correlations '''
        # traj_utils.plot_combined_correlations(reach_durations, reach_distances, path_distances, v_peaks, LDLJ_values,
        #             duration_distance_corr[0], duration_distance_corr[1], 
        #             duration_path_distance_corr[0], duration_path_distance_corr[1],
        #             distance_peak_speed_corr[0], distance_peak_speed_corr[1], 
        #             distance_ldlj_corr[0], distance_ldlj_corr[1],
        #             file_save_path, file_path)