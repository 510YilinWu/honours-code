import traj_utils
import numpy as np
from scipy.stats import pearsonr

# file_path = "/Users/yilinwu/Desktop/honours data/05/13/YW_tBBT01_Trajectories.csv"
file_path = "/Users/yilinwu/Desktop/YW_tBBT01.csv"
traj_data, Frame, time = traj_utils.CSV_To_traj_data(file_path)  
Traj_Space_data = traj_utils.Traj_Space_data(traj_data)

marker_name="RFIN" # C7, T10, CLAV, STRN, LSHO, LUPA, LUPB, LUPC, LELB, LMEP, LWRA, LWRB, LFRA, LFIN, RSHO, RUPA, RUPB, RUPC, RELB, RMEP, RWRA, RWRB, RFRA, RFIN
prominence_threshold_position = 30  # Adjust as needed
prominence_threshold_speed = 350  # Adjust as needed
save_path="/Users/yilinwu/Desktop/honours/Thesis/figure"

# CSV_To_traj_data.find_local_minima_peaks(data, prominence_threshold)
# position_minima, position_peaks = traj_utils.find_local_minima_peaks(Traj_Space_data[marker_name][0], prominence_threshold_position)
speed_minima, speed_peaks = traj_utils.find_local_minima_peaks(Traj_Space_data[marker_name][1], prominence_threshold_speed)

# # Thresholds for movement start and end
# z_range, speed_range, accel_range,x_threshold = (830, 860), (0, 150), (0, 3500),200
# # Extracting the start and end indices of the movement
# tStart_indices, tEnd_indices=traj_utils.extract_marker_start_end(marker_name, traj_data, Traj_Space_data, time, x_threshold, z_range, speed_range, accel_range)
# """
# tStart_representatives/tEnd_representatives: list[float]
#     A list of median of each cluster of tStart_indices/tEnd_indices for the start/end of movement clusters. 
#     where clusters are formed based on significant differences in consecutive time indices. top 16 indices are used to form clusters.
#     tStart_indices/tEnd_indices are caculated based on the threshold of RFIN_X, RFIN_Z, RFIN_Speed, and RFIN_Acceleration.
# """
# tStart_representatives, tEnd_representatives=traj_utils.cluster_and_find_representatives(tStart_indices, tEnd_indices, top_n=15)

# Define the speed threshold
speed_threshold = 500  # Adjust as needed
speed_segments = traj_utils.find_speed_segments(marker_name, Traj_Space_data, time, speed_threshold, speed_peaks)
reach_speed_segments, return_speed_segments= traj_utils.classify_speed_segments(speed_segments, traj_data, marker_name, time)
LDLJ_values, v_peaks, acc_peaks, reach_durations = traj_utils.calculate_ldlj_for_all_reaches(Traj_Space_data, marker_name, time, reach_speed_segments)


# Calculate the reach position change (reach distance) for each reach segment
reach_distances = traj_utils.calculate_reach_distances(Traj_Space_data, marker_name, reach_speed_segments, time)
path_distances = traj_utils.calculate_reach_path_distances(Traj_Space_data, marker_name, reach_speed_segments, time)

# Calculates the Pearson correlation coefficient and P-value between LDLJ_values and reach_durations
LDLJ_correlation, LDLJ_p_value = pearsonr(LDLJ_values, reach_durations)

# Calculates the Pearson correlation coefficient and P-value between reach_distances and reach_durations
reach_distance_correlation, reach_distance_p_value = pearsonr(reach_distances, reach_durations)

# Calculates the Pearson correlation coefficient and P-value between path_distances and reach_durations
path_distance_correlation, path_distance_p_value = pearsonr(path_distances, reach_durations)

# Calculates the Pearson correlation coefficient and P-value between v_peaks and reach_durations
v_peaks_correlation, v_peaks_p_value = pearsonr(v_peaks, reach_durations)

# Calculates the Pearson correlation coefficient and P-value between acc_peaks and reach_durations
acc_peaks_correlation, acc_peaks_p_value = pearsonr(acc_peaks, reach_durations)





# # Print all Pearson correlation coefficients and P-values
# print("LDLJ Correlation Coefficient:", LDLJ_correlation, "P-value:", LDLJ_p_value)
# print("Reach Distance Correlation Coefficient:", reach_distance_correlation, "P-value:", reach_distance_p_value)
# print("path Distance Correlation Coefficient:", path_distance_correlation, "P-value:", path_distance_p_value)
# print("Velocity Peaks Correlation Coefficient:", v_peaks_correlation, "P-value:", v_peaks_p_value)
# print("Acceleration Peaks Correlation Coefficient:", acc_peaks_correlation, "P-value:", acc_peaks_p_value)


'''
Plotting
'''

# '''
# Plotting the trajectory and space data
# '''
# traj_utils.plot_marker_trajectory_components(time, traj_data, Traj_Space_data, marker_name, save_path)
# traj_utils.plot_marker_with_start_end_representatives(time, traj_data, Traj_Space_data, marker_name, tStart_indices, tEnd_indices, tStart_representatives, tEnd_representatives, save_path, x_threshold, z_range, speed_range, accel_range)
# traj_utils.plot_x_position_and_speed_with_segments(time, traj_data, Traj_Space_data, marker_name, reach_speed_segments, save_path)

# '''
# Plotting the trajectory data
# '''
# traj_utils.plot_single_marker_traj(traj_data, time, marker_name,save_path)
# traj_utils.plot_marker_xyz_with_peaks_troughs_traj(traj_data, time, marker_name, save_path, prominence_threshold_position)
# traj_utils.plot_marker_3d_trajectory_traj(traj_data, marker_name, 0, 1001, save_path) # start_frame, end_frame
# traj_utils.plot_marker_radial_components_space(time, traj_data, marker_name, save_path)

# '''
# Plotting the space data
# '''
# traj_utils.plot_single_marker_space(Traj_Space_data, time, marker_name, save_path)
traj_utils.plot_pos_speed_one_extrema_space(time, Traj_Space_data, speed_minima, speed_peaks, marker_name, save_path)
# traj_utils.plot_pos_speed_one_extrema_space(time, Traj_Space_data, position_minima, position_peaks, marker_name, save_path)
# traj_utils.plot_pos_speed_two_extrema_space(time, Traj_Space_data, speed_minima, speed_peaks, position_minima, position_peaks, marker_name, save_path)
traj_utils.plot_speed_position_segmentsByspeed_space(marker_name, time, Traj_Space_data, speed_segments, speed_threshold, prominence_threshold_speed, save_path)
# traj_utils.plot_aligned_segments(time, Traj_Space_data, reach_speed_segments, marker_name, save_path)
# traj_utils.plot_aligned_segments_xyz(time, traj_data, reach_speed_segments, marker_name, save_path)
# traj_utils.plot_reach_acceleration_with_ldlj(time, Traj_Space_data, marker_name, reach_speed_segments, LDLJ_values, save_path)
# traj_utils.plot_reach_acceleration_with_ldlj_normalised(time, Traj_Space_data, marker_name, reach_speed_segments, LDLJ_values, save_path)
# traj_utils.plot_reach_speed_and_jerk(time, Traj_Space_data, marker_name, reach_speed_segments, LDLJ_values, save_path)

# '''
# statistics
# '''
# traj_utils.plot_reach_duration_vs_ldlj(reach_durations, LDLJ_values, LDLJ_correlation, LDLJ_p_value, save_path)
# traj_utils.plot_reach_distance_vs_duration(reach_distances, reach_durations, reach_distance_correlation, reach_distance_p_value, save_path)
# traj_utils.plot_path_distance_vs_duration(path_distances, reach_durations, path_distance_correlation, path_distance_p_value, save_path)
# traj_utils.plot_v_peak_vs_duration(v_peaks, reach_durations, v_peaks_correlation, v_peaks_p_value, save_path)
# traj_utils.plot_acc_peak_vs_duration(acc_peaks, reach_durations, acc_peaks_correlation, acc_peaks_p_value, save_path)
# traj_utils.plot_correlation_matrix(LDLJ_values, reach_distances, path_distances, v_peaks, acc_peaks, reach_durations, save_path)


# traj_utils.plot_speed_x_segmentsByspeed_space(marker_name, time, Traj_Space_data, traj_data, speed_segments, speed_minima, speed_peaks, speed_threshold, prominence_threshold_speed, save_path)