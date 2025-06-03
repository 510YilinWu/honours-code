import traj_utils
from scipy.stats import pearsonr


file_path = "/Users/yilinwu/Desktop/YW_tBBT01.csv"
# file_path = "/Users/yilinwu/Desktop/honours data/05/13/YW_tBBT01_Trajectories.csv"


save_path="/Users/yilinwu/Desktop/honours/Thesis/figure"

traj_data, Frame, time = traj_utils.CSV_To_traj_data(file_path)  
Traj_Space_data = traj_utils.Traj_Space_data(traj_data)


marker_name="RFIN" # C7, T10, CLAV, STRN, LSHO, LUPA, LUPB, LUPC, LELB, LMEP, LWRA, LWRB, LFRA, LFIN, RSHO, RUPA, RUPB, RUPC, RELB, RMEP, RWRA, RWRB, RFRA, RFIN

prominence_threshold_speed = 350  # Adjust as needed

speed_minima, speed_peaks = traj_utils.find_local_minima_peaks(Traj_Space_data[marker_name][1], prominence_threshold_speed)

speed_threshold = 400  # Adjust as needed
speed_segments = traj_utils.find_speed_segments(marker_name, Traj_Space_data, time, speed_threshold, speed_peaks)
# traj_utils.plot_speed_position_segmentsByspeed_space(marker_name, time, Traj_Space_data, speed_segments, speed_minima, speed_threshold, prominence_threshold_speed, save_path)

# traj_utils.plot_speed_x_segmentsByspeed_space(marker_name, time, Traj_Space_data, traj_data, speed_segments, speed_minima, speed_peaks, speed_threshold, prominence_threshold_speed, save_path)
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

# Print all Pearson correlation coefficients and P-values
# print("LDLJ Correlation Coefficient:", LDLJ_correlation, "P-value:", LDLJ_p_value)
# print("Reach Distance Correlation Coefficient:", reach_distance_correlation, "P-value:", reach_distance_p_value)
# print("Path Distance Correlation Coefficient:", path_distance_correlation, "P-value:", path_distance_p_value)
# print("Velocity Peaks Correlation Coefficient:", v_peaks_correlation, "P-value:", v_peaks_p_value)
# print("Acceleration Peaks Correlation Coefficient:", acc_peaks_correlation, "P-value:", acc_peaks_p_value)

# traj_utils.plot_correlation_matrix(LDLJ_values, reach_distances, path_distances, v_peaks, acc_peaks, reach_durations, speed_threshold, save_path)


# Split reach_speed_segments into two parts: start to peak speed, and peak to end
start_to_peak_segments, peak_to_end_segments = traj_utils.split_segments_at_peak_speed(reach_speed_segments, Traj_Space_data, time, v_peaks, marker_name)

traj_utils.plot_split_segments_speed(time, Traj_Space_data, marker_name, start_to_peak_segments, peak_to_end_segments, save_path)


# Apply all functions to the start-to-peak segments
LDLJ_values_start_to_peak, v_peaks_start_to_peak, acc_peaks_start_to_peak, reach_durations_start_to_peak = traj_utils.calculate_ldlj_for_all_reaches(
    Traj_Space_data, marker_name, time, start_to_peak_segments)
reach_distances_start_to_peak = traj_utils.calculate_reach_distances(Traj_Space_data, marker_name, start_to_peak_segments, time)
path_distances_start_to_peak = traj_utils.calculate_reach_path_distances(Traj_Space_data, marker_name, start_to_peak_segments, time)

LDLJ_correlation_start_to_peak, LDLJ_p_value_start_to_peak = pearsonr(LDLJ_values_start_to_peak, reach_durations_start_to_peak)
reach_distance_correlation_start_to_peak, reach_distance_p_value_start_to_peak = pearsonr(reach_distances_start_to_peak, reach_durations_start_to_peak)
path_distance_correlation_start_to_peak, path_distance_p_value_start_to_peak = pearsonr(path_distances_start_to_peak, reach_durations_start_to_peak)
v_peaks_correlation_start_to_peak, v_peaks_p_value_start_to_peak = pearsonr(v_peaks_start_to_peak, reach_durations_start_to_peak)
acc_peaks_correlation_start_to_peak, acc_peaks_p_value_start_to_peak = pearsonr(acc_peaks_start_to_peak, reach_durations_start_to_peak)

# Apply all functions to the peak-to-end segments
LDLJ_values_peak_to_end, v_peaks_peak_to_end, acc_peaks_peak_to_end, reach_durations_peak_to_end = traj_utils.calculate_ldlj_for_all_reaches(
    Traj_Space_data, marker_name, time, peak_to_end_segments)
reach_distances_peak_to_end = traj_utils.calculate_reach_distances(Traj_Space_data, marker_name, peak_to_end_segments, time)
path_distances_peak_to_end = traj_utils.calculate_reach_path_distances(Traj_Space_data, marker_name, peak_to_end_segments, time)

LDLJ_correlation_peak_to_end, LDLJ_p_value_peak_to_end = pearsonr(LDLJ_values_peak_to_end, reach_durations_peak_to_end)
reach_distance_correlation_peak_to_end, reach_distance_p_value_peak_to_end = pearsonr(reach_distances_peak_to_end, reach_durations_peak_to_end)
path_distance_correlation_peak_to_end, path_distance_p_value_peak_to_end = pearsonr(path_distances_peak_to_end, reach_durations_peak_to_end)
v_peaks_correlation_peak_to_end, v_peaks_p_value_peak_to_end = pearsonr(v_peaks_peak_to_end, reach_durations_peak_to_end)
acc_peaks_correlation_peak_to_end, acc_peaks_p_value_peak_to_end = pearsonr(acc_peaks_peak_to_end, reach_durations_peak_to_end)

# Print results for start-to-peak segments
print("Start-to-Peak Segments:")
print("LDLJ Correlation Coefficient:", LDLJ_correlation_start_to_peak, "P-value:", LDLJ_p_value_start_to_peak)
print("Reach Distance Correlation Coefficient:", reach_distance_correlation_start_to_peak, "P-value:", reach_distance_p_value_start_to_peak)
print("Path Distance Correlation Coefficient:", path_distance_correlation_start_to_peak, "P-value:", path_distance_p_value_start_to_peak)
print("Velocity Peaks Correlation Coefficient:", v_peaks_correlation_start_to_peak, "P-value:", v_peaks_p_value_start_to_peak)
print("Acceleration Peaks Correlation Coefficient:", acc_peaks_correlation_start_to_peak, "P-value:", acc_peaks_p_value_start_to_peak)

# Print results for peak-to-end segments
print("Peak-to-End Segments:")
print("LDLJ Correlation Coefficient:", LDLJ_correlation_peak_to_end, "P-value:", LDLJ_p_value_peak_to_end)
print("Reach Distance Correlation Coefficient:", reach_distance_correlation_peak_to_end, "P-value:", reach_distance_p_value_peak_to_end)
print("Path Distance Correlation Coefficient:", path_distance_correlation_peak_to_end, "P-value:", path_distance_p_value_peak_to_end)
print("Velocity Peaks Correlation Coefficient:", v_peaks_correlation_peak_to_end, "P-value:", v_peaks_p_value_peak_to_end)
print("Acceleration Peaks Correlation Coefficient:", acc_peaks_correlation_peak_to_end, "P-value:", acc_peaks_p_value_peak_to_end)


# # Plot correlation matrix for start-to-peak segments
# traj_utils.plot_correlation_matrix(
#     LDLJ_values_start_to_peak, 
#     reach_distances_start_to_peak, 
#     path_distances_start_to_peak, 
#     v_peaks_start_to_peak, 
#     acc_peaks_start_to_peak, 
#     reach_durations_start_to_peak, 
#     speed_threshold, 
#     save_path
# )

# # Plot correlation matrix for peak-to-end segments
# traj_utils.plot_correlation_matrix(
#     LDLJ_values_peak_to_end, 
#     reach_distances_peak_to_end, 
#     path_distances_peak_to_end, 
#     v_peaks_peak_to_end, 
#     acc_peaks_peak_to_end, 
#     reach_durations_peak_to_end, 
#     speed_threshold, 
#     save_path
# )


traj_utils.plot_combined_correlations(reach_durations, LDLJ_values, LDLJ_correlation, LDLJ_p_value, 
                                reach_distances, reach_distance_correlation, reach_distance_p_value, 
                                path_distances, path_distance_correlation, path_distance_p_value, 
                                v_peaks, v_peaks_correlation, v_peaks_p_value, 
                                acc_peaks, acc_peaks_correlation, acc_peaks_p_value, 
                                save_path)


# # Plot combined correlations for start-to-peak segments
# traj_utils.plot_combined_correlations(
#     reach_durations_start_to_peak, LDLJ_values_start_to_peak, LDLJ_correlation_start_to_peak, LDLJ_p_value_start_to_peak, 
#     reach_distances_start_to_peak, reach_distance_correlation_start_to_peak, reach_distance_p_value_start_to_peak, 
#     path_distances_start_to_peak, path_distance_correlation_start_to_peak, path_distance_p_value_start_to_peak, 
#     v_peaks_start_to_peak, v_peaks_correlation_start_to_peak, v_peaks_p_value_start_to_peak, 
#     acc_peaks_start_to_peak, acc_peaks_correlation_start_to_peak, acc_peaks_p_value_start_to_peak, 
#     save_path
# )

# # Plot combined correlations for peak-to-end segments
# traj_utils.plot_combined_correlations(
#     reach_durations_peak_to_end, LDLJ_values_peak_to_end, LDLJ_correlation_peak_to_end, LDLJ_p_value_peak_to_end, 
#     reach_distances_peak_to_end, reach_distance_correlation_peak_to_end, reach_distance_p_value_peak_to_end, 
#     path_distances_peak_to_end, path_distance_correlation_peak_to_end, path_distance_p_value_peak_to_end, 
#     v_peaks_peak_to_end, v_peaks_correlation_peak_to_end, v_peaks_p_value_peak_to_end, 
#     acc_peaks_peak_to_end, acc_peaks_correlation_peak_to_end, acc_peaks_p_value_peak_to_end, 
#     save_path
# )