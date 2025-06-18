file_paths = [
    "/Users/yilinwu/Desktop/honours data/Filter_10Hz/Trajectories/05/13/YW_tBBT01.csv",  # right hand
    "/Users/yilinwu/Desktop/honours data/Filter_10Hz/Trajectories/05/13/YW_tBBT02.csv",  # left hand
    "/Users/yilinwu/Desktop/honours data/Filter_10Hz/Trajectories/05/13/YW_tBBT03.csv",  # right hand
    "/Users/yilinwu/Desktop/honours data/Filter_10Hz/Trajectories/05/13/YW_tBBT04.csv"   # left hand
]


"""
CSV_To_traj_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
    Converts a CSV file containing trajectory data into numpy arrays for trajectory data, frame, and time.

Traj_Space_data(traj_data: np.ndarray) -> np.ndarray
    Processes trajectory data to compute spatial information for the given trajectory.
"""
import traj_utils
from scipy.stats import pearsonr
save_path="/Users/yilinwu/Desktop/honours/Thesis/figure"



file_path = file_paths[2]  # Change this to the desired file path
traj_data, Frame, time = traj_utils.CSV_To_traj_data(file_path)  
Traj_Space_data = traj_utils.Traj_Space_data(traj_data)


# Adjust as needed
marker_name="RFIN" # C7, T10, CLAV, STRN, LSHO, LUPA, LUPB, LUPC, LELB, LMEP, LWRA, LWRB, LFRA, LFIN, RSHO, RUPA, RUPB, RUPC, RELB, RMEP, RWRA, RWRB, RFRA, RFIN
# marker_name="LFIN" # C7, T10, CLAV, STRN, LSHO, LUPA, LUPB, LUPC, LELB, LMEP, LWRA, LWRB, LFRA, LFIN, RSHO, RUPA, RUPB, RUPC, RELB, RMEP, RWRA, RWRB, RFRA, RFIN

prominence_threshold_speed = 350  
speed_threshold = 400


# Find local minima and peaks in speed data for a specific marker. 
speed_minima, speed_peaks = traj_utils.find_local_minima_peaks(Traj_Space_data[marker_name][1], prominence_threshold_speed)

# Define segments based on speed thresholds.
speed_segments = traj_utils.find_speed_segments(marker_name, Traj_Space_data, time, speed_threshold, speed_peaks)

# Classify the speed segments into reach and return segments based on the trajectory data.
reach_speed_segments, return_speed_segments= traj_utils.classify_speed_segments(speed_segments, traj_data, marker_name, time)
print(f"Number of reach segments: {len(reach_speed_segments)}")

print(reach_speed_segments)


traj_utils.plot_speed_x_segmentsByspeed_space(marker_name, time, Traj_Space_data, traj_data, reach_speed_segments, speed_minima, speed_peaks, speed_threshold, prominence_threshold_speed, save_path)

# Calculate reach durations for all reaches.
reach_durations = traj_utils.calculate_reach_durations(reach_speed_segments)


# Calculate the reach position change (reach distance) for each reach segment
reach_distances = traj_utils.calculate_reach_distances(Traj_Space_data, marker_name, reach_speed_segments, time)
path_distances = traj_utils.calculate_reach_path_distances(Traj_Space_data, marker_name, reach_speed_segments, time)

v_peaks, v_peak_indices = traj_utils.calculate_v_peaks(Traj_Space_data, marker_name, time, reach_speed_segments)

'''
Define test windows for extracting further metrics
'''

# test_windows = reach_speed_segments

# Split reach_speed_segments into two parts: start to peak speed, and peak to end
# start_to_peak_segments, peak_to_end_segments = traj_utils.split_segments_at_peak_speed(reach_speed_segments, Traj_Space_data, time, v_peaks, marker_name)
# test_windows = start_to_peak_segments
# test_windows = peak_to_end_segments

# Define test windows as 50 ms before and after the peak speed for each reach segment
window_size = 0.05  # 50 ms

# Extract time windows for all reaches based on the peak speed indices
test_windows = []

for v_peak_index in v_peak_indices:
    start_time = time[v_peak_index - window_size * 200] # 50 ms before peak speed
    end_time = time[v_peak_index + window_size * 200] # 50 ms after peak speed
    test_windows.append((start_time, end_time))

# Calculates the Log Dimensionless Jerk (LDLJ) for all test windows of a given marker and saves the peak acceleration and jerk for each window.
LDLJ_values, acc_peaks, jerk_peaks = traj_utils.calculate_ldlj_for_all_reaches(Traj_Space_data, marker_name, time, test_windows)


print(test_windows)
'''
    H1:
    Reach duration increases with reach distance.
    Reach duration increases with path distance.
    Peak speed increases with reach distance.
    LLDJ increase (closer to 0) with decrease in reach distance. (shorter reaches are smoother.)
'''


# H1: Duration vs Distance
corr_dur_dist, p_dur_dist = pearsonr(reach_durations, reach_distances)
print(f"Correlation (Duration vs Distance): {corr_dur_dist}, P-value: {p_dur_dist}")

# H2: Duration vs Path Distance
corr_dur_path, p_dur_path = pearsonr(reach_durations, path_distances)
print(f"Correlation (Duration vs Path Distance): {corr_dur_path}, P-value: {p_dur_path}")

# H3: Peak Speed vs Distance
corr_peak_dist, p_peak_dist = pearsonr(v_peaks, reach_distances)
print(f"Correlation (Peak Speed vs Distance): {corr_peak_dist}, P-value: {p_peak_dist}")

# H4: LDLJ vs Distance
corr_ldlj_dist, p_ldlj_dist = pearsonr(LDLJ_values, reach_distances)
print(f"Correlation (LDLJ vs Distance): {corr_ldlj_dist}, P-value: {p_ldlj_dist}")
