import traj_utils

file_path = "/Users/yilinwu/Desktop/honours data/05/13/YW_tBBT01_Trajectories.csv"
traj_data, Frame, time = traj_utils.CSV_To_traj_data(file_path)  
Traj_Space_data = traj_utils.Traj_Space_data(traj_data)

marker_name="RFIN" # C7, T10, CLAV, STRN, LSHO, LUPA, LUPB, LUPC, LELB, LMEP, LWRA, LWRB, LFRA, LFIN, RSHO, RUPA, RUPB, RUPC, RELB, RMEP, RWRA, RWRB, RFRA, RFIN
prominence_threshold_position = 30  # Adjust as needed
prominence_threshold_speed = 350  # Adjust as needed
save_path="/Users/yilinwu/Desktop/honours/Thesis/figure"

# CSV_To_traj_data.find_local_minima_peaks(data, prominence_threshold)
position_minima, position_peaks = traj_utils.find_local_minima_peaks(Traj_Space_data[marker_name][0], prominence_threshold_position)
speed_minima, speed_peaks = traj_utils.find_local_minima_peaks(Traj_Space_data[marker_name][1], prominence_threshold_speed)

# Define the speed threshold
speed_threshold = 100  # Adjust as needed
speed_segments = traj_utils.find_speed_segments(marker_name, Traj_Space_data, time, speed_threshold, speed_peaks)

# Thresholds for movement start and end
z_range, speed_range, accel_range,x_threshold = (830, 860), (0, 150), (0, 3500),200
# Extracting the start and end indices of the movement
tStart_indices, tEnd_indices=traj_utils.extract_marker_start_end(marker_name, traj_data, Traj_Space_data, time, x_threshold, z_range, speed_range, accel_range)
"""
tStart_representatives/tEnd_representatives: list[float]
    A list of median of each cluster of tStart_indices/tEnd_indices for the start/end of movement clusters. 
    where clusters are formed based on significant differences in consecutive time indices. top 16 indices are used to form clusters.
    tStart_indices/tEnd_indices are caculated based on the threshold of RFIN_X, RFIN_Z, RFIN_Speed, and RFIN_Acceleration.
"""
tStart_representatives, tEnd_representatives=traj_utils.cluster_and_find_representatives(tStart_indices, tEnd_indices, top_n=15)


# '''
# Plotting the trajectory and space data
# '''
# traj_utils.plot_marker_trajectory_components(time, traj_data, Traj_Space_data, marker_name, save_path)
# traj_utils.plot_marker_with_start_end_representatives(time, traj_data, Traj_Space_data, marker_name, tStart_indices, tEnd_indices, tStart_representatives, tEnd_representatives, save_path, x_threshold, z_range, speed_range, accel_range)

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
# traj_utils.plot_pos_speed_speed_minima_space(time, Traj_Space_data, speed_minima, marker_name, save_path)
# traj_utils.plot_speed_position_segmentsByspeed_space(marker_name, time, Traj_Space_data, speed_segments, speed_threshold, prominence_threshold_speed, save_path)


