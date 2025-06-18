import traj_utils
from scipy.stats import pearsonr
import pprint
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import os
import itertools
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def process_files(file_paths, marker_name, prominence_threshold_speed, speed_threshold, window_size, time_window_method, save_path):
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
        reach_speed_segments, return_speed_segments = traj_utils.classify_speed_segments(speed_segments, traj_data, marker_name, time)

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
        # print(f"Processing time windows (Method {time_window_method})")
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
        
        # print(f"Method {time_window_method} - Correlation (Duration vs Distance): {corr_dur_dist}, P-value: {p_dur_dist}")
        # print(f"Method {time_window_method} - Correlation (Duration vs Path Distance): {corr_dur_path}, P-value: {p_dur_path}")
        # print(f"Method {time_window_method} - Correlation (Peak Speed vs Distance): {corr_dist_vpeaks}, P-value: {p_dist_vpeaks}")
        # print(f"Method {time_window_method} - Correlation (LDLJ vs Distance): {corr_dist_ldlj}, P-value: {p_dist_ldlj}")


        ''' Plotting '''
        # traj_utils.plot_marker_trajectory_components(time, traj_data, Traj_Space_data, marker_name, save_path)
        # traj_utils.plot_single_marker_space(Traj_Space_data, time, marker_name, save_path)
        # traj_utils.plot_pos_speed_one_extrema_space(time, Traj_Space_data, speed_minima, speed_peaks, marker_name, save_path)
        # traj_utils.plot_x_speed_one_extrema_space(time, Traj_Space_data, traj_data, speed_minima, speed_peaks, marker_name, save_path)
        # traj_utils.plot_speed_x_segmentsByspeed_space(marker_name, time, Traj_Space_data, traj_data, speed_segments, speed_minima, speed_peaks, speed_threshold, prominence_threshold_speed, save_path)
        # traj_utils.plot_speed_position_segmentsByspeed_space(marker_name, time, Traj_Space_data, speed_segments, speed_minima, speed_threshold, prominence_threshold_speed, save_path)
        # traj_utils.plot_marker_3d_trajectory_traj(traj_data, marker_name, 0, 1001, save_path) # start_frame, end_frame
        # traj_utils.plot_marker_radial_components_space(time, traj_data, marker_name, save_path)
        # traj_utils.plot_reach_acceleration_with_ldlj_normalised(time, Traj_Space_data, marker_name, reach_speed_segments, LDLJ_values, save_path)
        # traj_utils.plot_split_segments_speed(time, Traj_Space_data, marker_name, start_to_peak_segments, peak_to_end_segments, save_path)


        ''' important plots '''
        # traj_utils.plot_x_position_and_speed_with_segments(time, traj_data, Traj_Space_data, marker_name, reach_speed_segments, save_path, file_path) 
        # traj_utils.plot_aligned_segments(time, Traj_Space_data, reach_speed_segments, marker_name, save_path, file_path)
        # traj_utils.plot_aligned_segments_xyz(time, traj_data, reach_speed_segments, marker_name, save_path, file_path)
        # traj_utils.plot_reach_speed_and_jerk(time, Traj_Space_data, marker_name, reach_speed_segments, LDLJ_values, save_path, file_path)

        # ''' Plotting Time Windows Specific Metrics '''
        # traj_utils.rank_and_visualize_ldlj(reach_speed_segments, LDLJ_values, save_path,file_path)
        # traj_utils.plot_correlation_matrix(LDLJ_values, reach_distances, path_distances, v_peaks, acc_peaks, reach_durations, jerk_peaks, speed_threshold, save_path, file_path)

        # ''' Plotting correlations '''
        # traj_utils.plot_combined_correlations(reach_durations, reach_distances, path_distances, v_peaks, LDLJ_values,
        #                         corr_dur_dist, p_dur_dist, corr_dur_path, p_dur_path,
        #                         corr_dist_vpeaks, p_dist_vpeaks, corr_dist_ldlj, p_dist_ldlj,
        #                         save_path, file_path)

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

prominence_threshold_speed = 350
speed_threshold = 300

window_size = 0.05  # 50 ms
time_window_method = 3  # Choose 1, 2, or 3

# save_path="/Users/yilinwu/Desktop/honours/Thesis/figure"
save_path="/Users/yilinwu/Desktop/honours/Thesis/save_fig/L_M3"

# # Example usage:
# file_paths = [
#     "/Users/yilinwu/Desktop/honours data/Filter_10Hz/Trajectories/05/13/YW_tBBT01.csv",  # right hand
#     "/Users/yilinwu/Desktop/honours data/Filter_10Hz/Trajectories/05/13/YW_tBBT03.csv",  # right hand
# ]

file_paths = [
    "/Users/yilinwu/Desktop/honours data/Filter_10Hz/Trajectories/05/13/YW_tBBT02.csv",  # left hand
    "/Users/yilinwu/Desktop/honours data/Filter_10Hz/Trajectories/05/13/YW_tBBT04.csv"   # left hand
]

# # C7, T10, CLAV, STRN, LSHO, LUPA, LUPB, LUPC, LELB, LMEP, LWRA, LWRB, LFRA, LFIN, RSHO, RUPA, RUPB, RUPC, RELB, RMEP, RWRA, RWRB, RFRA, RFIN
# marker_name = "RFIN"
marker_name = "LFIN"  

results = process_files(file_paths, marker_name, prominence_threshold_speed, speed_threshold, window_size, time_window_method,save_path)



'''# Example usage for processing multiple files for both hands'''

# marker_name = "RFIN"
# file_paths = [
#     "/Users/yilinwu/Desktop/honours data/Filter_10Hz/Trajectories/05/13/YW_tBBT01.csv",  # right hand
#     "/Users/yilinwu/Desktop/honours data/Filter_10Hz/Trajectories/05/13/YW_tBBT03.csv",  # right hand
# ]
# results_R = process_files(file_paths, marker_name, prominence_threshold_speed, speed_threshold, window_size, time_window_method,save_path)

# marker_name = "LFIN"  
# file_paths = [
#     "/Users/yilinwu/Desktop/honours data/Filter_10Hz/Trajectories/05/13/YW_tBBT02.csv",  # left hand
#     "/Users/yilinwu/Desktop/honours data/Filter_10Hz/Trajectories/05/13/YW_tBBT04.csv"   # left hand
# ]
# results_L = process_files(file_paths, marker_name, prominence_threshold_speed, speed_threshold, window_size, time_window_method,save_path)

# def combine_results(results_L, results_R):
#     import numpy as np

#     combined_results = {}

#     # Merge both result sets
#     all_keys = set(results_L.keys()).union(results_R.keys())
#     for file_path in all_keys:
#         combined_results[file_path] = {}

#         # Initialize data
#         params_L = results_L.get(file_path, {}).get('parameters', {})
#         params_R = results_R.get(file_path, {}).get('parameters', {})

#         combined_parameters = {}

#         # Merge parameter arrays
#         for key in ['reach_durations', 'reach_distances', 'path_distances',
#                     'v_peaks', 'LDLJ_values', 'acc_peaks', 'jerk_peaks']:
#             data_L = params_L.get(key, [])
#             data_R = params_R.get(key, [])
#             combined_parameters[key] = np.concatenate([data_L, data_R])

#         # Merge correlation tuples
#         combined_correlations = {}
#         correlation_keys = ['duration_distance', 'duration_path_distance',
#                             'distance_peak_speed', 'distance_ldlj']
#         for key in correlation_keys:
#             tuple_L = params_L.get('correlations', {}).get(key, (np.nan, np.nan))
#             tuple_R = params_R.get('correlations', {}).get(key, (np.nan, np.nan))

#             # Average correlation and p-values, ignoring NaNs
#             corr_vals = [val for val in [tuple_L[0], tuple_R[0]] if not np.isnan(val)]
#             p_vals = [val for val in [tuple_L[1], tuple_R[1]] if not np.isnan(val)]
#             avg_corr = np.mean(corr_vals) if corr_vals else np.nan
#             avg_p = np.mean(p_vals) if p_vals else np.nan

#             combined_correlations[key] = (avg_corr, avg_p)

#         combined_parameters['correlations'] = combined_correlations
#         combined_results[file_path]['parameters'] = combined_parameters

#         # Merge test_windows
#         test_L = results_L.get(file_path, {}).get('test_windows', [])
#         test_R = results_R.get(file_path, {}).get('test_windows', [])
#         combined_results[file_path]['test_windows'] = test_L + test_R

#         # Merge time
#         time_L = results_L.get(file_path, {}).get('time', [])
#         time_R = results_R.get(file_path, {}).get('time', [])
#         combined_results[file_path]['time'] = np.concatenate([time_L, time_R])

#     return combined_results
# combined_results = combine_results(results_L, results_R)
# summary = traj_utils.summarize_data_across_files(combined_results)

''' Example usage for printing results '''
# # Print results in a more readable way
# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(results)

'''# Example usage for processing a single file'''
# def plot_overlayed_reaches(results, marker_name, save_path, file_paths):
#     """
#     Plots all reaches corresponding x, y, z coordinates in each file and overlays across files in separate subplots.
#     Groups reaches in sets of 4 to color them using 4 base colors.
#     """
#     fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
#     axes[0].set_title(f"Overlayed Reaches for Marker: {marker_name} ({', '.join([os.path.basename(fp) for fp in file_paths])})")
#     axes[0].set_ylabel("X Coordinate")
#     axes[1].set_ylabel("Y Coordinate")
#     axes[2].set_ylabel("Z Coordinate")
#     axes[2].set_xlabel("Time (s)")

#     base_colors = ["orange", "green", "blue", "purple"]
#     reach_counter = 0  # Tracks total number of reaches plotted

#     for file_path, data in results.items():
#         time = data['time']
#         test_windows = data['test_windows']
#         traj_data = traj_utils.Traj_Space_data(traj_utils.CSV_To_traj_data(file_path)[0])

#         for start_time, end_time in test_windows:
#             start_idx = np.searchsorted(time, start_time)
#             end_idx = np.searchsorted(time, end_time)

#             x_coords = traj_data[marker_name][0][start_idx:end_idx]
#             y_coords = traj_data[marker_name][1][start_idx:end_idx]
#             z_coords = traj_data[marker_name][2][start_idx:end_idx]
#             time_segment = time[start_idx:end_idx]

#             color = base_colors[(reach_counter // 4) % len(base_colors)]
#             # label = f"Reach {reach_counter + 1} ({start_time:.2f}-{end_time:.2f}s) from {os.path.basename(file_path)}"
#             label = f"{reach_counter + 1}"

#             axes[0].plot(time_segment - start_time, x_coords, color=color, label=label)
#             axes[1].plot(time_segment - start_time, y_coords, color=color)
#             axes[2].plot(time_segment - start_time, z_coords, color=color)

#             reach_counter += 1

#     for ax in axes:
#         handles, labels = ax.get_legend_handles_labels()
#         unique_labels = dict(zip(labels, handles))  # Remove duplicate labels
#         ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper right", fontsize=5, frameon=False)
#         ax.grid(True)

#     if save_path:
#         unique_name = f"overlayed_reaches_{marker_name}_xyz_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
#         full_path = os.path.join(save_path, unique_name)
#         plt.savefig(full_path)
#         print(f"Figure saved to {full_path}")

#     plt.tight_layout()
#     plt.show()
# plot_overlayed_reaches(results, marker_name, save_path,file_paths)

# traj_utils.rank_and_visualize_ldlj_files(results, save_path)

def find_best_predictor_combination(results, save_path):
    """
    Finds the best combination of predictors to predict reach_durations using multiple linear regression.
    Saves a table of tested combinations and their R² values as a figure.
    """

    # Extract data from results
    reach_distances = np.concatenate([results[file]['parameters']['reach_distances'] for file in results])
    path_distances = np.concatenate([results[file]['parameters']['path_distances'] for file in results])
    v_peaks = np.concatenate([results[file]['parameters']['v_peaks'] for file in results])
    LDLJ_values = np.concatenate([results[file]['parameters']['LDLJ_values'] for file in results])
    acc_peaks = np.concatenate([results[file]['parameters']['acc_peaks'] for file in results])
    jerk_peaks = np.concatenate([results[file]['parameters']['jerk_peaks'] for file in results])
    reach_durations = np.concatenate([results[file]['parameters']['reach_durations'] for file in results])

    # Create a DataFrame for easier manipulation
    data = pd.DataFrame({
        'reach_distances': reach_distances,
        'path_distances': path_distances,
        'v_peaks': v_peaks,
        'LDLJ_values': LDLJ_values,
        'acc_peaks': acc_peaks,
        'jerk_peaks': jerk_peaks,
        'reach_durations': reach_durations
    })

    # Define all predictors
    predictors = ['reach_distances', 'path_distances', 'v_peaks', 'LDLJ_values', 'acc_peaks', 'jerk_peaks']

    # Store best model info
    best_adj_r2 = -np.inf
    best_combo = None
    best_model = None

    results_summary = []

    # Iterate through all combinations of predictors
    for r in range(1, len(predictors) + 1):
        for combo in itertools.combinations(predictors, r):
            X = data[list(combo)]
            y = data['reach_durations']

            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            r2 = r2_score(y, y_pred)
            n = len(y)
            k = X.shape[1]
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

            results_summary.append({
                'features': combo,
                'r2': r2,
                'adjusted_r2': adj_r2
            })

            if adj_r2 > best_adj_r2:
                best_adj_r2 = adj_r2
                best_combo = combo
                best_model = model

    # Output best combination
    print(f"Best Feature Combination: {best_combo}")
    print(f"Adjusted R^2: {best_adj_r2:.4f}")
    print(f"Model Coefficients: {dict(zip(best_combo, best_model.coef_))}")
    print(f"Intercept: {best_model.intercept_:.4f}")

    # Optional: Convert all results to DataFrame for further inspection
    results_df = pd.DataFrame(results_summary).sort_values(by='adjusted_r2', ascending=False)

    # Identify the best adjusted R^2 value
    best_row = results_df.iloc[0]

    # Prepare the table data
    table_data = []
    for _, row in results_df.iterrows():
        is_best = "⬅️ Best" if row['adjusted_r2'] == best_row['adjusted_r2'] else ""
        table_data.append([
            ', '.join(row['features']),
            f"{row['r2']:.4f}",
            f"{row['adjusted_r2']:.4f}",
            is_best
        ])

    # Define headers
    headers = ["Feature Combination", "R²", "Adjusted R²", "Note"]

    # Create a figure for the table
    fig, ax = plt.subplots(figsize=(12, len(table_data) * 0.5))
    ax.axis('tight')
    ax.axis('off')

    # Add the table to the figure
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(headers))))

    # Save the figure
    output_path = os.path.join(save_path, "tested_combinations_table.png")
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Table figure saved to {output_path}")

    plt.show()

    return best_combo, best_adj_r2, best_model

find_best_predictor_combination(results, save_path)

'''example usage for summarizing data across files and examining correlations'''
summary = traj_utils.summarize_data_across_files(results)

# Examine correlations across all files using the summary data
summary_corr_dur_dist, summary_p_dur_dist = pearsonr(summary['reach_durations'], summary['reach_distances'])
summary_corr_dur_path, summary_p_dur_path = pearsonr(summary['reach_durations'], summary['path_distances'])
summary_corr_dist_vpeaks, summary_p_dist_vpeaks = pearsonr(summary['reach_distances'], summary['v_peaks'])
summary_corr_dist_ldlj, summary_p_dist_ldlj = pearsonr(summary['reach_distances'], summary['LDLJ_values'])

# print(f"Summary - Correlation (Duration vs Distance): {summary_corr_dur_dist}, P-value: {summary_p_dur_dist}")
# print(f"Summary - Correlation (Duration vs Path Distance): {summary_corr_dur_path}, P-value: {summary_p_dur_path}")
# print(f"Summary - Correlation (Peak Speed vs Distance): {summary_corr_dist_vpeaks}, P-value: {summary_p_dist_vpeaks}")
# print(f"Summary - Correlation (LDLJ vs Distance): {summary_corr_dist_ldlj}, P-value: {summary_p_dist_ldlj}")

# # Plotting correlations for summary data
traj_utils.plot_combined_correlations(
    summary['reach_durations'], summary['reach_distances'], summary['path_distances'], summary['v_peaks'], summary['LDLJ_values'],
    summary_corr_dur_dist, summary_p_dur_dist, summary_corr_dur_path, summary_p_dur_path,
    summary_corr_dist_vpeaks, summary_p_dist_vpeaks, summary_corr_dist_ldlj, summary_p_dist_ldlj,
    save_path,file_paths
)

# traj_utils.plot_summarize_correlation_matrix(summary, save_path,file_paths)











