import utils1 # Importing utils1 for data Pre-processing
import utils2 # Importing utils2 for reach metrics calculation and time window Specific calculation
import utils3 # Importing utils3 for plotting functions
import utils4 # Importing utils4 for image files
import utils5 # Importing utils5 for combining metrics
import utils6 # Importing utils6 for Data Analysis and Visualization
import utils7 # Importing utils7 for Motor Experiences
import utils8 # Importing utils8 for sBBTResult
# -------------------------------------------------------------------------------------------------------------------
import pickle
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

import pickle
import math
import numpy as np
from scipy.stats import zscore
from scipy.stats import wilcoxon
from scipy.stats import spearmanr

import pandas as pd
import seaborn as sns

import pingouin as pg
from scipy.stats import spearmanr, zscore
from scipy.stats import zscore, spearmanr

import statsmodels.api as sm
import statsmodels.formula.api as smf

# -------------------------------------------------------------------------------------------------------------------

Traj_folder = "/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025"
Box_Traj_folder = "/Users/yilinwu/Desktop/Yilin-Honours/Box/Traj/2025"
Figure_folder = "/Users/yilinwu/Desktop/honours/Thesis/figure"
DataProcess_folder = "/Users/yilinwu/Desktop/honours data/DataProcess"
tBBT_Image_folder = "/Users/yilinwu/Desktop/Yilin-Honours/tBBT_Image/2025/"

prominence_threshold_speed = 400
prominence_threshold_position = 80

# -------------------------------------------------------------------------------------------------------------------

# --- GET ALL DATES ---
All_dates = sorted(utils1.get_subfolders_with_depth(Traj_folder, depth=3))

# # --- SELECT DATES TO PROCESS ---
# All_dates = All_dates[25:len(All_dates)]  # Process all dates from index 25 to the end

# # # -------------------------------------------------------------------------------------------------------------------

# # PART 0: Data Pre-processing [!!! THINGS THAT NEED TO BE DONE ONCE !!!]

# # --- PROCESS ALL DATE AND SAVE ALL MOVEMENT DATA AS pickle file ---
# utils1.process_all_dates_separate(All_dates, Traj_folder, Box_Traj_folder, Figure_folder, DataProcess_folder, 
#                       prominence_threshold_speed, prominence_threshold_position)

# # # --- RENAME IMAGE FILES ---
# # # run this only once to rename the files in the tBBT_Image_folder
# # for date in All_dates[23:len(All_dates)]:  # Process all dates from index 23 to the end
# #     directory = f"{tBBT_Image_folder}{date}"
# #     print(f"Renaming files in directory: {directory}")
# #     utils4.rename_files(directory)

# # # --- FIND BEST CALIBRATION IMAGES COMBINATION FOR EACH SUBJECT ---
# # subjects = [date for date in All_dates] 
# # utils4.run_test_for_each_subject(subjects, tBBT_Image_folder)

# --- PROCESS ALL SUBJECTS' IMAGES RETURN tBBT ERROR FROM IMAGE, SAVE AS pickle file---
# All_Subject_tBBTs_errors = utils4.process_all_subjects_images(All_dates, tBBT_Image_folder, DataProcess_folder)

# def process_all_block_errors(All_Subject_tBBTs_errors, DataProcess_folder):
    
#     def compute_block_errors_for_error_trial(hand, markerindex, p3_block2, bg):
#         """
#         Compute block errors based on detected markers.

#         Parameters:
#             hand (str): 'right' or 'left'
#             markerindex (list or array-like): List of marker indices.
#             p3_block2 (list or array-like): List of points corresponding to markers.
#             bg: An object with attributes grid_xxL, grid_yyL, grid_xxR, grid_yyR.
            
#         Returns:
#             list: block_errors is a list of dictionaries with keys 'point', 'membership', and 'distance'.
#         """
#         if hand == 'right':  # right hand
#             x = bg.grid_xxL.flatten()
#             x = x[::-1]
#             y = bg.grid_yyL.flatten()
#         else:  # left hand
#             x = bg.grid_xxR.flatten()
#             y = bg.grid_yyR.flatten()

#         # Predefined block membership order
#         blockMembership = [12, 1, 14, 3, 8, 13, 2, 15, 0, 5, 6, 11, 4, 9, 7, 10]

#         block_errors = []

#         for i in range(len(markerindex)):
#             marker_idx = markerindex[i]
#             point = p3_block2[i]
#             membership = blockMembership[marker_idx]

#             if blockMembership[marker_idx] == 6:
#                 membership = 2
#             elif blockMembership[marker_idx] == 2:
#                 membership = 6

#             distance = (((point[0] + 12.5 - x[membership]) ** 2 +
#                          (point[1] + 12.5 - y[membership]) ** 2) ** 0.5)

#             block_errors.append({
#                 'point': point,
#                 'membership': membership,
#                 'distance': distance
#             })

#         return block_errors

#     left_file_indices = [19, 20, 21, 30, 31, 32]
#     right_file_indices = [20, 21, 30, 31, 32]

#     for hand in ['left', 'right']:
#         fixed_indices = left_file_indices if hand == 'left' else right_file_indices
#         for idx in fixed_indices:
#             p3_block2 = All_Subject_tBBTs_errors['08/08/NS', hand][idx]["p3_block2"]
#             bg = All_Subject_tBBTs_errors['08/08/NS', hand][idx]["bg"]
#             markerindex = list(range(16))  # Equivalent to [0, 1, 2, ..., 15]
#             All_Subject_tBBTs_errors['08/08/NS', hand][idx]['block_errors'] = compute_block_errors_for_error_trial(hand, markerindex, p3_block2, bg)

#     # Extract ordered distances
#     utils4.extract_ordered_distances(All_Subject_tBBTs_errors, DataProcess_folder)

#     output_file_path = os.path.join(DataProcess_folder, "All_Subject_tBBTs_errors.pkl")
#     with open(output_file_path, 'wb') as f:
#         pickle.dump(All_Subject_tBBTs_errors, f)

# process_all_block_errors(All_Subject_tBBTs_errors, DataProcess_folder)
# # # -------------------------------------------------------------------------------------------------------------------

# PART 1: CHECK IF DATA PROCESSING IS DONE AND LOAD RESULTS
# --- CHECK CALIBRATION FOLDERS FOR PICKLE FILES ---
utils4.check_calibration_folders_for_pickle(All_dates, tBBT_Image_folder)

# --- LOAD ALL SUBJECTS' tBBT ERROR FROM IMAGE, SAVE AS pickle file---
Block_Distance = utils4.load_selected_subject_errors(All_dates, DataProcess_folder)

# --- LOAD RESULTS FROM PICKLE FILE "processed_results.pkl" ---
results = utils1.load_selected_subject_results(All_dates, DataProcess_folder)

# -------------------------------------------------------------------------------------------------------------------
# PART 2: Reach Metrics Calculation
# --- GET REACH SPEED SEGMENTS ---
reach_speed_segments = utils2.get_reach_speed_segments(results)

# -------------------------------------------------------------------------------------------------------------------

# --- CALCULATE REACH METRICS ---
# reach_durations
# reach_cartesian_distances
# reach_path_distances
# reach_v_peaks
# reach_v_peak_indices
reach_metrics = utils2.calculate_reach_metrics(reach_speed_segments, results, fs=200)

# --- DEFINE TIME WINDOWS BASED ON SELECTED METHOD ---
# test_windows_1: Original full reach segments (start to end of movement)
# test_windows_2: From movement start to velocity peak (focuses on movement buildup)
# test_windows_3: Symmetric window around velocity peak (captures activity before and after peak) (500 ms total)
# test_windows_4: 100 ms before velocity peak (captures lead-up dynamics)
# test_windows_5: 100 ms after velocity peak (captures immediate post-peak activity)
# test_windows_6: Custom time window centered around the midpoint of each segment 
test_windows_1, test_windows_2, test_windows_3, test_windows_4, test_windows_5, test_windows_6 = utils2.define_time_windows(reach_speed_segments, reach_metrics, fs=200, window_size=0.25)

# --- CALCULATE REACH METRICS SPECIFIC TO TIME WINDOW ---
# reach_acc_peaks
# reach_jerk_peaks
# reach_LDLJ
reach_TW_metrics_test_windows_1 = utils2.calculate_reach_metrics_for_time_windows_Normalizing(test_windows_1, results)
reach_TW_metrics_test_windows_3 = utils2.calculate_reach_metrics_for_time_windows_Normalizing(test_windows_3, results)

# --- CALCULATE SPARC FOR EACH TEST WINDOW FOR ALL DATES, HANDS, AND TRIALS ---
reach_sparc_test_windows_1_Normalizing = utils2.calculate_reach_sparc_Normalizing(test_windows_1, results)
reach_sparc_test_windows_3_Normalizing = utils2.calculate_reach_sparc_Normalizing(test_windows_3, results)

# # --- Save ALL LDLJ VALUES BY SUBJECT, HAND, AND TRIAL ---
# utils2.save_ldlj_values(reach_TW_metrics_test_windows_1, DataProcess_folder)

# # --- Save ALL SPARC VALUES BY SUBJECT, HAND, AND TRIAL ---
# utils2.save_sparc_values(reach_sparc_test_windows_1_Normalizing, DataProcess_folder)

# # -------------------------------------------------------------------------------------------------------------------
# Calculate RMS reprojection error statistics
utils4.compute_rms_reprojection_error_stats()

# -------------------------------------------------------------------------------------------------------------------
# Load Motor Experiences from CSV into a dictionary
MotorExperiences = utils7.load_motor_experiences("/Users/yilinwu/Desktop/Yilin-Honours/MotorExperience.csv")
# Calculate demographic variables from MotorExperiences
utils7.display_motor_experiences_stats(MotorExperiences)
# Update MotorExperiences with weighted scores
utils7.update_overall_h_total_weighted(MotorExperiences)

# -------------------------------------------------------------------------------------------------------------------

# Load sBBTResult from CSV into a DataFrame and compute right and left hand scores
sBBTResult = utils8.load_and_compute_sbbt_result()
# Swap and rename sBBTResult scores for specific subjects
sBBTResult = utils8.swap_and_rename_sbbt_result(sBBTResult)
sBBTResult_stats = utils8.compute_sbbt_result_stats(sBBTResult)

# -------------------------------------------------------------------------------------------------------------------
# Examine correlations between motor experience metrics and sBBT scores by extracting the highest score for each hand
motor_keys = [
    "physical_h_total_weighted",
    "musical_h_total_weighted",
    "digital_h_total_weighted",
    "other_h_total_weighted",
    "overall_h_total_weighted"
]
score_columns = ["dominant", "non_dominant"]

utils7.analyze_motor_experience_correlations(motor_keys, score_columns, sBBTResult, MotorExperiences)

# -------------------------------------------------------------------------------------------------------------------
def plot_single_trial_p_v_a_j_Nj_in_one(results, reach_speed_segments=None,
                                  subject="06/19/CZ", hand="right", trial=1,
                                  file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/06/19/CZ/CZ_tBBT01.csv',
                                  fs=200, target_samples=101, plot_option=1, seg_index=0):
    """
    Plots normalized jerk (and other signals) from trajectory data with three options:
      1. Plot the entire trial.
      2. Plot a single segment from the trial (requires reach_speed_segments and seg_index).
      3. Overlay all segments from the trial (requires reach_speed_segments).

    Parameters:
        results (dict): Dictionary containing trajectory data.
        reach_speed_segments (dict or None): Dictionary of segmentation ranges 
            with structure: reach_speed_segments[subject][hand][file_path] = list of (start_idx, end_idx) tuples.
            Required for plot_option 2 and 3.
        subject (str): Subject identifier.
        hand (str): Hand identifier.
        trial (int): Trial index.
        file_path (str): Path to the CSV file.
        fs (int): Sampling rate in Hz.
        target_samples (int): Number of samples for the normalized jerk signal.
        plot_option (int): 
            1 => Plot whole trial,
            2 => Plot one segment (specify seg_index),
            3 => Overlay all segments.
        seg_index (int): Index of the segment to plot (used when plot_option == 2).

    This function extracts the trajectory data (Position, Velocity, Acceleration, Jerk)
    using marker 'RFIN' for right hand or 'LFIN' for left hand, computes a normalized jerk
    signal (via linear interpolation) and plots five subplots:
      Position, Velocity, Acceleration, Original Jerk, and Normalized Jerk.
    """
    import matplotlib.pyplot as plt

    # Select marker based on hand.
    marker = 'RFIN' if hand == 'right' else 'LFIN'
    
    # Extract full trajectory arrays.
    traj_data = results[subject][hand][trial][file_path]['traj_space'][marker]
    position_full = traj_data[0]
    velocity_full = traj_data[1]
    acceleration_full = traj_data[2]
    jerk_full = traj_data[3]
    
    # Option 1: Plot whole trial.
    if plot_option == 1:
        duration = len(jerk_full) / fs  # total duration in seconds
        t_orig = np.linspace(0, duration, num=len(jerk_full))
        t_std = np.linspace(0, duration, num=target_samples)
        warped_jerk = np.interp(t_std, t_orig, jerk_full)

        fig, axs = plt.subplots(5, 1, figsize=(12, 10))
        axs[0].plot(position_full, color='blue')
        axs[0].set_title('Position - Whole Trial')
        
        axs[1].plot(velocity_full, color='green')
        axs[1].set_title('Velocity - Whole Trial')
        
        axs[2].plot(acceleration_full, color='red')
        axs[2].set_title('Acceleration - Whole Trial')
        
        axs[3].plot(jerk_full, color='purple')
        axs[3].set_title('Original Jerk - Whole Trial')
        
        axs[4].plot(t_std, warped_jerk, color='orange', linestyle='--')
        percent_ticks = np.linspace(0, 100, 6)
        time_ticks = np.linspace(0, duration, 6)
        axs[4].set_xticks(time_ticks)
        axs[4].set_xticklabels([f"{int(p)}%" for p in percent_ticks])
        axs[4].set_title('Normalized Jerk - Whole Trial')
        
        plt.tight_layout()
        plt.show()

    # Option 2: Plot a single segment from the trial.
    elif plot_option == 2:
        if reach_speed_segments is None:
            print("reach_speed_segments is required for segment plotting (plot_option 2).")
            return

        seg_ranges = reach_speed_segments[subject][hand][file_path]
        if seg_index < 0 or seg_index >= len(seg_ranges):
            print("Invalid seg_index provided.")
            return

        start_idx, end_idx = seg_ranges[seg_index]
        pos_seg = position_full[start_idx:end_idx]
        vel_seg = velocity_full[start_idx:end_idx]
        acc_seg = acceleration_full[start_idx:end_idx]
        jerk_seg = jerk_full[start_idx:end_idx]

        duration = len(jerk_seg) / fs
        t_orig = np.linspace(0, duration, num=len(jerk_seg))
        t_std = np.linspace(0, duration, num=target_samples)
        warped_jerk = np.interp(t_std, t_orig, jerk_seg)

        # Calculate the integral of the squared, warped jerk segment
        jerk_squared_integral = np.trapezoid(warped_jerk**2, t_std)

        # Get the peak speed for the current segment
        vpeak = vel_seg.max()
        dimensionless_jerk = (duration**3 / vpeak**2) * jerk_squared_integral
        LDLJ = -math.log(abs(dimensionless_jerk), math.e)

        fig, axs = plt.subplots(5, 1, figsize=(12, 10))
        axs[0].plot(pos_seg, color='blue')
        axs[0].set_title(f'Position - Segment {seg_index+1}')
        
        axs[1].plot(vel_seg, color='green')
        axs[1].set_title(f'Velocity - Segment {seg_index+1}')
        
        axs[2].plot(acc_seg, color='red')
        axs[2].set_title(f'Acceleration - Segment {seg_index+1}')
        
        axs[3].plot(jerk_seg, color='purple')
        axs[3].set_title(f'Original Jerk - Segment {seg_index+1}')
        
        axs[4].plot(t_std, warped_jerk, color='orange', linestyle='--')
        percent_ticks = np.linspace(0, 100, 6)
        time_ticks = np.linspace(0, duration, 6)
        axs[4].set_xticks(time_ticks)
        axs[4].set_xticklabels([f"{int(p)}%" for p in percent_ticks])
        axs[4].set_title(f'Normalized Jerk - Segment {seg_index+1} - (LDLJ: {LDLJ:.2f})')
        
        plt.tight_layout()
        plt.show()

    # Option 3: Overlay all segments from the trial.
    elif plot_option == 3:
        if reach_speed_segments is None:
            print("reach_speed_segments is required for overlay segment plotting (plot_option 3).")
            return

        seg_ranges = reach_speed_segments[subject][hand][file_path]
        fig, axs = plt.subplots(5, 1, figsize=(12, 10))
        
        # Set titles for each subplot.
        axs[0].set_title('Overlay - Position')
        axs[1].set_title('Overlay - Velocity')
        axs[2].set_title('Overlay - Acceleration')
        axs[3].set_title('Overlay - Original Jerk')
        axs[4].set_title('Overlay - Normalized Jerk')
        
        # For normalized jerk x-axis, we will compute duration per segment.
        for (start_idx, end_idx) in seg_ranges:
            seg_length = end_idx - start_idx
            pos_seg = position_full[start_idx:end_idx]
            vel_seg = velocity_full[start_idx:end_idx]
            acc_seg = acceleration_full[start_idx:end_idx]
            jerk_seg = jerk_full[start_idx:end_idx]

            duration = len(jerk_seg) / fs
            t_orig = np.linspace(0, duration, num=len(jerk_seg))
            t_std = np.linspace(0, duration, num=target_samples)
            warped_jerk = np.interp(t_std, t_orig, jerk_seg)
            x_vals = np.arange(seg_length)
            
            axs[0].plot(x_vals, pos_seg, alpha=0.7)
            axs[1].plot(x_vals, vel_seg, alpha=0.7)
            axs[2].plot(x_vals, acc_seg, alpha=0.7)
            axs[3].plot(x_vals, jerk_seg, alpha=0.7)
            # For normalized jerk, use percentage scale (0 to 100)
            percent_x = np.linspace(0, 100, len(warped_jerk))
            axs[4].plot(percent_x, warped_jerk, linestyle='--', alpha=0.7)
        
        # Set x-axis ticks for the normalized jerk subplot.
        percent_ticks = np.linspace(0, 100, 6)
        axs[4].set_xticks(percent_ticks)
        axs[4].set_xticklabels([f"{int(p)}%" for p in percent_ticks])
        
        plt.tight_layout()
        plt.show()
        
    else:
        print("Invalid plot_option. Choose 1 (whole trial), 2 (one segment), or 3 (overlay segments).")

# Option 1: Plot whole trial
plot_single_trial_p_v_a_j_Nj_in_one(results, subject="07/22/HW", hand="left", trial=1,
                              file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv',
                              fs=200, target_samples=101, plot_option=1)

# Option 2: Plot a single segment (e.g., first segment)
plot_single_trial_p_v_a_j_Nj_in_one(results, test_windows_3, subject="07/22/HW", hand="left", trial=1,
                              file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv',
                              fs=200, target_samples=101, plot_option=2, seg_index=0)

# Option 3: Overlay all segments
plot_single_trial_p_v_a_j_Nj_in_one(results, test_windows_3, subject="07/22/HW", hand="left", trial=1,
                              file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv',
                              fs=200, target_samples=101, plot_option=3)


def plot_trials_p_v_a_j_Nj_by_location(results, reach_speed_segments,
                                           subject="07/22/HW", hand="left", trial=1,
                                           fs=200, target_samples=101, metrics=None,
                                           mode="single", file_path=None):
    """
    Combined function to plot segmented signals with normalized jerk.
    
    Options:
      - mode="single": Plot only one selected trial.
      - mode="all":    Overlay plots from all trials (all file paths).
    
    Parameters:
        results (dict): Dictionary with trajectory data.
        reach_speed_segments (dict): Dictionary with segmentation ranges.
        subject (str): Subject identifier.
        hand (str): Hand identifier.
        trial (int): Trial index.
        fs (int): Sampling rate.
        target_samples (int): Number of samples for the normalized jerk signal.
        metrics (list or None): List of metrics to plot.
            Options include: 'pos', 'vel', 'acc', 'jerk', 'norm_jerk'.
            If None, all metrics are plotted.
        mode (str): "single" to plot one selected trial, "all" to overlay all trials.
        file_path (str or None): Required if mode is "single"; path to the CSV file.
    """
    import matplotlib.pyplot as plt

    # Default metrics if not provided.
    if metrics is None:
        metrics = ["pos", "vel", "acc", "jerk", "norm_jerk"]
    
    # Select marker based on hand.
    marker = 'RFIN' if hand == 'right' else 'LFIN'
    
    # -------------------------- Mode 1: Single trial --------------------------
    if mode == "single":
        if file_path is None:
            print("For mode 'single' you must provide a file_path.")
            return

        # Get segmentation ranges for the provided file.
        seg_ranges = reach_speed_segments[subject][hand][file_path]
        n_segments = len(seg_ranges)
        
        # Create grid of subplots.
        n_rows = int(np.ceil(np.sqrt(n_segments)))
        n_cols = int(np.ceil(n_segments / n_rows))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        axs = np.array(axs).flatten()
        
        # Extract signals from the selected file.
        traj_data = results[subject][hand][trial][file_path]['traj_space'][marker]
        position = traj_data[0]
        velocity = traj_data[1]
        acceleration = traj_data[2]
        jerk = traj_data[3]
        
        for i, (start_idx, end_idx) in enumerate(seg_ranges):
            seg_length = end_idx - start_idx
            x_vals = np.arange(seg_length)
            
            # Slice signals.
            pos_seg = position[start_idx:end_idx]
            vel_seg = velocity[start_idx:end_idx]
            acc_seg = acceleration[start_idx:end_idx]
            jerk_seg = jerk[start_idx:end_idx]
            
            # Normalize jerk via interpolation.
            duration = len(jerk_seg) / fs
            t_orig = np.linspace(0, duration, num=len(jerk_seg))
            t_std = np.linspace(0, duration, num=target_samples)
            warped_jerk = np.interp(t_std, t_orig, jerk_seg)
            
            # Mapping metric keys to value and plotting options.
            available_metrics = {
                "pos":      {"data": pos_seg, "label": "Pos", "x": x_vals, "linestyle": "-"},
                "vel":      {"data": vel_seg, "label": "Vel", "x": x_vals, "linestyle": "-"},
                "acc":      {"data": acc_seg, "label": "Acc", "x": x_vals, "linestyle": "-"},
                "jerk":     {"data": jerk_seg, "label": "Jerk", "x": x_vals, "linestyle": "-"},
                "norm_jerk": {"data": warped_jerk, "label": "Norm Jerk", "x": np.linspace(0, 100, len(warped_jerk)), "linestyle": "--"}
            }
            
            ax = axs[i]
            for key in metrics:
                if key in available_metrics:
                    met = available_metrics[key]
                    ax.plot(met["x"], met["data"], label=met["label"], linestyle=met["linestyle"], alpha=0.7)
            
            ax.set_title(f"Segment {i+1}\nDuration: {duration:.2f}s")
            ax.set_xlabel("Samples / %")
            ax.set_ylabel("Signal")
            ax.grid(True)
            ax.legend(fontsize=8)
        
        # Hide any unused subplots.
        for j in range(i + 1, len(axs)):
            axs[j].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # -------------------------- Mode 2: All trials --------------------------
    elif mode == "all":
        # Get all file paths for the trial.
        file_paths = list(results[subject][hand][trial].keys())
        if not file_paths:
            print("No files found for the specified subject/hand/trial.")
            return
        
        # Use segmentation of the first file to setup grid.
        seg_ranges = reach_speed_segments[subject][hand][file_paths[0]]
        n_segments = len(seg_ranges)
        
        # Create grid of subplots.
        n_rows = int(np.ceil(np.sqrt(n_segments)))
        n_cols = int(np.ceil(n_segments / n_rows))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        axs = np.array(axs).flatten()
        
        # Loop over each segment index.
        for seg_idx in range(n_segments):
            ax = axs[seg_idx]
            plotted_labels = {}
            avg_duration = 0
            count = 0
            
            # Loop over each file for overlay.
            for fp in file_paths:
                # Get segmentation range for current segment.
                start_idx, end_idx = reach_speed_segments[subject][hand][fp][seg_idx]
                seg_length = end_idx - start_idx
                x_vals = np.arange(seg_length)
                
                # Extract signals from current file.
                traj_data = results[subject][hand][trial][fp]['traj_space'][marker]
                position = traj_data[0]
                velocity = traj_data[1]
                acceleration = traj_data[2]
                jerk = traj_data[3]
                
                pos_seg = position[start_idx:end_idx]
                vel_seg = velocity[start_idx:end_idx]
                acc_seg = acceleration[start_idx:end_idx]
                jerk_seg = jerk[start_idx:end_idx]
                
                # Normalize jerk.
                duration = len(jerk_seg) / fs
                avg_duration += duration
                count += 1
                t_orig = np.linspace(0, duration, num=len(jerk_seg))
                t_std = np.linspace(0, duration, num=target_samples)
                warped_jerk = np.interp(t_std, t_orig, jerk_seg)
                
                available_metrics = {
                    "pos":       {"data": pos_seg, "label": "Pos", "x": x_vals, "linestyle": "-"},
                    "vel":       {"data": vel_seg, "label": "Vel", "x": x_vals, "linestyle": "-"},
                    "acc":       {"data": acc_seg, "label": "Acc", "x": x_vals, "linestyle": "-"},
                    "jerk":      {"data": jerk_seg, "label": "Jerk", "x": x_vals, "linestyle": "-"},
                    "norm_jerk": {"data": warped_jerk, "label": "Norm Jerk", "x": np.linspace(0, 100, len(warped_jerk)), "linestyle": "--"}
                }
                
                for key in metrics:
                    if key in available_metrics:
                        met = available_metrics[key]
                        label = met["label"] if key not in plotted_labels else None
                        plotted_labels[key] = True
                        ax.plot(met["x"], met["data"], label=label, linestyle=met["linestyle"], alpha=0.7)
            
            # Average duration over files.
            avg_duration = avg_duration / count if count else 0
            ax.set_title(f"Segment {seg_idx+1}\n(Avg duration: {avg_duration:.2f}s)")
            ax.set_xlabel("Samples / %")
            ax.set_ylabel("Signal")
            ax.grid(True)
            ax.legend(fontsize=8)
        
        # Hide unused subplots.
        for j in range(seg_idx + 1, len(axs)):
            axs[j].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    else:
        print("Invalid mode selected. Choose 'single' or 'all'.")

# Example call for option 1 (single trial):
plot_trials_p_v_a_j_Nj_by_location(results, test_windows_1,
                                       subject="07/22/HW", hand="left", trial=1,
                                       fs=200, target_samples=101,
                                       metrics=["pos"],
                                       mode="single",
                                       file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv')

# Example call for option 2 (all trials):
plot_trials_p_v_a_j_Nj_by_location(results, test_windows_1,
                                       subject="07/22/HW", hand="left", trial=1,
                                       fs=200, target_samples=101,
                                       metrics=["vel"],
                                       mode="all")



def plot_tw_comparison_subplots(ldlj_tw1, ldlj_tw3, sparc_tw1, sparc_tw3, subjects, mode='single'):
    """
    Creates a 2x2 subplot grid comparing Test Window 1 vs Test Window 3 for both LDLJ and SPARC.
    Two modes are available:
    
      mode = 'single': plots data for one subject only. The parameter subjects must be a string.
      mode = 'all':    overlays data from multiple subjects. The parameter subjects must be a list of subject identifiers.
    
    For each hand (left and right):
      Left column: LDLJ comparison from ldlj_tw1 and ldlj_tw3.
      Right column: SPARC comparison from sparc_tw1 and sparc_tw3.
    
    Parameters:
        ldlj_tw1 (dict): Dictionary with LDLJ values for Test Window 1. Expected structure:
                         ldlj_tw1['reach_LDLJ'][subject][hand][file_path] -> list of values.
        ldlj_tw3 (dict): Dictionary with LDLJ values for Test Window 3 (same structure).
        sparc_tw1 (dict): Dictionary with SPARC values for Test Window 1.
        sparc_tw3 (dict): Dictionary with SPARC values for Test Window 3.
        subjects (str or list): Subject identifier if mode=='single', or a list of subject identifiers if mode=='all'.
        mode (str): 'single' for one subject, or 'all' for overlaying all subjects.
    """
    import matplotlib.pyplot as plt

    # If mode is 'single', wrap the subject string into a list.
    if mode == 'single':
        subjects = [subjects]

    hands = ['left', 'right']
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    for i, hand in enumerate(hands):
        # LDLJ comparison (left column)
        ax_ldlj = axs[i, 0]
        all_tw1_ldlj = []
        all_tw3_ldlj = []
        for subject in subjects:
            file_paths = list(ldlj_tw1['reach_LDLJ'][subject][hand].keys())
            for fp in file_paths:
                ldlj_1 = ldlj_tw1['reach_LDLJ'][subject][hand][fp]
                ldlj_3 = ldlj_tw3['reach_LDLJ'][subject][hand][fp]
                if mode == 'all':
                    ax_ldlj.scatter(ldlj_1, ldlj_3, alpha=0.5, label=subject)
                else:
                    ax_ldlj.scatter(ldlj_1, ldlj_3, alpha=0.5)
                all_tw1_ldlj.extend(ldlj_1)
                all_tw3_ldlj.extend(ldlj_3)
        corr_ldlj, p_ldlj = spearmanr(all_tw1_ldlj, all_tw3_ldlj)
        ax_ldlj.set_xlabel('reach_LDLJ (TW1)')
        ax_ldlj.set_ylabel('reach_LDLJ (TW3)')
        title_subject = subjects[0] if mode == 'single' else "All Subjects"
        ax_ldlj.set_title(f"{title_subject} {hand} LDLJ\nSpearman r: {corr_ldlj:.2f} (p = {p_ldlj:.3f})")
        ax_ldlj.grid(True)

        # SPARC comparison (right column)
        ax_sparc = axs[i, 1]
        all_tw1_sparc = []
        all_tw3_sparc = []
        for subject in subjects:
            file_paths = list(sparc_tw1[subject][hand].keys())
            for fp in file_paths:
                sparc_1 = sparc_tw1[subject][hand][fp]
                sparc_3 = sparc_tw3[subject][hand][fp]
                if mode == 'all':
                    ax_sparc.scatter(sparc_1, sparc_3, alpha=0.5, label=subject)
                else:
                    ax_sparc.scatter(sparc_1, sparc_3, alpha=0.5)
                all_tw1_sparc.extend(sparc_1)
                all_tw3_sparc.extend(sparc_3)
        corr_sparc, p_sparc = spearmanr(all_tw1_sparc, all_tw3_sparc)
        ax_sparc.set_xlabel('SPARC (TW1)')
        ax_sparc.set_ylabel('SPARC (TW3)')
        ax_sparc.set_title(f"{title_subject} {hand} SPARC\nSpearman r: {corr_sparc:.2f} (p = {p_sparc:.3f})")
        ax_sparc.grid(True)

    plt.tight_layout()
    plt.show()

# Example usage:
# To plot one subject only:
plot_tw_comparison_subplots(reach_TW_metrics_test_windows_1,
                            reach_TW_metrics_test_windows_3,
                            reach_sparc_test_windows_1_Normalizing,
                            reach_sparc_test_windows_3_Normalizing,
                            subjects="07/22/HW", mode='single')

# To overlay data for all subjects:
plot_tw_comparison_subplots(reach_TW_metrics_test_windows_1,
                            reach_TW_metrics_test_windows_3,
                            reach_sparc_test_windows_1_Normalizing,
                            reach_sparc_test_windows_3_Normalizing,
                            subjects=All_dates, mode='all')


# # # # -------------------------------------------------------------------------------------------------------------------
# PART 3: Combine Metrics and Save Results
# --- PROCESS AND SAVE COMBINED METRICS [DURATIONS, SPARC, LDLJ, AND DISTANCE, CALCULATED SPEED AND ACCURACY FOR ALL DATES]---
utils5.process_and_save_combined_metrics(Block_Distance, reach_metrics, reach_sparc_test_windows_1_Normalizing, reach_TW_metrics_test_windows_1, All_dates, DataProcess_folder)

utils5.process_and_save_combined_metrics_acorss_TWs(Block_Distance, reach_metrics,
                                                    reach_sparc_test_windows_1_Normalizing, reach_TW_metrics_test_windows_1,
                                                    reach_sparc_test_windows_3_Normalizing, reach_TW_metrics_test_windows_3,
                                                    All_dates, DataProcess_folder)
# # -------------------------------------------------------------------------------------------------------------------
# --- LOAD ALL COMBINED METRICS PER SUBJECT FROM PICKLE FILE ---
all_combined_metrics = utils5.load_selected_subject_results(All_dates, DataProcess_folder)
# Swap and rename metrics for consistency
all_combined_metrics = utils5.swap_and_rename_metrics(all_combined_metrics, All_dates)
# Filter all_combined_metrics based on distance and count NaNs
filtered_metrics, total_nan, Nan_counts_per_subject_per_hand, Nan_counts_per_index = utils5.filter_combined_metrics_and_count_nan(all_combined_metrics)
# Update filtered metrics and count NaN replacements based on distance and duration thresholds: distance_threshold=15, duration_threshold=1.6
updated_metrics, Cutoff_counts_per_subject_per_hand, Cutoff_counts_per_index, total_nan_per_subject_hand = utils5.update_filtered_metrics_and_count(filtered_metrics)

# # ------------------------------------------------------------------------------------------------------------------- 
all_combined_metrics_acorss_TWs = utils5.load_selected_subject_results_acorss_TWs(All_dates, DataProcess_folder)
all_combined_metrics_acorss_TWs = utils5.swap_and_rename_metrics(all_combined_metrics_acorss_TWs, All_dates)
filtered_metrics_acorss_TWs, total_nan_acorss_TWs, Nan_counts_per_subject_per_hand_acorss_TWs, Nan_counts_per_index_acorss_TWs = utils5.filter_combined_metrics_and_count_nan(all_combined_metrics_acorss_TWs)
updated_metrics_acorss_TWs, Cutoff_counts_per_subject_per_hand_acorss_TWs, Cutoff_counts_per_index_acorss_TWs, total_nan_per_subject_hand_acorss_TWs = utils5.update_filtered_metrics_and_count(filtered_metrics_acorss_TWs)
# # -------------------------------------------------------------------------------------------------------------------
def plot_histograms(filtered_metrics, sd_multiplier=5, overlay_median=True, overlay_sd=True, overlay_iqr=True):
    """
    Plots histograms for all durations and distances stored in filtered_metrics.
    Also finds reach indices where duration > 1.6 or distance > 15 and returns them separately.
    
    Parameters:
      filtered_metrics (dict): Dictionary containing 'durations' and 'distance' per subject and hand.
      sd_multiplier (int or float): The factor to multiply standard deviation for SD overlay lines.
      overlay_median (bool): Whether to overlay the median for the plots.
      overlay_sd (bool): Whether to overlay lines at median ± (sd_multiplier * standard deviation).
      overlay_iqr (bool): Whether to overlay the Q1 and Q3 boundaries (IQR boundaries).
      
    Returns:
      dict: A dictionary with two keys "duration" and "distance". Each key maps to a dictionary of outlier reach indices
            where duration > 1.6 or distance > 15 respectively.
            Structure: {subject: {hand: {trial: [reach_indices, ...], ...}, ...}, ...}
    """
    all_durations = []
    all_distances = []

    for subject in filtered_metrics:
        for hand in filtered_metrics[subject]:
            if 'durations' in filtered_metrics[subject][hand] and 'distance' in filtered_metrics[subject][hand]:
                durations = filtered_metrics[subject][hand]['durations']
                distances = filtered_metrics[subject][hand]['distance']
                for trial in durations:
                    all_durations.extend(durations[trial])
                for trial in distances:
                    all_distances.extend(distances[trial])
    
    # Filter out NaN values
    all_durations_clean = [d for d in all_durations if not np.isnan(d)]
    all_distances_clean = [d for d in all_distances if not np.isnan(d)]

    # Calculate cutoff values for durations and distances so that 99.5% of data is included.
    lower_duration = np.percentile(all_durations_clean, 0)
    upper_duration = np.percentile(all_durations_clean, 99.5)
    lower_distance = np.percentile(all_distances_clean, 0)
    upper_distance = np.percentile(all_distances_clean, 99.5)

    print(f"99.5%  of Duration: {lower_duration:.2f} - {upper_duration:.2f}")
    print(f"99.5% of Distance: {lower_distance:.2f} - {upper_distance:.2f}")

    max_duration = np.max(all_durations_clean)
    min_duration = np.min(all_durations_clean)
    max_distance = np.max(all_distances_clean)
    min_distance = np.min(all_distances_clean)

    
    # Find the location (subject, hand, trial, reach index) for the maximum duration and distance
    loc_max_duration = None
    loc_max_distance = None
    for subject in filtered_metrics:
        for hand in filtered_metrics[subject]:
            if 'durations' in filtered_metrics[subject][hand] and 'distance' in filtered_metrics[subject][hand]:
                durations = filtered_metrics[subject][hand]['durations']
                distances = filtered_metrics[subject][hand]['distance']
                for trial in durations:
                    for idx, d in enumerate(durations[trial]):
                        if d == max_duration:
                            loc_max_duration = (subject, hand, trial, idx + 1)
                            break
                    if loc_max_duration:
                        break
                for trial in distances:
                    for idx, d in enumerate(distances[trial]):
                        if d == max_distance:
                            loc_max_distance = (subject, hand, trial, idx + 1)
                            break
                    if loc_max_distance:
                        break
        if loc_max_duration and loc_max_distance:
            break

    print(f"Maximum Duration: {max_duration:.2f}")
    if loc_max_duration:
        print(f"Located at: Subject = {loc_max_duration[0]}, Hand = {loc_max_duration[1]}, Trial = {loc_max_duration[2]}, Reach Index = {loc_max_duration[3]}")
    print(f"Maximum Distance: {max_distance:.2f}")
    if loc_max_distance:
        print(f"Located at: Subject = {loc_max_distance[0]}, Hand = {loc_max_distance[1]}, Trial = {loc_max_distance[2]}, Reach Index = {loc_max_distance[3]}")

    median_duration = np.median(all_durations_clean)
    median_distance = np.median(all_distances_clean)

    # Find the location (subject, hand, trial, reach index) for the median duration and distance
    loc_median_duration = None
    loc_median_distance = None
    for subject in filtered_metrics:
        for hand in filtered_metrics[subject]:
            if 'durations' in filtered_metrics[subject][hand] and 'distance' in filtered_metrics[subject][hand]:
                durations = filtered_metrics[subject][hand]['durations']
                distances = filtered_metrics[subject][hand]['distance']
                for trial in durations:
                    for idx, d in enumerate(durations[trial]):
                        if d == median_duration:
                            loc_median_duration = (subject, hand, trial, idx + 1)
                            break
                    if loc_median_duration:
                        break
                for trial in distances:
                    for idx, d in enumerate(distances[trial]):
                        if d == median_distance:
                            loc_median_distance = (subject, hand, trial, idx + 1)
                            break
                    if loc_median_distance:
                        break
        if loc_median_duration and loc_median_distance:
            break

    print(f"Median Duration: {median_duration:.2f}")
    if loc_median_duration:
        print(f"Located at: Subject = {loc_median_duration[0]}, Hand = {loc_median_duration[1]}, Trial = {loc_median_duration[2]}, Reach Index = {loc_median_duration[3]}")
    print(f"Median Distance: {median_distance:.2f}")
    if loc_median_distance:
        print(f"Located at: Subject = {loc_median_distance[0]}, Hand = {loc_median_distance[1]}, Trial = {loc_median_distance[2]}, Reach Index = {loc_median_distance[3]}")

    # Find the location (subject, hand, trial, reach index) for the maximum duration
    # and find all locations where distance > 25
    loc_distances_over_25 = []

    for subject in filtered_metrics:
        for hand in filtered_metrics[subject]:
            # Process distances to find all locations where distance > 25
            if 'distance' in filtered_metrics[subject][hand]:
                distances = filtered_metrics[subject][hand]['distance']
                for trial in distances:
                    for idx, d in enumerate(distances[trial]):
                        if d > 25:
                            loc_distances_over_25.append((subject, hand, trial, idx + 1))
    
    print("Locations where Distance > 25:")
    for loc in loc_distances_over_25:
        print(f"Subject = {loc[0]}, Hand = {loc[1]}, Trial = {loc[2]}, Reach Index = {loc[3]}, Distance = {filtered_metrics[loc[0]][loc[1]]['distance'][loc[2]][loc[3]-1]:.2f}")



    # Compute statistics for durations
    std_duration = np.std(all_durations_clean)
    q1_duration = np.percentile(all_durations_clean, 25)
    q3_duration = np.percentile(all_durations_clean, 75)
    
    # Compute statistics for distances
    std_distance = np.std(all_distances_clean)
    q1_distance = np.percentile(all_distances_clean, 25)
    q3_distance = np.percentile(all_distances_clean, 75)
    
    plt.figure(figsize=(12, 5))
    
    # Plot histogram for durations
    plt.subplot(1, 2, 1)
    plt.hist(all_durations_clean, bins=60, color='skyblue', edgecolor='black')
    plt.xlabel('Duration')
    plt.ylabel('Frequency')
    plt.xlim(min_duration, max_duration)
    plt.title('Histogram of All Durations')
    
    if overlay_median:
        plt.axvline(median_duration, color='red', linestyle='-', linewidth=2, label=f'Median: {median_duration:.2f}')
    if overlay_sd:
        plt.axvline(median_duration - sd_multiplier * std_duration, color='green', linestyle='--', linewidth=2,
                    label=f'-{sd_multiplier} SD: {median_duration - sd_multiplier * std_duration:.2f}')
        plt.axvline(median_duration + sd_multiplier * std_duration, color='green', linestyle='--', linewidth=2,
                    label=f'+{sd_multiplier} SD: {median_duration + sd_multiplier * std_duration:.2f}')
    if overlay_iqr:
        plt.axvline(q1_duration, color='purple', linestyle=':', linewidth=2, label=f'Q1: {q1_duration:.2f}')
        plt.axvline(q3_duration, color='purple', linestyle=':', linewidth=2, label=f'Q3: {q3_duration:.2f}')
    plt.legend()
    
    # Plot histogram for distances
    plt.subplot(1, 2, 2)
    plt.hist(all_distances_clean, bins=60, color='salmon', edgecolor='black')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.xlim(min_distance, max_distance)
    plt.title('Histogram of All Distances')
    
    if overlay_median:
        plt.axvline(median_distance, color='red', linestyle='-', linewidth=2, label=f'Median: {median_distance:.2f}')
    if overlay_sd:
        plt.axvline(median_distance - sd_multiplier * std_distance, color='green', linestyle='--', linewidth=2,
                    label=f'-{sd_multiplier} SD: {median_distance - sd_multiplier * std_distance:.2f}')
        plt.axvline(median_distance + sd_multiplier * std_distance, color='green', linestyle='--', linewidth=2,
                    label=f'+{sd_multiplier} SD: {median_distance + sd_multiplier * std_distance:.2f}')
    if overlay_iqr:
        plt.axvline(q1_distance, color='purple', linestyle=':', linewidth=2, label=f'Q1: {q1_distance:.2f}')
        plt.axvline(q3_distance, color='purple', linestyle=':', linewidth=2, label=f'Q3: {q3_distance:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Find reach indices where duration > 1.6 or distance > 15, and save separately.
    outlier_reaches_duration = {}
    outlier_reaches_distance = {}
    for subject in filtered_metrics:
        for hand in filtered_metrics[subject]:
            if 'durations' in filtered_metrics[subject][hand] and 'distance' in filtered_metrics[subject][hand]:
                durations = filtered_metrics[subject][hand]['durations']
                distances = filtered_metrics[subject][hand]['distance']
                for trial in durations:
                    duration_indices = []
                    distance_indices = []
                    for idx, d in enumerate(durations[trial]):
                        if d > 1.8:
                            duration_indices.append(idx + 1)
                        if distances[trial][idx] > 20:
                            distance_indices.append(idx + 1)
                    if duration_indices:
                        if subject not in outlier_reaches_duration:
                            outlier_reaches_duration[subject] = {}
                        if hand not in outlier_reaches_duration[subject]:
                            outlier_reaches_duration[subject][hand] = {}
                        outlier_reaches_duration[subject][hand][trial] = duration_indices
                    if distance_indices:
                        if subject not in outlier_reaches_distance:
                            outlier_reaches_distance[subject] = {}
                        if hand not in outlier_reaches_distance[subject]:
                            outlier_reaches_distance[subject][hand] = {}
                        outlier_reaches_distance[subject][hand][trial] = distance_indices

    # Return the outlier reaches separately for duration and distance.
    return {"duration": outlier_reaches_duration, "distance": outlier_reaches_distance}

# Example call to plot histograms and get outlier reach indices
outliers = plot_histograms(filtered_metrics, sd_multiplier=4, overlay_median=True, overlay_sd=False, overlay_iqr=True)
# -------------------------------------------------------------------------------------------------------------------
def create_metrics_dataframe(updated_metrics_acorss_TWs):
    """
    Creates a DataFrame from the combined metrics stored in the updated_metrics_acorss_TWs dictionary.
    It loops over each subject, hand, trial, and location (assumed to be 16) and collects the available
    metrics into a list of dictionaries. If a 'durations' value is NaN, a message is printed and that record
    is skipped.

    Parameters:
        updated_metrics_acorss_TWs (dict): A nested dictionary that contains metrics per subject, hand, and trial.

    Returns:
        df (DataFrame): A pandas DataFrame with columns:
            ['Subject', 'Hand', 'Trial', 'Location', 'durations', 
            'cartesian_distances', 'path_distances', 'v_peaks', 
            'TW1_acc_peaks', 'TW1_jerk_peaks', 'TW1_LDLJ', 
            'TW3_acc_peaks', 'TW3_jerk_peaks', 'TW3_LDLJ', 
            'TW1_sparc', 'TW3_sparc', 'distance']
    """
    rows = []
    for subject in updated_metrics_acorss_TWs:
        for hand in updated_metrics_acorss_TWs[subject]:
            for trial in updated_metrics_acorss_TWs[subject][hand]['durations']:
                for loc in range(16):
                    duration_val = updated_metrics_acorss_TWs[subject][hand]['durations'][trial][loc]
                    if not np.isnan(duration_val):
                        rows.append({
                            'Subject': subject,
                            'Hand': hand,
                            'Trial': trial,
                            'Location': loc + 1,  # Location index (1 to 16)
                            'durations': duration_val,
                            'cartesian_distances': updated_metrics_acorss_TWs[subject][hand]['cartesian_distances'][trial][loc],
                            'path_distances': updated_metrics_acorss_TWs[subject][hand]['path_distances'][trial][loc],
                            'v_peaks': updated_metrics_acorss_TWs[subject][hand]['v_peaks'][trial][loc],
                            'TW1_acc_peaks': updated_metrics_acorss_TWs[subject][hand]['TW1_acc_peaks'][trial][loc],
                            'TW1_jerk_peaks': updated_metrics_acorss_TWs[subject][hand]['TW1_jerk_peaks'][trial][loc],
                            'TW1_LDLJ': updated_metrics_acorss_TWs[subject][hand]['TW1_LDLJ'][trial][loc],
                            'TW3_acc_peaks': updated_metrics_acorss_TWs[subject][hand]['TW3_acc_peaks'][trial][loc],
                            'TW3_jerk_peaks': updated_metrics_acorss_TWs[subject][hand]['TW3_jerk_peaks'][trial][loc],
                            'TW3_LDLJ': updated_metrics_acorss_TWs[subject][hand]['TW3_LDLJ'][trial][loc],
                            'TW1_sparc': updated_metrics_acorss_TWs[subject][hand]['TW1_sparc'][trial][loc],
                            'TW3_sparc': updated_metrics_acorss_TWs[subject][hand]['TW3_sparc'][trial][loc],
                            'distance': updated_metrics_acorss_TWs[subject][hand]['distance'][trial][loc]
                        })
                    else:
                        print(f"Skipping NaN for Subject: {subject}, Hand: {hand}, Trial: {trial}, Location: {loc+1}")
            
    df = pd.DataFrame(rows)
    print("Data sample:")
    print(df.head())
    return df

# Create DataFrame from updated metrics across test windows
df = create_metrics_dataframe(updated_metrics_acorss_TWs)
# -------------------------------------------------------------------------------------------------------------------

# Run ANOVA 
model = smf.ols('durations ~ C(Hand) + TW1_acc_peaks + TW1_jerk_peaks + TW1_LDLJ + TW1_sparc + C(Location)', data=df).fit()
model = smf.ols('distance ~ C(Hand) + TW1_acc_peaks + TW1_jerk_peaks + TW1_LDLJ + TW1_sparc + C(Location)', data=df).fit()
model = smf.ols('durations ~ C(Hand) + TW1_acc_peaks + TW1_jerk_peaks + TW1_LDLJ + TW1_sparc + + TW3_acc_peaks + TW3_jerk_peaks + TW3_LDLJ + TW3_sparc + cartesian_distances + C(Location) + v_peaks', data=df).fit()

anova_table = sm.stats.anova_lm(model, typ=2)
print("\nANOVA results:")
print(anova_table)

# Mixed-effects model with Subject as a random effect
model_mixed = smf.mixedlm("durations ~ C(Hand) + TW1_acc_peaks + TW1_jerk_peaks + TW1_LDLJ + TW1_sparc + C(Location)", df, groups=df["Subject"])
model_mixed = smf.mixedlm("distance ~ C(Hand) + TW1_acc_peaks + TW1_jerk_peaks + TW1_LDLJ + TW1_sparc + C(Location)", df, groups=df["Subject"])
model_mixed = smf.mixedlm("durations ~ C(Hand) + TW1_acc_peaks + TW1_jerk_peaks + TW1_LDLJ + TW1_sparc + + TW3_acc_peaks + TW3_jerk_peaks + TW3_LDLJ + TW3_sparc + cartesian_distances + C(Location) + v_peaks", df, groups=df["Subject"])

result_mixed = model_mixed.fit()
print("\nMixed-effects results:")
print(result_mixed.summary())


# -------------------------------------------------------------------------------------------------------------------
def plot_metric_boxplots(updated_metrics_acorss_TWs, metrics=["TW3_LDLJ", "TW3_sparc", "durations", "distance"], use_median=False):
    import matplotlib.pyplot as plt

    # Set up the subplot grid.
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    # Loop through each metric to compute stats and plot the boxplots.
    for i, metric in enumerate(metrics):
        # Compute statistics for each subject and hand for the current metric.
        stats_results = {}
        for subject, subject_data in updated_metrics_acorss_TWs.items():
            stats_results[subject] = {}
            for hand, hand_data in subject_data.items():
                if metric in hand_data:
                    all_vals = []
                    for trial, trial_vals in hand_data[metric].items():
                        all_vals.extend(trial_vals)
                    all_vals = np.array(all_vals)
                    if all_vals.size > 0:
                        if use_median:
                            central_val = np.nanmedian(all_vals)
                            # Compute interquartile range (IQR)
                            q1 = np.nanpercentile(all_vals, 25)
                            q3 = np.nanpercentile(all_vals, 75)
                            spread = q3 - q1
                        else:
                            central_val = np.nanmean(all_vals)
                            spread = np.nanstd(all_vals)
                    else:
                        central_val = np.nan
                        spread = np.nan
                    stats_results[subject][hand] = {'central': central_val, 'spread': spread}

        # Prepare a DataFrame from the computed results.
        data = []
        for subject, hands in stats_results.items():
            for hand, stats in hands.items():
                data.append({
                    "subject": subject,
                    "hand": hand,
                    "central": stats["central"],
                    "spread": stats["spread"]
                })
        df = pd.DataFrame(data)

        # Compute Wilcoxon signed-rank test for subjects with both hands.
        pivot_df = df.pivot(index='subject', columns='hand', values='central')
        pivot_df = pivot_df.dropna(subset=['non_dominant', 'dominant'])
        if not pivot_df.empty:
            stat, p = wilcoxon(pivot_df['non_dominant'], pivot_df['dominant'])
            test_title = f"Wilcoxon stat = {stat:.2f}, p-value = {p:.3f}"
        else:
            test_title = "Not enough paired data for Wilcoxon test."

        # Select the corresponding axis.
        ax = axs[i]
        order = ["non_dominant", "dominant"]
        sns.boxplot(x="hand", y="central", data=df, palette="Set2", order=order, ax=ax)
        sns.swarmplot(x="hand", y="central", data=df, color="black", size=5, alpha=0.8, order=order, ax=ax)

        # Update title and axis labels depending on whether we're using median or mean.
        label = "Median" if use_median else "Mean"
        ax.set_title(f"Box Plot of {metric.upper()} {label} Values by Hand\n{test_title}")
        ax.set_xlabel("Hand")
        ax.set_ylabel(f"{label} {metric.upper()}")

        # Draw a dashed line connecting the two hands for each subject.
        for subject in pivot_df.index:
            nd_value = pivot_df.loc[subject, "non_dominant"]
            d_value = pivot_df.loc[subject, "dominant"]
            ax.plot([0, 1], [nd_value, d_value], color="gray", linestyle="--", linewidth=1, alpha=0.7)

    plt.tight_layout()
    plt.show()

# Call the function to plot boxplots for all four metrics in a 2x2 grid,
# using the median and IQR instead of mean and std.
plot_metric_boxplots(updated_metrics_acorss_TWs, metrics=["TW3_LDLJ", "TW3_sparc", "durations", "distance"], use_median=True)


def plot_scatter_correlations(updated_metrics_acorss_TWs, use_zscore=False, selected_indep=None):
    """
    For each hand, plots scatter plots for the selected independent metrics (ldlj and/or sparc)
    versus durations and distance from updated_metrics_acorss_TWs.

    Parameters:
        updated_metrics_acorss_TWs (dict): Dictionary containing combined metrics across test windows.
        use_zscore (bool): If True, the data is z-scored before plotting.
        selected_indep (list or None): List of independent metric keys to plot. If None, defaults to ['ldlj', 'sparc'].

    For each selected independent metric, plots:
      - independent metric vs durations
      - independent metric vs distance

    Computes the Spearman correlation for each pairing and displays the results on the plots.
    """
    # Default selection if not provided.
    if selected_indep is None:
        selected_indep = ['ldlj', 'sparc']
    
    # Create pairings for each selected independent metric.
    pairings = []
    for metric in selected_indep:
        pairings.append((metric, 'durations'))
        pairings.append((metric, 'distance'))
    
    hands = ['non_dominant', 'dominant']
    
    for hand in hands:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()
        for i, (metric_x, metric_y) in enumerate(pairings):
            x_vals = []
            y_vals = []
            # Collect data from all subjects and trials from updated_metrics_acorss_TWs for the given hand.
            for subject, subject_data in updated_metrics_acorss_TWs.items():
                if hand in subject_data:
                    hand_data = subject_data[hand]
                    if metric_x in hand_data and metric_y in hand_data:
                        for trial, vals_x in hand_data[metric_x].items():
                            vals_y = hand_data[metric_y].get(trial, [])
                            # Ensure both lists are of equal length.
                            if len(vals_x) == len(vals_y):
                                x_vals.extend(vals_x)
                                y_vals.extend(vals_y)
            x_arr = np.array(x_vals)
            y_arr = np.array(y_vals)
            # Remove NaN values.
            valid = ~np.isnan(x_arr) & ~np.isnan(y_arr)
            x_arr = x_arr[valid]
            y_arr = y_arr[valid]
            if len(x_arr) == 0:
                continue
            # Optionally z-score the arrays.
            if use_zscore:
                x_arr = zscore(x_arr)
                y_arr = zscore(y_arr)
                xlabel = metric_x.upper() + " (Z-scored)"
                ylabel = metric_y.upper() + " (Z-scored)"
            else:
                xlabel = metric_x.upper()
                ylabel = metric_y.upper()

            corr, p_val = spearmanr(x_arr, y_arr)
            ax = axs[i]
            ax.scatter(x_arr, y_arr, color='blue', alpha=0.6)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{hand.capitalize()} {metric_x.upper()} vs {metric_y.upper()}\n"
                         f"Spearman: {corr:.2f}, p: {p_val:.3f}")
            ax.grid(True)
        plt.tight_layout()
        plt.show()

# Plot using raw values / z-scored
plot_scatter_correlations(updated_metrics_acorss_TWs, use_zscore=False, selected_indep=['TW3_LDLJ', 'TW3_sparc'])
plot_scatter_correlations(updated_metrics_acorss_TWs, use_zscore=False, selected_indep=['TW1_LDLJ', 'TW1_sparc'])


# -------------------------------------------------------------------------------------------------------------------
# PART 4: Data Analysis and Visualization
# -------------------------------------------------------------------------------------------------------------------
# Do reach types that are faster on average also tend to be less accurate on average?
result_Check_SAT_in_trials_mean_median_of_reach_indices = utils6.Check_SAT_in_trials_mean_median_of_reach_indices(updated_metrics_acorss_TWs, '07/22/HW', 'durations', 'distance', stat_type="median")

# Within one reach location, is there still a speed–accuracy trade-off across repetitions?
utils6.scatter_plot_duration_distance_by_choice(updated_metrics_acorss_TWs, overlay_hands=False, selected_subjects=['07/22/HW'], special_indices=[0], show_hyperbolic_fit=False, color_mode="uniform", show_median_overlay=False)
_, corr_results, result_Check_SAT_in_reach_indices_by_hand_by_subject, heatmap_medians = utils6.Check_SAT_in_reach_indices_by_index_or_subject(updated_metrics_acorss_TWs, '07/22/HW', grouping="hand_by_subject", hyperbolic=False)

utils6.heatmap_spearman_correlation_reach_indices_signifcant(corr_results, hand="both", simplified=False, return_medians=True, overlay_median=True)
# -------------------------------------------------------------------------------------------------------------------
# Specify the path to the pickle file
pickle_file = "/Users/yilinwu/Desktop/honours data/DataProcess/All_Subject_tBBTs_errors.pkl"
cmap_choice = LinearSegmentedColormap.from_list("GreenWhiteBlue", ["green", "white", "blue"], N=256)

# Load the pickle file without using a function
with open(pickle_file, "rb") as file:
    All_Subject_tBBTs_errors = pickle.load(file)

### Separate data by hand and plot 3D scatter plots for each hand
utils4.plot_xy_density_for_each_hand(All_Subject_tBBTs_errors, cmap_choice)

### Combine data from both hands and plot combined density heatmap with grid markers
utils4.plot_combined_xy_density(All_Subject_tBBTs_errors, cmap_choice)

### Combine 16 blocks data into one for each subject and hand
Combine_blocks = utils4.Combine_16_blocks(All_Subject_tBBTs_errors)

### Plot left and right hand 16 blocks as one density histograms with 0.0 at the center of the view
utils4.plot_left_right_hand_new_coordinates_density(Combine_blocks, cmap_choice)

### Plot left and right hand 16 blocks as one polar histograms (rose diagrams)
utils4.plot_left_right_hand_polar_histogram(Combine_blocks, cmap_choice)


subject = '07/22/HW'
hand = 'right'
target_file = '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT63.csv'

# Get all file keys in the trial dictionary (results[subject][hand][1])
file_keys = list(results[subject][hand][1].keys())

# Find the index of the target file in the file keys list
target_index = file_keys.index(target_file)
print("Index of target file:", target_index)

### Plot p3_box2 and p3_block2 coordinates in a 3D scatter plot for a specific subject, hand, and trial
utils4.plot_p3_coordinates(All_Subject_tBBTs_errors, subject='07/22/HW', hand='right', trial_index=31)

### Plot hand trajectory with velocity-coded coloring and highlighted segments
utils4.plot_trajectory(results, subject='07/22/HW', hand='right', trial=1,
                file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT63.csv',
                overlay_trial=0, velocity_segment_only=True, plot_mode='segment')

### Combine hand trajectory and error coordinates in a single 3D plot
utils4.combined_plot_trajectory_and_errors(results, All_Subject_tBBTs_errors,
                                      subject='07/22/HW', hand='right',
                                      trial=1, trial_index=31,
                                      file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT63.csv',
                                      overlay_trial=0, velocity_segment_only=True, plot_mode='segment')
# # -------------------------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------------------------
mean_stats, median_stats = utils6.calculate_trials_mean_median_of_reach_indices(updated_metrics, 'durations', 'distance')

def plot_duration_vs_distance(mean_stats, reach_index=1, hand='non_dominant'):
    """
    Create a scatter plot for average duration vs average distance for a given reach index and hand.

    Parameters:
        mean_stats (dict): Dictionary containing mean statistics for subjects.
        reach_index (int): The index of the reach to select (default is 1).
        hand (str): The hand to select from the stats dictionary (default is 'non_dominant').
    """
    subjects = []
    avg_durations = []
    avg_distances = []

    for subject, stats in mean_stats.items():
        try:
            duration = stats[hand][reach_index]['avg_duration']
            distance = stats[hand][reach_index]['avg_distance']
            subjects.append(subject)
            avg_durations.append(duration)
            avg_distances.append(distance)
        except KeyError:
            # Skip subjects lacking the required keys
            continue

    plt.figure(figsize=(8, 6))
    plt.scatter(avg_durations, avg_distances, color='blue')

    # Annotate each point with subject labels
    for i, subject in enumerate(subjects):
        plt.annotate(subject, (avg_durations[i], avg_distances[i]),
                     textcoords="offset points", xytext=(5,5), ha='center')

    plt.title(f"{hand.capitalize()} Hand: Avg Duration vs Avg Distance (Reach Index {reach_index})")
    plt.xlabel("Avg Duration")
    plt.ylabel("Avg Distance")
    plt.grid(True)
    plt.show()

plot_duration_vs_distance(mean_stats, reach_index=1, hand='non_dominant')


def Get_SAT_Z_MotorAcuity(stats_data, hand='non_dominant', plot_type='raw', return_distances=False):
    """
    Plots scatter subplots for 16 reach indices using data from stats_data.
    
    Parameters:
        stats_data (dict): Dictionary with average or median duration and distance per subject.
        hand (str): Hand to use from the stats dictionary (default is 'non_dominant').
        plot_type (str): One of the following options:
            'raw'      - Plot raw average/median duration vs avg/median distance.
            'zscore'   - Plot z-scored data with red lines at zero and custom axis labels.
            'distance' - Plot z-scored data with a 45° diagonal line and overlay lines showing signed perpendicular distance.
        return_distances (bool): If True and plot_type=='distance', returns a dict with computed signed perpendicular distances.
        
    Returns:
        If plot_type == 'distance' and return_distances is True, returns a dict where each key is the subject and
        each value is another dict with reach index (1-16) as keys and their corresponding signed perpendicular distance as float.
        Otherwise, returns None.
    """
    import matplotlib.pyplot as plt

    def annotate_axis(ax):
        # Add qualitative annotations to the axis without overlapping the x or y labels.
        ax.text(0, -0.25, "Good", transform=ax.transAxes, ha='center', va='center', color='green')
        ax.text(1, -0.25, "Bad", transform=ax.transAxes, ha='center', va='center', color='red')
        ax.text(-0.25, 0, "Good", transform=ax.transAxes, ha='center', va='center', rotation=90, color='green')
        ax.text(-0.25, 1, "Bad", transform=ax.transAxes, ha='center', va='center', rotation=90, color='red')

    def zscore(arr):
        # Compute z-scores for an array.
        if arr.size:
            m = arr.mean()
            s = arr.std()
            if s != 0:
                return (arr - m) / s
        return arr

    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    axs = axs.flatten()

    # To store distances if needed for 'distance' plot_type.
    distances_dict = {}

    for reach_index in range(16):
        subjects = []
        avg_durations = []
        avg_distances = []
        
        # Gather data from each subject.
        for subject, stats in stats_data.items():
            try:
                data = stats[hand][reach_index]
                # Use avg keys if available; otherwise, use median keys.
                if 'avg_duration' in data and 'avg_distance' in data:
                    duration = data['avg_duration']
                    distance = data['avg_distance']
                else:
                    duration = data['median_duration']
                    distance = data['median_distance']
                subjects.append(subject)
                avg_durations.append(duration)
                avg_distances.append(distance)
            except KeyError:
                continue
        
        ax = axs[reach_index]
        ax.set_title(f"{hand.capitalize()}: Index {reach_index + 1}")
        
        if plot_type == 'raw':
            ax.scatter(avg_durations, avg_distances, color='blue')
            ax.set_xlabel("Duration")
            ax.set_ylabel("Distance")
            annotate_axis(ax)
            ax.grid(True)
        
        elif plot_type in ['zscore', 'distance']:
            durations_arr = np.array(avg_durations)
            distances_arr = np.array(avg_distances)
            z_durations = zscore(durations_arr)
            z_distances = zscore(distances_arr)
            
            ax.scatter(z_durations, z_distances, color='blue')
            ax.set_xlabel("Z-scored Duration")
            ax.set_ylabel("Z-scored Distance")
            annotate_axis(ax)
            
            if plot_type == 'zscore':
                ax.axhline(y=0, color='red')
                ax.axvline(x=0, color='red')
            
            elif plot_type == 'distance':
                if z_durations.size and z_distances.size:
                    min_lim = min(z_durations.min(), z_distances.min())
                    max_lim = max(z_durations.max(), z_distances.max())
                else:
                    min_lim, max_lim = -1, 1
                
                x_vals = np.linspace(min_lim, max_lim, 100)
                ax.plot(x_vals, x_vals, color='green', linestyle='--', label='45° line')
                
                # Compute signed perpendicular distances and plot projection lines.
                for i, (x, y) in enumerate(zip(z_durations, z_distances)):
                    # Compute signed perpendicular distance to the line x=y.
                    signed_distance = (y - x) / math.sqrt(2)
                    subj = subjects[i]
                    if subj not in distances_dict:
                        distances_dict[subj] = {}
                    distances_dict[subj][reach_index + 1] = signed_distance
                    
                    proj = ((x + y) / 2, (x + y) / 2)
                    ax.plot([x, proj[0]], [y, proj[1]], color='purple', linestyle=':', linewidth=0.8)
                
                ax.axhline(y=0, color='red')
                ax.axvline(x=0, color='red')
            ax.grid(True)
        else:
            raise ValueError("Invalid plot_type. Options are 'raw', 'zscore', 'distance'.")
    
    plt.tight_layout()
    plt.show()
    
    if plot_type == 'distance' and return_distances:
        return distances_dict

# Plot raw data using mean_stats.
Get_SAT_Z_MotorAcuity(mean_stats, hand='non_dominant', plot_type='raw')

# Plot z-scored data using mean_stats.
Get_SAT_Z_MotorAcuity(mean_stats, hand='non_dominant', plot_type='zscore')

# Plot distance data and get computed MotorAcuity using mean_stats.
MotorAcuity = Get_SAT_Z_MotorAcuity(mean_stats, hand='non_dominant', plot_type='distance', return_distances=True)

Get_SAT_Z_MotorAcuity(median_stats, hand='non_dominant', plot_type='raw')

# zscore acorss subjects for median_stats.
MotorAcuity_Mean = {}
# Plot distance data and get computed MotorAcuity.
MotorAcuity_Mean['non_dominant'] = Get_SAT_Z_MotorAcuity(mean_stats, hand='non_dominant', plot_type='distance', return_distances=True)
MotorAcuity_Mean['dominant'] = Get_SAT_Z_MotorAcuity(mean_stats, hand='dominant', plot_type='distance', return_distances=True)

MotorAcuity_Median = {}
# Plot distance data and get computed MotorAcuity.
MotorAcuity_Median['non_dominant'] = Get_SAT_Z_MotorAcuity(median_stats, hand='non_dominant', plot_type='distance', return_distances=True)
MotorAcuity_Median['dominant'] = Get_SAT_Z_MotorAcuity(median_stats, hand='dominant', plot_type='distance', return_distances=True)

## -------------------------------------------------------------------------------------------------------------------
# Global dictionary to store the z-scored data.
updated_metrics_zscore = {}

def process_and_plot_scatter_for_subject_hand(updated_metrics, subject, hand, show_plots=True):
    global updated_metrics_zscore

    # Get the data dictionaries for the specified subject and hand.
    durations_dict = updated_metrics[subject][hand]['durations']
    distance_dict = updated_metrics[subject][hand]['distance']

    # Get sorted trial keys for consistent ordering.
    trial_keys = sorted(durations_dict.keys())
    num_reaches = len(durations_dict[trial_keys[0]])

    # Create matrices of shape (num_trials, num_reaches).
    durations_matrix = np.array([durations_dict[trial] for trial in trial_keys])
    distance_matrix  = np.array([distance_dict[trial] for trial in trial_keys])

    # --- Original Scatter Plots Per Reach Index ---
    rows = math.ceil(math.sqrt(num_reaches))
    cols = math.ceil(num_reaches / rows)
    if show_plots:
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
        for i in range(num_reaches):
            ax = axs[i // cols][i % cols]
            ax.scatter(durations_matrix[:, i], distance_matrix[:, i], color='blue')
            ax.set_title(f"Reach {i + 1}")
            ax.set_xlabel("Duration")
            ax.set_ylabel("Distance")
            ax.grid(True)
        fig.suptitle(f"{subject} - {hand.capitalize()}: Original Duration vs Distance", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # --- Z-score Computation using nanmean and nanstd ---
    z_durations_matrix = (durations_matrix - np.nanmean(durations_matrix, axis=0)) / np.nanstd(durations_matrix, axis=0, ddof=0)
    z_distance_matrix  = (distance_matrix - np.nanmean(distance_matrix, axis=0)) / np.nanstd(distance_matrix, axis=0, ddof=0)

    # --- Z-scored Scatter Plots Per Reach Index with Diagonal and Perpendicular Distances ---
    # Dictionaries to store z-scored values and perpendicular distances
    zscore_durations = {}
    zscore_distance  = {}
    zscore_perp_distances = {}
    
    if show_plots:
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
        for i in range(num_reaches):
            # Save z-scored data for each reach (using 1-indexed keys)
            zscore_durations[i + 1] = z_durations_matrix[:, i].tolist()
            zscore_distance[i + 1] = z_distance_matrix[:, i].tolist()
            
            ax = axs[i // cols][i % cols]
            ax.scatter(z_durations_matrix[:, i], z_distance_matrix[:, i], color='purple')
            ax.set_title(f"Reach {i + 1}")
            ax.set_xlabel("Z-scored Duration")
            ax.set_ylabel("Z-scored Distance")
            ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
            ax.axvline(x=0, color='red', linestyle='--', linewidth=0.8)
            
            # Get axis limits and plot the 45° diagonal line
            xlims = ax.get_xlim()
            ylims = ax.get_ylim()
            min_lim = min(xlims[0], ylims[0])
            max_lim = max(xlims[1], ylims[1])
            x_vals = np.linspace(min_lim, max_lim, 100)
            ax.plot(x_vals, x_vals, color='green', linestyle='--', label='45° line')
            
            # Compute and overlay signed perpendicular distances for each dot.
            perp_distances = []
            for j, (x_val, y_val) in enumerate(zip(z_durations_matrix[:, i], z_distance_matrix[:, i])):
                # Signed perpendicular distance to the line x=y.
                distance_val = (y_val - x_val) / math.sqrt(2)
                proj = ((x_val + y_val) / 2, (x_val + y_val) / 2)
                ax.plot([x_val, proj[0]], [y_val, proj[1]], color='magenta', linestyle=':', linewidth=0.8)
                perp_distances.append(distance_val)
            zscore_perp_distances[i + 1] = perp_distances
            ax.legend()
            ax.grid(True)
        fig.suptitle(f"{subject} - {hand.capitalize()}: Z-scored Duration vs Distance", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    else:
        # If plotting is disabled: still save the z-scored data and compute signed perpendicular distances.
        for i in range(num_reaches):
            zscore_durations[i + 1] = z_durations_matrix[:, i].tolist()
            zscore_distance[i + 1] = z_distance_matrix[:, i].tolist()
            perp_distances = []
            for j, (x_val, y_val) in enumerate(zip(z_durations_matrix[:, i], z_distance_matrix[:, i])):
                distance_val = (y_val - x_val) / math.sqrt(2)
                perp_distances.append(distance_val)
            zscore_perp_distances[i + 1] = perp_distances

    # Save the z-scored data and perpendicular distances in the global structure.
    if subject not in updated_metrics_zscore:
        updated_metrics_zscore[subject] = {}
    updated_metrics_zscore[subject][hand] = {
        'durations': zscore_durations,
        'distance': zscore_distance,
        'MotorAcuity': zscore_perp_distances
    }

def get_updated_metrics_zscore(updated_metrics, show_plots=True):
    global updated_metrics_zscore
    # Loop over all subjects and all hands to process the z-scored metrics.
    for subject in updated_metrics:
        for hand in updated_metrics[subject]:
            process_and_plot_scatter_for_subject_hand(updated_metrics, subject, hand, show_plots=show_plots)
    return updated_metrics_zscore

# Call the function and return the complete updated_metrics_zscore.
updated_metrics_zscore = get_updated_metrics_zscore(updated_metrics, show_plots=False)

## -------------------------------------------------------------------------------------------------------------------
def plot_indep_dep_scatter(updated_metrics, updated_metrics_zscore, subject, hand='non_dominant',
                           indep_var='ldlj', dep_var='MotorAcuity', cols=4):
    """
    Plots scatter subplots for each reach index showing independent vs dependent variable values 
    and computes the Spearman correlation and p-value for each reach index.

    Parameters:
        updated_metrics (dict): Dictionary containing independent variable data (e.g., ldlj, sparc) per subject and hand.
        updated_metrics_zscore (dict): Dictionary containing dependent variable data (e.g., durations, distance, MotorAcuity)
                                        per subject and hand (z-scored).
        subject (str): Subject identifier.
        hand (str): Hand to process (default 'non_dominant').
        indep_var (str): Key selecting the independent variable from updated_metrics ('ldlj' or 'sparc').
        dep_var (str): Key selecting the dependent variable from updated_metrics_zscore ('durations', 'distance', or 'MotorAcuity').
        cols (int): Number of columns in the subplot grid.

    Returns:
        dict: A dictionary of correlation results with reach indices as keys.
    """
    import matplotlib.pyplot as plt

    # Retrieve independent data and dependent data.
    indep_data = updated_metrics[subject][hand][indep_var]
    dep_data = updated_metrics_zscore[subject][hand][dep_var]

    # Get sorted trial keys (to ensure consistent ordering across trials)
    trial_keys = sorted(indep_data.keys())
    # Assume that each trial's independent variable list has the same length (i.e. same number of reach indices)
    num_reaches = len(indep_data[trial_keys[0]])
    # Reach indices (assumed to be 1-indexed to match dependent data keys)
    reach_indices = list(range(1, num_reaches + 1))

    # Set up subplot grid.
    rows = (num_reaches + cols - 1) // cols
    plt.figure(figsize=(cols * 4, rows * 4))
    correlation_results = {}

    for i, reach_idx in enumerate(reach_indices):
        # Collect independent values for this reach index across trials.
        indep_values = [indep_data[trial][reach_idx - 1] for trial in trial_keys]
        
        # Retrieve dependent values for this reach index.
        dep_values = dep_data.get(reach_idx, [])
        
        # Remove pairs where either value is NaN.
        indep_values_clean = []
        dep_values_clean = []
        for indep_val, dep_val in zip(indep_values, dep_values):
            if not (np.isnan(indep_val) or np.isnan(dep_val)):
                indep_values_clean.append(indep_val)
                dep_values_clean.append(dep_val)
        
        # Compute Spearman correlation if clean lists are not empty.
        if indep_values_clean and dep_values_clean:
            corr, p_value = spearmanr(indep_values_clean, dep_values_clean)
        else:
            corr, p_value = np.nan, np.nan
            
        correlation_results[reach_idx] = {"spearman_corr": corr, "p_value": p_value}
        
        plt.subplot(rows, cols, i + 1)
        plt.scatter(indep_values_clean, dep_values_clean, color='blue')
        plt.xlabel(f'{indep_var.upper()}')
        plt.ylabel(f'{dep_var}')
        plt.title(f'Reach Index {reach_idx}\nSpearman: {corr:.2f}, p: {p_value:.3f}')
        plt.grid(True)

    plt.suptitle(f"{subject} - {hand.capitalize()} {indep_var.upper()} vs {dep_var} Scatter", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return correlation_results

def heatmap_spearman_correlation_all_subjects(updated_metrics, updated_metrics_zscore, hand="non_dominant",
                                              indep_var='ldlj', dep_var='MotorAcuity',
                                              simplified=False, return_medians=False, overlay_median=False):
    """
    Computes Spearman correlations between the independent variable and dependent variable values for each subject and reach index,
    and plots a heatmap of these correlations. If hand is set to "both", it creates subplots for both hands.
    Optionally overlays a green rectangle on each row at the cell closest to the row median and returns median values.
    Also returns the computed correlation matrices.

    Parameters:
        updated_metrics (dict): Dictionary containing independent variable values per subject and hand.
        updated_metrics_zscore (dict): Dictionary containing dependent variable (z-scored) values per subject and hand.
        hand (str): Hand to process (default "non_dominant"). If "both", plots for both 'non_dominant' and 'dominant'.
        indep_var (str): Key for the independent variable from updated_metrics ('ldlj' or 'sparc').
        dep_var (str): Key for the dependent variable from updated_metrics_zscore ('durations', 'distance', or 'MotorAcuity').
        simplified (bool): If True, plots a compact version with no annotations and minimal labels.
        return_medians (bool): If True, returns a dictionary with column and row medians.
        overlay_median (bool): If True, overlays a green rectangle on each row at the cell closest to the row median.

    Returns:
        dict: If return_medians is True, returns a dictionary with keys:
              'correlations' and 'medians'. Otherwise, returns a dictionary with key 'correlations'.
              The 'correlations' value contains the correlation matrices (and subject order) for the heatmap.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    def compute_corr_and_p_matrix(hand_key):
        subjects = sorted(updated_metrics.keys())
        num_reaches = 16
        num_subjects = len(subjects)

        corr_matrix = np.full((num_subjects, num_reaches), np.nan)
        p_matrix = np.full((num_subjects, num_reaches), np.nan)

        for s_idx, subject in enumerate(subjects):
            if hand_key not in updated_metrics[subject]:
                continue

            indep_data = updated_metrics[subject][hand_key][indep_var]
            dep_data = updated_metrics_zscore[subject][hand_key][dep_var]

            trial_keys = sorted(indep_data.keys())
            for reach_idx in range(1, num_reaches + 1):
                indep_values = [indep_data[trial][reach_idx - 1] for trial in trial_keys]
                dep_values = dep_data.get(reach_idx, [])

                # Remove pairs with NaN.
                indep_clean = []
                dep_clean = []
                for l_val, m_val in zip(indep_values, dep_values):
                    if not (np.isnan(l_val) or np.isnan(m_val)):
                        indep_clean.append(l_val)
                        dep_clean.append(m_val)

                if indep_clean and dep_clean:
                    corr, p_val = spearmanr(indep_clean, dep_clean)
                else:
                    corr, p_val = np.nan, np.nan

                corr_matrix[s_idx, reach_idx - 1] = corr
                p_matrix[s_idx, reach_idx - 1] = p_val
        return subjects, corr_matrix

    correlations_result = {}
    medians_result = {}
    
    if hand == "both":
        hands_to_plot = ["non_dominant", "dominant"]
        if simplified:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        else:
            fig, axes = plt.subplots(2, 1, figsize=(12, len(updated_metrics) * 0.5))
        axes = axes.flatten() if simplified else axes
        for idx, hand_key in enumerate(hands_to_plot):
            subjects, corr_matrix = compute_corr_and_p_matrix(hand_key)
            correlations_result[hand_key] = {"subjects": subjects, "corr_matrix": corr_matrix}
            df = pd.DataFrame(corr_matrix, index=subjects, columns=range(1, 17))
            
            ax = axes[idx]
            annot = not simplified
            yticklabels = False if simplified else df.index
            sns.heatmap(df, annot=annot, fmt=".2f", cmap="coolwarm", cbar=True,
                        xticklabels=df.columns, yticklabels=yticklabels, vmin=-1, vmax=1, ax=ax)
            ax.set_xlabel("Reach Index", fontsize=20)
            ax.set_ylabel("Subject", fontsize=20)
            ax.set_title(f"{hand_key.capitalize()} Hand", fontsize=24)
            ax.set_xticklabels(range(1, 17), fontsize=12, rotation=0)
            
            if overlay_median:
                for i, subject in enumerate(df.index):
                    row_values = df.loc[subject].dropna()
                    if row_values.empty:
                        continue
                    median_val = np.median(row_values.values)
                    col_idx = np.argmin(np.abs(df.loc[subject].values - median_val))
                    ax.add_patch(patches.Rectangle((col_idx, i), 1, 1, fill=False, edgecolor='green', lw=2))
            
            if return_medians:
                medians_result[hand_key] = {
                    "column_medians": df.median(axis=0).to_dict(),
                    "row_medians": df.median(axis=1).to_dict()
                }
        plt.tight_layout()
        plt.show()

    else:
        subjects, corr_matrix = compute_corr_and_p_matrix(hand)
        correlations_result = {"subjects": subjects, "corr_matrix": corr_matrix}
        df = pd.DataFrame(corr_matrix, index=subjects, columns=range(1, 17))
        figsize = (8, 4) if simplified else (12, 0.5 * len(subjects))
        plt.figure(figsize=figsize)
        annot = not simplified
        yticklabels = False if simplified else df.index
        ax = sns.heatmap(df, annot=annot, fmt=".2f", cmap="coolwarm", cbar=True,
                         xticklabels=df.columns, yticklabels=yticklabels, vmin=-1, vmax=1)
        ax.set_xlabel("Reach Index", fontsize=20)
        ax.set_ylabel("Subject", fontsize=20)
        ax.set_xticklabels(range(1, 17), fontsize=12, rotation=0)

        if overlay_median:
            for i, subject in enumerate(df.index):
                row_values = df.loc[subject].dropna()
                if row_values.empty:
                    continue
                median_val = np.median(row_values.values)
                col_idx = np.argmin(np.abs(df.loc[subject].values - median_val))
                ax.add_patch(patches.Rectangle((col_idx, i), 1, 1, fill=False, edgecolor='green', lw=2))

        if return_medians:
            medians_result = {
                "column_medians": df.median(axis=0).to_dict(),
                "row_medians": df.median(axis=1).to_dict()
            }
        plt.tight_layout()
        plt.show()

    if return_medians:
        return {"correlations": correlations_result, "medians": medians_result}
    else:
        return {"correlations": correlations_result}

def plot_histogram_spearman_corr_with_stats_reach_indices_by_subject(heatmap_results, show_value_on_legend=True):
    """
    Plots histograms of median Spearman correlations for each subject,
    overlaying non_dominant and dominant hands in different colors. Reports median, IQR,
    and Wilcoxon signed-rank test result by hand.

    Parameters:
        heatmap_results (dict): Heatmap results containing medians with keys 'non_dominant' and 'dominant'.
        show_value_on_legend (bool): If True, numerical values are shown in the legend.
    """
    import matplotlib.pyplot as plt

    non_dominant_row_medians = list(heatmap_results['medians']["non_dominant"]["row_medians"].values())
    dominant_row_medians = list(heatmap_results['medians']["dominant"]["row_medians"].values())

    # Calculate statistics for non dominant hand.
    median_non_dominant = np.median(non_dominant_row_medians)
    iqr_non_dominant = np.percentile(non_dominant_row_medians, 75) - np.percentile(non_dominant_row_medians, 25)
    q1_non_dominant = np.percentile(non_dominant_row_medians, 25)
    q3_non_dominant = np.percentile(non_dominant_row_medians, 75)

    # Calculate statistics for dominant hand.
    median_dominant = np.median(dominant_row_medians)
    iqr_dominant = np.percentile(dominant_row_medians, 75) - np.percentile(dominant_row_medians, 25)
    q1_dominant = np.percentile(dominant_row_medians, 25)
    q3_dominant = np.percentile(dominant_row_medians, 75)

    # Perform Wilcoxon signed-rank test for each hand separately.
    stat_non_dominant, p_value_non_dominant = wilcoxon(non_dominant_row_medians)
    stat_dominant, p_value_dominant = wilcoxon(dominant_row_medians)

    # Define labels based on option.
    label_median_non = f"Median non dominant: {median_non_dominant:.2f}" if show_value_on_legend else "Median non dominant"
    label_median_dom = f"Median dominant: {median_dominant:.2f}" if show_value_on_legend else "Median dominant"
    label_iqr_non = f"IQR non dominant: {iqr_non_dominant:.2f}" if show_value_on_legend else "IQR non dominant"
    label_iqr_dom = f"IQR dominant: {iqr_dominant:.2f}" if show_value_on_legend else "IQR dominant"

    # Plot histogram for median Spearman correlations.
    plt.figure(figsize=(8, 6))
    plt.hist(non_dominant_row_medians, bins=15, color='orange', alpha=0.7, edgecolor='black', label='non dominant Hand')
    plt.hist(dominant_row_medians, bins=15, color='blue', alpha=0.7, edgecolor='black', label='dominant Hand')
    plt.axvline(median_non_dominant, color='orange', linestyle='--', label=label_median_non)
    plt.axvline(median_dominant, color='blue', linestyle='--', label=label_median_dom)
    plt.axvspan(q1_non_dominant, q3_non_dominant, color='orange', alpha=0.2, label=label_iqr_non)
    plt.axvspan(q1_dominant, q3_dominant, color='blue', alpha=0.2, label=label_iqr_dom)
    plt.title("Histogram of Median Spearman Correlations by Hand")
    plt.xlabel("Median Spearman Correlation", fontsize=24)
    plt.ylabel("Frequency", fontsize=24)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()

    print(f"Non-dominant Hand: Median = {median_non_dominant:.2f}, IQR = {iqr_non_dominant:.2f}, Wilcoxon stat = {stat_non_dominant:.2f}, p-value = {p_value_non_dominant:.4f}")
    print(f"Dominant Hand: Median = {median_dominant:.2f}, IQR = {iqr_dominant:.2f}, Wilcoxon stat = {stat_dominant:.2f}, p-value = {p_value_dominant:.4f}")

## -------------------------------------------------------------------------------------------------------------------
# independent variable: ldlj or sparc
# dependent variable: durations, distance, or MotorAcuity

## -------------------------------------------------------------------------------------------------------------------
## -------------------- LDLJ ---------------------------------
## -------------------------------------------------------------------------------------------------------------------
# 1. indep_var='ldlj', dep_var='durations'
# Example call: using 'ldlj' as independent variable and 'durations' as dependent variable
correlation_results = plot_indep_dep_scatter(updated_metrics, updated_metrics_zscore,
                                             subject='07/22/HW', hand='non_dominant',
                                             indep_var='ldlj', dep_var='durations', cols=4)


# Example call: using 'ldlj' as independent variable and 'durations' as dependent variable for both hands
heatmap_results = heatmap_spearman_correlation_all_subjects(
    updated_metrics, updated_metrics_zscore, hand="both",
    indep_var='ldlj', dep_var='durations',
    simplified=True, return_medians=True, overlay_median=True
)
# Example call to plot histogram with statistics
plot_histogram_spearman_corr_with_stats_reach_indices_by_subject(heatmap_results, show_value_on_legend=True)
## -------------------------------------------------------------------------------------------------------------------

## -------------------------------------------------------------------------------------------------------------------
# 2. indep_var='ldlj', dep_var='distance'
# Example call: using 'ldlj' as independent variable and 'distance' as dependent variable
correlation_results = plot_indep_dep_scatter(updated_metrics, updated_metrics_zscore,
                                             subject='07/22/HW', hand='non_dominant',
                                             indep_var='ldlj', dep_var='distance', cols=4)
# Example call: using 'ldlj' as independent variable and 'distance' as dependent variable for both hands
heatmap_results = heatmap_spearman_correlation_all_subjects(
    updated_metrics, updated_metrics_zscore, hand="both",
    indep_var='ldlj', dep_var='distance',
    simplified=True, return_medians=True, overlay_median=True
)
# Example call to plot histogram with statistics
plot_histogram_spearman_corr_with_stats_reach_indices_by_subject(heatmap_results, show_value_on_legend=True)
## -------------------------------------------------------------------------------------------------------------------

## -------------------------------------------------------------------------------------------------------------------
# 3. indep_var='ldlj', dep_var='MotorAcuity'
# Example call: using 'ldlj' as independent variable and 'MotorAcuity' as dependent variable
correlation_results = plot_indep_dep_scatter(updated_metrics, updated_metrics_zscore,
                                             subject='07/22/HW', hand='non_dominant',
                                             indep_var='ldlj', dep_var='MotorAcuity', cols=4)

# Example call: using 'ldlj' as independent variable and 'MotorAcuity' as dependent variable for both hands
heatmap_results = heatmap_spearman_correlation_all_subjects(
    updated_metrics, updated_metrics_zscore, hand="both",
    indep_var='ldlj', dep_var='MotorAcuity',
    simplified=True, return_medians=True, overlay_median=True
)
# Example call to plot histogram with statistics
plot_histogram_spearman_corr_with_stats_reach_indices_by_subject(heatmap_results, show_value_on_legend=True)
## -------------------------------------------------------------------------------------------------------------------

## -------------------------------------------------------------------------------------------------------------------
## -------------------- SPARC ---------------------------------
## -------------------------------------------------------------------------------------------------------------------
# 1. indep_var='sparc', dep_var='durations'
# Example call: using 'sparc' as independent variable and 'durations' as dependent variable
correlation_results = plot_indep_dep_scatter(updated_metrics, updated_metrics_zscore,
                                             subject='07/22/HW', hand='non_dominant',
                                             indep_var='sparc', dep_var='durations', cols=4)
# Example call: using 'sparc' as independent variable and 'durations' as dependent variable for both hands
heatmap_results = heatmap_spearman_correlation_all_subjects(
    updated_metrics, updated_metrics_zscore, hand="both",
    indep_var='sparc', dep_var='durations',
    simplified=True, return_medians=True, overlay_median=True
)
# Example call to plot histogram with statistics
plot_histogram_spearman_corr_with_stats_reach_indices_by_subject(heatmap_results, show_value_on_legend=True)
## -------------------------------------------------------------------------------------------------------------------

## -------------------------------------------------------------------------------------------------------------------
# 2. indep_var='sparc', dep_var='distance'
# Example call: using 'sparc' as independent variable and 'distance' as dependent variable
correlation_results = plot_indep_dep_scatter(updated_metrics, updated_metrics_zscore,
                                             subject='07/22/HW', hand='non_dominant',
                                             indep_var='sparc', dep_var='distance', cols=4)
# Example call: using 'sparc' as independent variable and 'distance' as dependent variable for both hands
heatmap_results = heatmap_spearman_correlation_all_subjects(
    updated_metrics, updated_metrics_zscore, hand="both",
    indep_var='sparc', dep_var='distance',
    simplified=True, return_medians=True, overlay_median=True
)
# Example call to plot histogram with statistics
plot_histogram_spearman_corr_with_stats_reach_indices_by_subject(heatmap_results, show_value_on_legend=True)
## -------------------------------------------------------------------------------------------------------------------

## -------------------------------------------------------------------------------------------------------------------
# 3. indep_var='sparc', dep_var='MotorAcuity'
# Example call: using 'sparc' as independent variable and 'MotorAcuity' as dependent variable
correlation_results = plot_indep_dep_scatter(updated_metrics, updated_metrics_zscore,
                                             subject='07/22/HW', hand='non_dominant',
                                             indep_var='sparc', dep_var='MotorAcuity', cols=4)
# Example call: using 'sparc' as independent variable and 'MotorAcuity' as dependent variable for both hands
heatmap_results = heatmap_spearman_correlation_all_subjects(
    updated_metrics, updated_metrics_zscore, hand="both",
    indep_var='sparc', dep_var='MotorAcuity',
    simplified=True, return_medians=True, overlay_median=True
)
# Example call to plot histogram with statistics
plot_histogram_spearman_corr_with_stats_reach_indices_by_subject(heatmap_results, show_value_on_legend=True)
## -------------------------------------------------------------------------------------------------------------------

def plot_ldlj_vs_sparc(updated_metrics, subject='07/22/HW', hand='non_dominant'):
    import matplotlib.pyplot as plt
    
    # Retrieve the Ldlj and Sparc data for the chosen subject and hand
    ldlj_data = updated_metrics[subject][hand]['ldlj']
    sparc_data = updated_metrics[subject][hand]['sparc']

    # Get sorted trial keys for consistent ordering
    trial_keys = sorted(ldlj_data.keys())
    num_reaches = len(ldlj_data[trial_keys[0]])

    # Set up subplot grid based on the number of reach indices
    rows = int(np.ceil(np.sqrt(num_reaches)))
    cols = int(np.ceil(num_reaches / rows))

    plt.figure(figsize=(cols * 4, rows * 4))

    for reach_index in range(num_reaches):
        # Collect values across trials for the current reach index
        ldlj_vals = [ldlj_data[trial][reach_index] for trial in trial_keys]
        sparc_vals = [sparc_data[trial][reach_index] for trial in trial_keys]

        print(f"Reach Index {reach_index + 1}: Ldlj values: {ldlj_vals}")
        print(f"Reach Index {reach_index + 1}: Sparc values: {sparc_vals}")
        
        # Remove pairs with NaN values
        ldlj_clean, sparc_clean = [], []
        for l_val, s_val in zip(ldlj_vals, sparc_vals):
            if not (np.isnan(l_val) or np.isnan(s_val)):
                ldlj_clean.append(l_val)
                sparc_clean.append(s_val)
        
        # Compute Spearman correlation and p-value, if available
        if ldlj_clean and sparc_clean:
            corr, p_value = spearmanr(ldlj_clean, sparc_clean)
        else:
            corr, p_value = np.nan, np.nan

        plt.subplot(rows, cols, reach_index + 1)
        plt.scatter(ldlj_clean, sparc_clean, color='blue')
        plt.xlabel("Ldlj")
        plt.ylabel("Sparc")
        plt.title(f"Index {reach_index + 1}\nSpearman: {corr:.2f}, p: {p_value:.3f}")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

# Example call to plot Ldlj vs Sparc for a specific subject and hand
plot_ldlj_vs_sparc(updated_metrics, subject='07/22/HW', hand='non_dominant')

def heatmap_spearman_correlation_ldlj_sparc(updated_metrics, hand="non_dominant",
                                            simplified=False, return_medians=False, overlay_median=False):
    """
    Computes Spearman correlations (and p-values) between ldlj and sparc for each subject and reach index,
    and plots a heatmap of these values. If hand is set to "both", it creates subplots for both
    'non_dominant' and 'dominant'. Optionally overlays a green rectangle on each row at the cell
    closest to its median and returns median values along with the correlation and p-value matrices.

    Parameters:
        updated_metrics (dict): Dictionary containing keys for each subject and hand with 'ldlj' and 'sparc' data.
        hand (str): Hand to process. Either a single hand (e.g., "non_dominant") or "both" for both hands.
        simplified (bool): If True, plots a compact version with minimal annotations.
        return_medians (bool): If True, returns median values for each row and column.
        overlay_median (bool): If True, overlays a green rectangle on each row at the cell closest to its median.

    Returns:
        dict: If return_medians is True, returns a dict with keys 'correlations' and 'medians';
              otherwise, returns a dict with key 'correlations'. The correlations dict contains both
              correlation and p-value matrices.
    """
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    def compute_corr_matrix(hand_key):
        subjects = sorted(updated_metrics.keys())
        num_subjects = len(subjects)
        num_reaches = None

        # Determine num_reaches from the first subject with the hand_key.
        for subject in subjects:
            if hand_key in updated_metrics[subject]:
                ldlj_data = updated_metrics[subject][hand_key].get('ldlj', {})
                if ldlj_data:
                    trial_keys = sorted(ldlj_data.keys())
                    if trial_keys:
                        num_reaches = len(ldlj_data[trial_keys[0]])
                        break
        if num_reaches is None:
            raise ValueError("Could not determine number of reaches.")

        corr_matrix = np.full((num_subjects, num_reaches), np.nan)
        p_matrix = np.full((num_subjects, num_reaches), np.nan)

        for s_idx, subject in enumerate(subjects):
            if hand_key not in updated_metrics[subject]:
                continue
            ldlj_data = updated_metrics[subject][hand_key].get('ldlj', {})
            sparc_data = updated_metrics[subject][hand_key].get('sparc', {})
            trial_keys = sorted(ldlj_data.keys())
            for reach_idx in range(num_reaches):
                ldlj_vals = []
                sparc_vals = []
                for trial in trial_keys:
                    try:
                        l_val = ldlj_data[trial][reach_idx]
                        s_val = sparc_data[trial][reach_idx]
                        if not (np.isnan(l_val) or np.isnan(s_val)):
                            ldlj_vals.append(l_val)
                            sparc_vals.append(s_val)
                    except (KeyError, IndexError):
                        continue
                if ldlj_vals and sparc_vals:
                    corr, p_val = spearmanr(ldlj_vals, sparc_vals)
                else:
                    corr, p_val = np.nan, np.nan
                corr_matrix[s_idx, reach_idx] = corr
                p_matrix[s_idx, reach_idx] = p_val
        return subjects, corr_matrix, p_matrix

    correlations_result = {}
    medians_result = {}

    def plot_for_hand(hand_key, ax):
        subjects, corr_matrix, p_matrix = compute_corr_matrix(hand_key)
        # Create dataframes for correlations and p-values.
        df_corr = pd.DataFrame(corr_matrix, index=subjects, columns=range(1, corr_matrix.shape[1] + 1))
        df_p = pd.DataFrame(p_matrix, index=subjects, columns=range(1, p_matrix.shape[1] + 1))
        # Create annotation dataframe combining correlation and p-value.
        annot_df = df_corr.copy().astype(str)
        for i in range(df_corr.shape[0]):
            for j in range(df_corr.shape[1]):
                corr_val = df_corr.iat[i, j]
                p_val = df_p.iat[i, j]
                if pd.isna(corr_val) or pd.isna(p_val):
                    annot_df.iat[i, j] = ""
                else:
                    annot_df.iat[i, j] = f"{corr_val:.2f}\np={p_val:.3f}"
        correlations_result[hand_key] = {"subjects": subjects, "corr_matrix": corr_matrix, "p_matrix": p_matrix}
        annot = not simplified
        yticklabels = False if simplified else df_corr.index
        sns.heatmap(df_corr, annot=annot_df if annot else False, fmt="", cmap="coolwarm", cbar=True,
                    xticklabels=df_corr.columns, yticklabels=yticklabels, vmin=-1, vmax=1, ax=ax)
        ax.set_xlabel("Reach Index", fontsize=14)
        ax.set_ylabel("Subject", fontsize=14)
        ax.set_title(f"{hand_key.capitalize()} Hand (Ldlj vs Sparc)", fontsize=16)
        ax.set_xticklabels(df_corr.columns, fontsize=12, rotation=0)
        if overlay_median:
            for i, subject in enumerate(df_corr.index):
                row_values = df_corr.loc[subject].dropna()
                if row_values.empty:
                    continue
                median_val = np.median(row_values.values)
                col_idx = np.argmin(np.abs(row_values.values - median_val))
                ax.add_patch(patches.Rectangle((col_idx, i), 1, 1, fill=False, edgecolor='green', lw=2))
        if return_medians:
            medians_result[hand_key] = {
                "column_medians": df_corr.median(axis=0).to_dict(),
                "row_medians": df_corr.median(axis=1).to_dict()
            }

    if hand == "both":
        hands_to_plot = ["non_dominant", "dominant"]
        if simplified:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        else:
            fig, axes = plt.subplots(2, 1, figsize=(12, len(updated_metrics) * 0.5))
        axes = axes.flatten()
        for idx, hand_key in enumerate(hands_to_plot):
            plot_for_hand(hand_key, axes[idx])
        plt.tight_layout()
        plt.show()
    else:
        subjects, corr_matrix, p_matrix = compute_corr_matrix(hand)
        correlations_result = {"subjects": subjects, "corr_matrix": corr_matrix, "p_matrix": p_matrix}
        df_corr = pd.DataFrame(corr_matrix, index=subjects, columns=range(1, (pd.DataFrame(corr_matrix).shape[1] or 17) + 1))
        df_p = pd.DataFrame(p_matrix, index=subjects, columns=range(1, (pd.DataFrame(p_matrix).shape[1] or 17) + 1))
        annot_df = df_corr.copy().astype(str)
        for i in range(df_corr.shape[0]):
            for j in range(df_corr.shape[1]):
                corr_val = df_corr.iat[i, j]
                p_val = df_p.iat[i, j]
                if pd.isna(corr_val) or pd.isna(p_val):
                    annot_df.iat[i, j] = ""
                else:
                    annot_df.iat[i, j] = f"{corr_val:.2f}\np={p_val:.3f}"
        figsize = (8, 4) if simplified else (12, 0.5 * len(subjects))
        plt.figure(figsize=figsize)
        annot = not simplified
        ax = sns.heatmap(df_corr, annot=annot_df if annot else False, fmt="", cmap="coolwarm", cbar=True,
                         xticklabels=df_corr.columns, yticklabels=False if simplified else df_corr.index, vmin=-1, vmax=1)
        ax.set_xlabel("Reach Index", fontsize=14)
        ax.set_ylabel("Subject", fontsize=14)
        ax.set_xticklabels(df_corr.columns, fontsize=12, rotation=0)
        if overlay_median:
            import matplotlib.patches as patches
            for i, subject in enumerate(df_corr.index):
                row_values = df_corr.loc[subject].dropna()
                if row_values.empty:
                    continue
                median_val = np.median(row_values.values)
                col_idx = np.argmin(np.abs(row_values.values - median_val))
                ax.add_patch(patches.Rectangle((col_idx, i), 1, 1, fill=False, edgecolor='green', lw=2))
        plt.tight_layout()
        plt.show()
        if return_medians:
            medians_result = {
                "column_medians": df_corr.median(axis=0).to_dict(),
                "row_medians": df_corr.median(axis=1).to_dict()
            }
    if return_medians:
        return {"correlations": correlations_result, "medians": medians_result}
    else:
        return {"correlations": correlations_result}

# Example call to compute and plot heatmap of Spearman correlations (with p-values) between Ldlj and Sparc for both hands
heatmap_results = heatmap_spearman_correlation_ldlj_sparc(
    updated_metrics, hand="both",
    simplified=True, return_medians=True, overlay_median=True
)

def compute_icc_for_ldlj_sparc(updated_metrics, subject, hand):
    """
    Computes the intraclass correlation coefficient (ICC) between Ldlj and Sparc values
    across trials for each reach index for a given subject and hand.

    Parameters:
        updated_metrics (dict): Dictionary containing subjects' data.
        subject (str): Subject identifier.
        hand (str): Hand identifier ('non_dominant' or 'dominant').

    Returns:
        dict: Dictionary with reach indices as keys and corresponding ICC (ICC2) values.
    """
    icc_results = {}
    ldlj_data = updated_metrics[subject][hand].get('ldlj', {})
    sparc_data = updated_metrics[subject][hand].get('sparc', {})
    
    trial_keys = sorted(ldlj_data.keys())
    if not trial_keys:
        return icc_results

    num_reaches = len(ldlj_data[trial_keys[0]])
    
    for reach_idx in range(num_reaches):
        records = []
        for trial in trial_keys:
            try:
                ldlj_val = ldlj_data[trial][reach_idx]
                sparc_val = sparc_data[trial][reach_idx]
                # Create entries for both measures per trial
                records.append({'trial': trial, 'measure': 'ldlj', 'value': ldlj_val})
                records.append({'trial': trial, 'measure': 'sparc', 'value': sparc_val})
            except (IndexError, KeyError):
                continue
        
        if not records:
            continue
        
        df = pd.DataFrame(records)
        # Rename columns to match pingouin's expected names: 'targets', 'raters', 'ratings'
        df = df.rename(columns={'trial': 'targets', 'measure': 'raters', 'value': 'ratings'})
        # Compute ICC using a two-way random-effects model (consistency ICC2)
        icc_table = pg.intraclass_corr(data=df, targets='targets', raters='raters', ratings='ratings')
        # Select ICC2 value (if available)
        icc_value = icc_table.loc[icc_table['Type'] == 'ICC2', 'ICC'].values
        icc_results[reach_idx + 1] = icc_value[0] if icc_value.size > 0 else np.nan

    return icc_results

# Example usage:
icc_ldlj_sparc = compute_icc_for_ldlj_sparc(updated_metrics, subject='07/22/HW', hand='non_dominant')
print("Intraclass Correlation (ICC) for Ldlj vs Sparc by Reach Index:")
for reach_idx, icc_val in icc_ldlj_sparc.items():
    if np.isnan(icc_val):
        interp = "Insufficient data to compute ICC."
    elif icc_val < 0.5:
        interp = "Poor reliability"
    elif icc_val < 0.75:
        interp = "Moderate reliability"
    elif icc_val < 0.90:
        interp = "Good reliability"
    else:
        interp = "Excellent reliability"
    print(f"Reach {reach_idx}: ICC = {icc_val:.2f}  -> {interp}")

def plot_corr_histogram_overlay(heatmap_results, significant_only=True, significance_threshold=0.05):
    """
    Plots an overlayed histogram of Spearman correlation values for both hands,
    filtering based on the p-values from the heatmap_results p_matrix.
    
    Parameters:
        heatmap_results (dict): Dictionary containing heatmap correlation results.
        significant_only (bool): If True, only plot correlations with p-values less than significance_threshold.
        significance_threshold (float): The p-value threshold for filtering significant correlations.
    """
    import matplotlib.pyplot as plt

    # Extract data for non_dominant hand.
    corr_matrix_nd = np.array(heatmap_results["correlations"]["non_dominant"]["corr_matrix"])
    p_matrix_nd = np.array(heatmap_results["correlations"]["non_dominant"]["p_matrix"])
    valid_nd = ~np.isnan(corr_matrix_nd) & ~np.isnan(p_matrix_nd)
    corr_values_nd = corr_matrix_nd[valid_nd]
    p_values_nd = p_matrix_nd[valid_nd]
    if significant_only:
        corr_values_nd = corr_values_nd[p_values_nd < significance_threshold]

    # Extract data for dominant hand.
    corr_matrix_dom = np.array(heatmap_results["correlations"]["dominant"]["corr_matrix"])
    p_matrix_dom = np.array(heatmap_results["correlations"]["dominant"]["p_matrix"])
    valid_dom = ~np.isnan(corr_matrix_dom) & ~np.isnan(p_matrix_dom)
    corr_values_dom = corr_matrix_dom[valid_dom]
    p_values_dom = p_matrix_dom[valid_dom]
    if significant_only:
        corr_values_dom = corr_values_dom[p_values_dom < significance_threshold]

    plt.figure(figsize=(10, 6))
    bins = 20

    # Plot histograms overlayed for both hands.
    plt.hist(corr_values_nd, bins=bins, color='orange', alpha=0.5, edgecolor='black', label='Non-dominant')
    plt.hist(corr_values_dom, bins=bins, color='blue', alpha=0.5, edgecolor='black', label='Dominant')

    title_type = "Significant " if significant_only else "All "
    plt.title(f"Histogram of {title_type}Spearman Correlations (Both Hands)")
    plt.xlabel("Spearman Correlation")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example call to plot overlayed histograms for both hands.
plot_corr_histogram_overlay(heatmap_results, significant_only=False, significance_threshold=0.05)

# Example call to plot histogram with statistics
plot_histogram_spearman_corr_with_stats_reach_indices_by_subject(heatmap_results, show_value_on_legend=True)

def plot_ldlj_sparc_correlation(updated_metrics):
    """
    Plots Ldlj vs Sparc scatter plots for each subject and hand,
    computes the Spearman correlation and p-value,
    and returns the results.

    Parameters:
        updated_metrics (dict): Dictionary containing the 'ldlj' and 'sparc' data
                                for each subject and hand.

    Returns:
        dict: A nested dictionary with the structure:
              { subject: { hand: (spearman_correlation, p_value), ... }, ... }
    """

    subjects = sorted(updated_metrics.keys())
    hands = ['non_dominant', 'dominant']
    
    total_plots = len(subjects) * len(hands)
    rows = 6
    cols = int(np.ceil(total_plots / rows))
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axs = axs.flatten()
    
    correlation_results = {}
    plot_idx = 0
    
    for subject in subjects:
        correlation_results[subject] = {}
        for hand in hands:
            if hand not in updated_metrics[subject]:
                correlation_results[subject][hand] = (np.nan, np.nan)
                axs[plot_idx].set_axis_off()
                plot_idx += 1
                continue

            ldlj_values = []
            sparc_values = []

            # Concatenate all trial data for the current subject and hand
            for trial in updated_metrics[subject][hand]['ldlj']:
                ldlj_values.extend(updated_metrics[subject][hand]['ldlj'][trial])
                sparc_values.extend(updated_metrics[subject][hand]['sparc'][trial])

            ldlj_values = np.array(ldlj_values)
            sparc_values = np.array(sparc_values)

            # Remove NaN values
            mask = ~np.isnan(ldlj_values) & ~np.isnan(sparc_values)
            ldlj_clean = ldlj_values[mask]
            sparc_clean = sparc_values[mask]

            # Compute Spearman correlation and p-value
            corr, p_value = spearmanr(ldlj_clean, sparc_clean)
            
            correlation_results[subject][hand] = (corr, p_value)

            ax = axs[plot_idx]
            ax.scatter(ldlj_clean, sparc_clean, color='blue', alpha=0.7)
            ax.set_xlabel('Ldlj')
            ax.set_ylabel('Sparc')
            ax.set_title(f"{subject} - {hand}\nSpearman: {corr:.2f}, p: {p_value:.4f}")
            ax.grid(True)
            plot_idx += 1

    # Turn off any remaining subplots if they exist
    for ax in axs[plot_idx:]:
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()

    return correlation_results

# Example call to plot Ldlj vs Sparc correlations for all subjects and hands
ldlj_sparc_correlations = plot_ldlj_sparc_correlation(updated_metrics)

# Extract correlation and p-value values from ldlj_sparc_correlations dictionary.
corr_values = []
p_values = []
for subject, hands in ldlj_sparc_correlations.items():
    for hand, (corr, p_val) in hands.items():
        if not np.isnan(corr):
            corr_values.append(corr)
            p_values.append(p_val)

# Plot histogram of the Spearman correlations.
plt.figure(figsize=(8, 6))
plt.hist(corr_values, bins=10, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel('Spearman Correlation')
plt.ylabel('Frequency')
plt.title('Histogram of Ldlj vs Sparc Correlations')

# Add a vertical line for the median.
median_corr = np.median(corr_values)
plt.axvline(median_corr, color='red', linestyle='--', linewidth=2, label=f"Median: {median_corr:.2f}")
plt.legend()

plt.grid(True)
plt.show()

# Calculate percentage of p-values less than 0.05
p_values_array = np.array(p_values)
significant_count = np.sum(p_values_array < 0.05)
percentage_significant = (significant_count / len(p_values_array)) * 100
print(f"Percentage of p-values < 0.05: {percentage_significant:.2f}%")


