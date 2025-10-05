import utils1 # Importing utils1 for data Pre-processing
import utils2 # Importing utils2 for reach metrics calculation and time window Specific calculation
import utils3 # Importing utils3 for plotting functions
import utils4 # Importing utils4 for image files
import utils5 # Importing utils5 for combining metrics
import utils6 # Importing utils6 for Data Analysis and Visualization
import utils7 # Importing utils7 for Motor Experiences
import utils8 # Importing utils8 for sBBTResult
import utils9 # Importing utils9 for thesis
# -------------------------------------------------------------------------------------------------------------------
import pickle
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib as mpl
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

import math
import numpy as np
from scipy.stats import zscore
from scipy.stats import wilcoxon
from scipy.stats import spearmanr
from scipy.stats import chisquare
from scipy.stats import circmean, rayleigh
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import norm

import pandas as pd
import seaborn as sns

import pingouin as pg

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.proj3d import proj_transform
import matplotlib.patches as patches

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from scipy.stats import spearmanr
from scipy.stats import linregress
import numpy as np
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import wilcoxon
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib as mpl
from scipy.stats import spearmanr, shapiro

from scipy.stats import shapiro, spearmanr
from scipy.optimize import curve_fit
from scipy.stats import linregress, shapiro
from matplotlib.patches import Patch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Patch, Rectangle
from scipy.stats import pearsonr
# -------------------------------------------------------------------------------------------------------------------

Traj_folder = "/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025"
Box_Traj_folder = "/Users/yilinwu/Desktop/Yilin-Honours/Box/Traj/2025"
Figure_folder = "/Users/yilinwu/Desktop/honours/Thesis/figure"
DataProcess_folder = "/Users/yilinwu/Desktop/honours data/DataProcess"
tBBT_Image_folder = "/Users/yilinwu/Desktop/Yilin-Honours/tBBT_Image/2025/"

prominence_threshold_speed = 400
prominence_threshold_position = 80
# -------------------------------------------------------------------------------------------------------------------
# If x axis is hand, x axis label as Non-Dominant / Dominant

# overlay sample size
# n=32 reaches 
# n=16 locations 
# n=29 participants

# stat test
# paired t-test


# distance label as Error (mm)
# duration label as Duration (s)
# if the y axis is corrlation coefficient, y axis label as Correlation
# if colcor bar is correlation coefficient, label as Correlation

# 1. sbbt results
# y axis label as sBBT Score (no. of blocks)
# y-limits to round numbers and consider having the lowest value be 0.
# If x axis is hand, x axis label as Non-Dominant / Dominant
# n =29 participants show on the plot
# do paired t-test and  to assess whether dominant score is higher and indicate significance on the plot


# 1. sbbt results
# y-axis label → "sBBT Score (no. of blocks)"
# y-axis limits → rounded, minimum forced to 0
# x-tick label → "Non-Dominant / Dominant"
# x-axis label → "Hand"
# n = 29 participants displayed on the plot
# Annotate sample size in the figure right top
# paired t-test performed and annotated on the figure (showing significance if dominant > non-dominant).
# stattest result size = 50
# figsize=(6, 4)
# axis_label_font=14,
# tick_label_font=14,
# marker_size=50,
# alpha=0.7,
# bar_width=0.6,
# order=("Non-dominant", "Dominant"),
# box_colors = ["Non-dominant":"#D3D3D3","Dominant" "#F0F0F0"]
# random_jitter=0.04
# no tittle
# no Grid
# # Perform paired t-test (Dominant > Non-dominant) and annotate significance
# t_stat, p_val = ttest_rel(sBBTResult["dominant"], sBBTResult["non_dominant"])
# y_sig = y_max * 1.05
# ax.plot([indices[0], indices[1]], [y_sig, y_sig], color="black", linewidth=1.5)

# if p_val < 0.001:
#     stars = "***"
# elif p_val < 0.01:
#     stars = "**"
# elif p_val < 0.05:
#     stars = "*"
# else:
#     stars = "ns"
# ax.text(np.mean(indices), y_sig + (y_max * 0.02), stars,
#         ha="center", va="bottom", fontsize=axis_label_font)



# 2. tbbt results



def generate_placement_colors(show_plot=True):
    """
    Generate 16 distinct colors (4 categories x 4 shades) for placement locations.
    
    Parameters:
        show_plot (bool): If True, display a bar plot of the colors.
        
    Returns:
        np.ndarray: Array of shape (16, 4) with RGBA colors.
    """
    def generate_shades(base_rgb, n_shades=4):
        factors = np.linspace(0.4, 1.0, n_shades)  # avoid very dark
        shades = np.array([base_rgb * f for f in factors])
        shades = np.clip(shades, 0, 1)
        return np.hstack([shades, np.ones((n_shades, 1))])  # add alpha=1

    # Base colors (RGB)
    base_colors = [
        np.array([0.0, 0.3, 0.8]),  # blue
        np.array([0.0, 0.6, 0.2]),  # green
        np.array([0.8, 0.1, 0.1]),  # red
        np.array([0.9, 0.5, 0.0])   # orange
    ]
    
    # Generate shades and combine
    placement_location_colors = np.vstack([generate_shades(c) for c in base_colors])
    
    if show_plot:
        plt.figure(figsize=(12, 2))
        for i, c in enumerate(placement_location_colors):
            plt.bar(i, 1, color=c)
            plt.text(i, 0.5, str(i+1), ha='center', va='center', 
                     color='white' if np.mean(c[:3]) < 0.5 else 'black', fontsize=12)
        plt.axis('off')
        plt.show()
    
    return placement_location_colors
def create_icon_layout(colors, mirror=False):
    """
    Create compact square icon with annotated numbers using given colors.
    Layout:
      13 14 15 16
       9 10 11 12
       5  6  7  8
       1  2  3  4
    If mirror is True, the layout will be mirrored horizontally.
    """
    layout = np.array([
        [13, 14, 15, 16],
        [9, 10, 11, 12],
        [5, 6, 7, 8],
        [1, 2, 3, 4]
    ])

    if mirror:
        layout = np.fliplr(layout)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.axis("off")
    
    # Draw squares
    for row in range(4):
        for col in range(4):
            num = layout[row, col]
            color = colors[num-1]
            rect = patches.Rectangle((col, 3-row), 1, 1, linewidth=1,
                                     edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            # Choose text color based on brightness
            text_color = 'white' if np.mean(color[:3]) < 0.5 else 'black'
            ax.text(col+0.5, 3-row+0.5, str(num), ha='center', va='center',
                    fontsize=40, color=text_color, weight="bold")
    
    plt.show()

# Generate colors and draw icon (not mirrored)
placement_location_colors = generate_placement_colors(show_plot=False)
# create_icon_layout(placement_location_colors, mirror=False)
# create_icon_layout(placement_location_colors, mirror=True)


# -------------------------------------------------------------------------------------------------------------------
# --- GET ALL DATES ---
All_dates = sorted(utils1.get_subfolders_with_depth(Traj_folder, depth=3))

# --- LOAD ALL SUBJECTS' tBBT ERROR FROM IMAGE, SAVE AS pickle file---
Block_Distance = utils4.load_selected_subject_errors(All_dates, DataProcess_folder)

# --- LOAD RESULTS FROM PICKLE FILE "processed_results.pkl" ---
results = utils1.load_selected_subject_results(All_dates, DataProcess_folder)
# -------------------------------------------------------------------------------------------------------------------
# Calculate RMS reprojection error statistics
utils4.compute_rms_reprojection_error_stats()
# # -------------------------------------------------------------------------------------------------------------------
# Load sBBTResult from CSV into a DataFrame and compute right and left hand scores
sBBTResult = utils8.load_and_compute_sbbt_result()
# Swap and rename sBBTResult scores for specific subjects
sBBTResult = utils8.swap_and_rename_sbbt_result(sBBTResult)
sBBTResult_stats = utils8.compute_sbbt_result_stats(sBBTResult)
sBBT_combine_stat = utils8.analyze_sbbt_results(sBBTResult)
# -------------------------------------------------------------------------------------------------------------------
# PART 2: Reach Metrics Calculation
# --- GET REACH SPEED SEGMENTS ---
reach_speed_segments = utils2.get_reach_speed_segments(results)
# --- CALCULATE REACH METRICS ---
# reach_durations
# reach_cartesian_distances
# reach_path_distances
# reach_v_peaks
# reach_v_peak_indices
reach_metrics = utils2.calculate_reach_metrics(reach_speed_segments, results, fs=200)
test_windows_7 = utils9.compute_test_window_7(results, reach_speed_segments, reach_metrics)

def analyze_trajectory(traj_data, coord_prefix, start_idx, end_idx, dt=1/200,
                       v_threshold=500, time_threshold=0.05, immobility_threshold=100,
                       distance_threshold=300, jerk_tol=300):
    # Extract coordinates, velocity, acceleration
    x = np.array(traj_data[coord_prefix + "X"][start_idx:end_idx])
    y = np.array(traj_data[coord_prefix + "Y"][start_idx:end_idx])
    z = np.array(traj_data[coord_prefix + "Z"][start_idx:end_idx])
    vx = np.array(traj_data[coord_prefix + "VX"][start_idx:end_idx])
    vy = np.array(traj_data[coord_prefix + "VY"][start_idx:end_idx])
    vz = np.array(traj_data[coord_prefix + "VZ"][start_idx:end_idx])
    ax_data = np.array(traj_data[coord_prefix + "AX"][start_idx:end_idx])
    ay = np.array(traj_data[coord_prefix + "AY"][start_idx:end_idx])
    az = np.array(traj_data[coord_prefix + "AZ"][start_idx:end_idx])

    # Compute speed, acceleration along velocity, jerk
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    acc_along_vel = (vx*ax_data + vy*ay + vz*az) / (v + 1e-8)
    jerk_along_vel = np.gradient(acc_along_vel, dt)

    # Global velocity peak (with dip detection)
    global_peak_idx = None
    for i in range(1, len(v)-1):
        # Case 1: normal + → − flip
        cond1 = (acc_along_vel[i-1] > 0 and acc_along_vel[i+1] < 0)

        # Case 2: local minimum (dip, still positive)
        cond2 = (acc_along_vel[i-1] > acc_along_vel[i] < acc_along_vel[i+1])

        if (cond1 or cond2) and v[i] >= v_threshold and i*dt >= time_threshold:
            global_peak_idx = i
            break

    # Sub-movement end
    sub_movement_idx = None
    if global_peak_idx is not None:
        for i in range(global_peak_idx, len(v)-1):
            if v[i] <= immobility_threshold \
               or (acc_along_vel[i-1] < 0 and acc_along_vel[i+1] > 0) \
               or (jerk_along_vel[i-1] > 0 and jerk_along_vel[i+1] < 0) \
               or abs(jerk_along_vel[i]) < jerk_tol:
                sub_movement_idx = i
                break

    # Ballistic phase end
    end_of_ballistic_idx = None
    target = np.array([x[-1], y[-1], z[-1]])
    if sub_movement_idx is not None:
        for i in range(sub_movement_idx, len(v)):
            if np.linalg.norm(np.array([x[i], y[i], z[i]]) - target) <= distance_threshold:
                end_of_ballistic_idx = i
                break

    return {
        "x": x, "y": y, "z": z, "v": v, "acc_along_vel": acc_along_vel,
        "global_peak_idx": global_peak_idx,
        "sub_movement_idx": sub_movement_idx,
        "end_of_ballistic_idx": end_of_ballistic_idx,
        "dt": dt
    }
def compute_phase_indices(All_dates, results, test_windows_7):
    phase_indices = {
        "global_peak_idx": {},
        "sub_movement_idx": {},
        "end_of_ballistic_idx": {}
    }

    for subject in All_dates:
        phase_indices["global_peak_idx"][subject] = {}
        phase_indices["sub_movement_idx"][subject] = {}
        phase_indices["end_of_ballistic_idx"][subject] = {}

        for hand in results[subject]:
            phase_indices["global_peak_idx"][subject][hand] = {}
            phase_indices["sub_movement_idx"][subject][hand] = {}
            phase_indices["end_of_ballistic_idx"][subject][hand] = {}

            for file_path in results[subject][hand][1]:
                phase_indices["global_peak_idx"][subject][hand][file_path] = []
                phase_indices["sub_movement_idx"][subject][hand][file_path] = []
                phase_indices["end_of_ballistic_idx"][subject][hand][file_path] = []

                traj_data = results[subject][hand][1][file_path]['traj_data']
                coord_prefix = "LFIN_" if hand == "left" else "RFIN_"
                num_segments = len(test_windows_7[subject][hand][file_path])

                for seg_index in range(num_segments):
                    start_idx, end_idx = test_windows_7[subject][hand][file_path][seg_index]
                    results_dict = analyze_trajectory(traj_data, coord_prefix, start_idx, end_idx)

                    phase_indices["global_peak_idx"][subject][hand][file_path].append(
                        results_dict["global_peak_idx"]
                    )
                    phase_indices["sub_movement_idx"][subject][hand][file_path].append(
                        results_dict["sub_movement_idx"]
                    )
                    phase_indices["end_of_ballistic_idx"][subject][hand][file_path].append(
                        results_dict["end_of_ballistic_idx"]
                    )

    return phase_indices
phase_indices = compute_phase_indices(All_dates, results, test_windows_7)

def check_nan_phase_indices(phase_indices):
    import math
    nan_count = 0
    nan_details = []

    for subject in phase_indices["end_of_ballistic_idx"]:
        for hand in phase_indices["end_of_ballistic_idx"][subject]:
            for file_path, indices in phase_indices["end_of_ballistic_idx"][subject][hand].items():
                for seg_idx, idx in enumerate(indices):
                    if idx is None or (isinstance(idx, float) and math.isnan(idx)):
                        nan_count += 1
                        nan_details.append((subject, hand, file_path, seg_idx))
                        
    print(f"Total NaN (or None) entries in end_of_ballistic_idx: {nan_count}")
    for detail in nan_details:
        print(f"NaN found in Subject: {detail[0]}, Hand: {detail[1]}, File: {detail[2]}, Segment index: {detail[3]}")
        
    # Optionally return the details for further processing
    return nan_count, nan_details
nan_count, nan_details = check_nan_phase_indices(phase_indices)

# --- COMPUTE BALLISTIC AND CORRECTIVE PHASES ---
def compute_two_time_windows(test_windows, phase_indices):
    """
    Compute ballistic and corrective phases using test_window 7 indices and phase_indices.

    Ballistic phase: from the start of test_window 7 (tw[0]) to the end_of_ballistic_idx from phase_indices.
    Corrective phase: from end_of_ballistic_idx from phase_indices to the end of test_window 7 (tw[1]).

    Parameters:
        test_windows (dict): Nested dictionary with test window 7 indices.
                             For each subject, hand, file_path, and segment, it provides a tuple (start, end).
        phase_indices (dict): Nested dictionary that includes 'end_of_ballistic_idx' for each subject, hand, file_path, segment.

    Returns:
        ballistic_phase (dict): Dictionary with lists of (start, end_of_ballistic_idx) tuples.
        correction_phase (dict): Dictionary with lists of (end_of_ballistic_idx, end) tuples.
    """
    ballistic_phase = {
        subject: {
            hand: {
                file_path: [
                    (tw[0],
                     phase_indices["end_of_ballistic_idx"][subject][hand][file_path][seg] + tw[0]
                     if phase_indices["end_of_ballistic_idx"][subject][hand][file_path][seg] is not None
                     else math.floor((tw[1]+ tw[0]) / 2))
                    for seg, tw in enumerate(test_windows[subject][hand][file_path])
                ]
                for file_path in test_windows[subject][hand]
            }
            for hand in test_windows[subject]
        }
        for subject in test_windows
    }

    correction_phase = {
        subject: {
            hand: {
                file_path: [
                    ((tw[0] + phase_indices["end_of_ballistic_idx"][subject][hand][file_path][seg]) if phase_indices["end_of_ballistic_idx"][subject][hand][file_path][seg] is not None else math.floor((tw[1]+tw[0]) / 2), tw[1])
                    for seg, tw in enumerate(test_windows[subject][hand][file_path])
                ]
                for file_path in test_windows[subject][hand]
            }
            for hand in test_windows[subject]
        }
        for subject in test_windows
    }

    return ballistic_phase, correction_phase
ballistic_phase, correction_phase = compute_two_time_windows(test_windows_7, phase_indices)
# -------------------------------------------------------------------------------------------------------------------
def plot_phase_duration_averages(ballistic_phase, correction_phase, test_windows_7, reach_speed_segments, fs=200):
    """
    For each subject, average the phase durations across hands and trials,
    then plot histograms for the subject-averaged durations for:
      - ballistic phases,
      - correction phases,
      - test window phases, and
      - reach speed segments.
    """
    import matplotlib.pyplot as plt

    ballistic_subj_avgs = []
    correction_subj_avgs = []
    test_window_subj_avgs = []
    speed_segments_subj_avgs = []

    subjects = ballistic_phase.keys()  # assuming same subjects exist in all dictionaries

    # Process each subject independently
    for subject in subjects:
        subj_ballistic = []
        subj_correction = []
        subj_test = []
        subj_speed = []

        # Loop over hands for ballistic, correction, and test window phases
        for hand in ballistic_phase[subject]:
            # Process ballistic phases for this hand (each phase is a tuple (start, end))
            for file_path in ballistic_phase[subject][hand]:
                phases = ballistic_phase[subject][hand][file_path]
                for phase in phases:
                    if phase is not None and isinstance(phase, (list, tuple)) and len(phase) == 2:
                        start, end = phase
                        duration = (end - start) / fs
                        subj_ballistic.append(duration)
            # Process correction phases for this hand
            for file_path in correction_phase[subject][hand]:
                phases = correction_phase[subject][hand][file_path]
                for phase in phases:
                    if phase is not None and isinstance(phase, (list, tuple)) and len(phase) == 2:
                        start, end = phase
                        duration = (end - start) / fs
                        subj_correction.append(duration)
            # Process test window phases for this hand
            for file_path in test_windows_7[subject][hand]:
                phases = test_windows_7[subject][hand][file_path]
                for phase in phases:
                    if phase is not None and isinstance(phase, (list, tuple)) and len(phase) == 2:
                        start, end = phase
                        duration = (end - start) / fs
                        subj_test.append(duration)
            # Process reach speed segments for this hand
            for file_path in reach_speed_segments[subject][hand]:
                segments = reach_speed_segments[subject][hand][file_path]
                for segment in segments:
                    if segment is not None and isinstance(segment, (list, tuple)) and len(segment) == 2:
                        start, end = segment
                        duration = (end - start) / fs
                        subj_speed.append(duration)

        # Average durations for each phase type within subject if data exists
        if subj_ballistic:
            ballistic_subj_avgs.append(np.mean(subj_ballistic))
        if subj_correction:
            correction_subj_avgs.append(np.mean(subj_correction))
        if subj_test:
            test_window_subj_avgs.append(np.mean(subj_test))
        if subj_speed:
            speed_segments_subj_avgs.append(np.mean(subj_speed))

    # Print basic statistics for subject-average durations
    if ballistic_subj_avgs:
        print("Subject-Average Ballistic Phase Duration: Mean = {:.3f}s, Std = {:.3f}s, Count = {}".format(
            np.mean(ballistic_subj_avgs), np.std(ballistic_subj_avgs), len(ballistic_subj_avgs)))
    if correction_subj_avgs:
        print("Subject-Average Correction Phase Duration: Mean = {:.3f}s, Std = {:.3f}s, Count = {}".format(
            np.mean(correction_subj_avgs), np.std(correction_subj_avgs), len(correction_subj_avgs)))
    if test_window_subj_avgs:
        print("Subject-Average Test Window Duration: Mean = {:.3f}s, Std = {:.3f}s, Count = {}".format(
            np.mean(test_window_subj_avgs), np.std(test_window_subj_avgs), len(test_window_subj_avgs)))
    if speed_segments_subj_avgs:
        print("Subject-Average Reach Speed Segment Duration: Mean = {:.3f}s, Std = {:.3f}s, Count = {}".format(
            np.mean(speed_segments_subj_avgs), np.std(speed_segments_subj_avgs), len(speed_segments_subj_avgs)))

    # Plot histograms of the subject-averaged durations
    plt.figure(figsize=(12,6))
    if ballistic_subj_avgs:
        plt.hist(ballistic_subj_avgs, bins=10, alpha=0.7, label='Ballistic Phase (avg per subject)', color='skyblue')
    if correction_subj_avgs:
        plt.hist(correction_subj_avgs, bins=10, alpha=0.7, label='Correction Phase (avg per subject)', color='salmon')
    if test_window_subj_avgs:
        plt.hist(test_window_subj_avgs, bins=10, alpha=0.7, label='Test Window (avg per subject)', color='lightgreen')
    if speed_segments_subj_avgs:
        plt.hist(speed_segments_subj_avgs, bins=10, alpha=0.7, label='Reach Speed Segments (avg per subject)', color='plum')
    plt.xlabel('Duration (seconds)', fontsize=14)
    plt.ylabel('Number of Subjects', fontsize=14)
    plt.title('Histogram of Subject-Average Phase Durations', fontsize=16)
    plt.legend(fontsize=12)
    plt.show()

# Call the function (this assumes that ballistic_phase, correction_phase, test_windows_7, and reach_speed_segments are already defined)
plot_phase_duration_averages(ballistic_phase, correction_phase, test_windows_7, reach_speed_segments)

# -------------------------------------------------------------------------------------------------------------------
reach_TW_metrics_ballistic_phase = utils2.calculate_reach_metrics_for_time_windows_Normalizing(ballistic_phase, results)
reach_TW_metrics_correction_phase = utils2.calculate_reach_metrics_for_time_windows_Normalizing(correction_phase, results)
reach_TW_metrics_TW = utils2.calculate_reach_metrics_for_time_windows_Normalizing(test_windows_7, results)

reach_sparc_ballistic_phase = utils2.calculate_reach_sparc_Normalizing(ballistic_phase, results)
reach_sparc_correction_phase = utils2.calculate_reach_sparc_Normalizing(correction_phase, results)
reach_sparc_TW = utils2.calculate_reach_sparc_Normalizing(test_windows_7, results)

# -------------------------------------------------------------------------------------------------------------------
utils5.process_and_save_combined_metrics_acorss_phases(
    Block_Distance, reach_metrics,
    reach_TW_metrics_ballistic_phase, reach_TW_metrics_correction_phase,
    reach_sparc_ballistic_phase, reach_sparc_correction_phase,
    reach_TW_metrics_TW, reach_sparc_TW,
    All_dates, DataProcess_folder)
# --- LOAD ALL COMBINED METRICS PER SUBJECT FROM PICKLE FILE ---
all_combined_metrics_acorss_phases = utils5.load_selected_subject_results_acorss_TWs(All_dates, DataProcess_folder)

# Swap and rename metrics for consistency
all_combined_metrics_acorss_phases = utils5.swap_and_rename_metrics(all_combined_metrics_acorss_phases, All_dates)

# Filter all_combined_metrics based on distance and count NaNs
filtered_metrics_acorss_phases, total_nan_acorss_phases, Nan_counts_per_subject_per_hand_acorss_phases, Nan_counts_per_index_acorss_phases = utils5.filter_combined_metrics_and_count_nan(all_combined_metrics_acorss_phases)

# Update filtered metrics and count NaN replacements based on distance and duration thresholds: distance_threshold=15, duration_threshold=1.6
updated_metrics_acorss_phases, Cutoff_counts_per_subject_per_hand_acorss_phases, Cutoff_counts_per_index_acorss_phases, total_nan_per_subject_hand_acorss_phases = utils5.update_filtered_metrics_and_count(filtered_metrics_acorss_phases)

# -------------------------------------------------------------------------------------------------------------------
# 1. sBBT task performance: Dominant vs Non-Dominant hand
sbbt_plot_config = dict(
    figsize=(5, 4),
    scale_factor=1,
    axis_label_font=14,
    tick_label_font=14,
    title_font=16,
    marker_size=50,
    alpha=0.4,
    bar_width=0.5,
    bar_edge_width=1.5,
    bar_colors={"Non-dominant": "#A9A9A9", "Dominant": "#F0F0F0"},
    order=("Non-dominant", "Dominant"),
    random_jitter=0.04,
    bar_spacing=0.1,  # spacing fraction between bars
    show_title=False,
    show_grid=False,
    x_ticks=True,
    y_ticks=True,
    annotate_n=True,
    n_loc=(0.95, 1.05),  # "top-right" or "bottom" or (x, y) in axes fraction
    n_unit="participants",  # optional: "blocks", "participants", "cm", "locations", or None
    annotate_sig=True,
    sig_text_offset=-0.05,
    sig_marker_size=40,
    sig_line=True,
    sig_line_width=1.5,
    sig_line_color="black",
    sig_y_loc=90,
    sig_levels=[(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")],
    test_type="greater", # "greater" (if Dominant > Non-dominant), "less", or "two-sided" 
    y_unit="blocks",  # optional: "blocks", "participants", "cm", "locations", or None
    show_whiskers=False,  #  option: show error bar whiskers or not
    show_points=True     #  option: overlay individual data points or not
)

# def plot_sbbt_bargraph(sBBTResult, config):
#     """
#     Plot a bar graph of sBBT scores by hand with full configurability.
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from scipy.stats import ttest_rel
#     import pandas as pd
#     import seaborn as sns



#     # Scaling factor
#     sf = config.get("scale_factor", 1.0)

#     # Figure & fonts
#     figsize = config.get("figsize", (6, 4))
#     axis_label_font = config.get("axis_label_font", 14) * sf
#     tick_label_font = config.get("tick_label_font", 14) * sf
#     title_font = config.get("title_font", 16) * sf

#     # Bar & scatter
#     marker_size = config.get("marker_size", 50) * sf
#     alpha = config.get("alpha", 0.7)
#     bar_width = config.get("bar_width", 0.6)
#     bar_edge_width = config.get("bar_edge_width", 1.5) * sf
#     bar_colors = config.get("bar_colors", {"Non-dominant": "#A9A9A9", "Dominant": "#F0F0F0"})
#     order = config.get("order", ("Non-dominant", "Dominant"))
#     random_jitter = config.get("random_jitter", 0.04)
#     bar_spacing = config.get("bar_spacing", 0.3)  # spacing fraction between bars

#     # Axis & grid
#     show_title = config.get("show_title", False)
#     show_grid = config.get("show_grid", False)
#     x_ticks = config.get("x_ticks", True)
#     y_ticks = config.get("y_ticks", True)

#     # Sample size
#     annotate_n = config.get("annotate_n", True)
#     n_loc = config.get("n_loc", "top-right")
#     n_unit = config.get("n_unit", "participants")

#     # Significance
#     annotate_sig = config.get("annotate_sig", True)
#     sig_text_offset = config.get("sig_text_offset", 0.02)
#     sig_marker_size = config.get("sig_marker_size", 14) * sf
#     sig_line = config.get("sig_line", True)
#     sig_line_width = config.get("sig_line_width", 1.5) * sf
#     sig_line_color = config.get("sig_line_color", "black")
#     sig_y_loc = config.get("sig_y_loc", "auto")
#     sig_levels = config.get("sig_levels", [(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")])
#     test_type = config.get("test_type", "greater")

#     # Y-axis unit
#     y_unit = config.get("y_unit", None)
#     y_label_base = "sBBT Score"
#     y_label = f"{y_label_base} (no. of {y_unit})" if y_unit else y_label_base

#     # New options
#     show_whiskers = config.get("show_whiskers", True)
#     show_points = config.get("show_points", True)

#     # Subjects
#     n = len(sBBTResult)
#     means = [sBBTResult["non_dominant"].mean(), sBBTResult["dominant"].mean()]
#     sems = [sBBTResult["non_dominant"].std() / np.sqrt(n),
#             sBBTResult["dominant"].std() / np.sqrt(n)]

#     # Figure
#     fig, ax = plt.subplots(figsize=(figsize[0]*sf, figsize[1]*sf))
#     indices = np.arange(len(order)) * (1 + bar_spacing)

#     # Bars with or without whiskers
#     error_kw = {'elinewidth': bar_edge_width} if show_whiskers else None
#     yerr = sems if show_whiskers else None
#     ax.bar(indices, means,
#            yerr=yerr,
#            color=[bar_colors[order[0]], bar_colors[order[1]]],
#            width=bar_width,
#            capsize=8 if show_whiskers else 0,
#            edgecolor='black',
#            linewidth=bar_edge_width,
#            error_kw=error_kw)

#     # Scatter points overlay if enabled
#     if show_points:
#         for i, key in enumerate(["non_dominant", "dominant"]):
#             x_vals = indices[i] + np.random.uniform(-random_jitter, random_jitter, n)
#             ax.scatter(x_vals, sBBTResult[key], color='black', s=marker_size, zorder=10, alpha=alpha)

#     # Labels
#     ax.set_xlabel("Hand", fontsize=axis_label_font)
#     ax.set_ylabel(y_label, fontsize=axis_label_font)
#     if show_title:
#         ax.set_title("sBBT Bar Graph", fontsize=title_font)
#     ax.set_xticks(indices)
#     ax.set_xticklabels(["Non-dominant", "Dominant"], fontsize=tick_label_font)
#     ax.tick_params(axis='x', which='both', bottom=x_ticks)
#     ax.tick_params(axis='y', which='both', left=y_ticks, labelsize=tick_label_font)

#     # Y-axis limits
#     max_score = sBBTResult[["non_dominant", "dominant"]].max().max()
#     y_max = int(np.ceil(max_score / 5.0)) * 5
#     if y_max < max_score:
#         y_max += 5
#     ax.set_ylim(0, y_max)

#     # Spines & grid
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.grid(show_grid)

#     # Sample size annotation
#     if annotate_n:
#         if isinstance(n_loc, tuple):
#             ax.text(n_loc[0], n_loc[1], f"n = {n} {n_unit}", transform=ax.transAxes,
#                     ha="center", va="center", fontsize=tick_label_font)
#         elif n_loc == "top-right":
#             ax.text(0.95, 0.95, f"n = {n} {n_unit}", transform=ax.transAxes,
#                     ha="right", va="top", fontsize=tick_label_font)
#         elif n_loc == "bottom":
#             ax.text(0.5, -0.15, f"n = {n} {n_unit}", transform=ax.transAxes,
#                     ha="center", va="center", fontsize=tick_label_font)

#     # Significance annotation
#     if annotate_sig:
#         t_stat, p_val_two = ttest_rel(sBBTResult["dominant"], sBBTResult["non_dominant"])
#         if test_type == "greater":
#             p_val = p_val_two / 2 if t_stat > 0 else 1.0
#         elif test_type == "less":
#             p_val = p_val_two / 2 if t_stat < 0 else 1.0
#         else:
#             p_val = p_val_two

#         # Determine significance stars
#         for threshold, symbol in sig_levels:
#             if p_val <= threshold:
#                 stars = symbol
#                 break

#         # Determine y location
#         y_sig = y_max * 1.05 if sig_y_loc == "auto" else sig_y_loc

#         # Draw line connecting bars if enabled
#         if sig_line:
#             ax.plot([indices[0], indices[1]], [y_sig, y_sig],
#                     color=sig_line_color, linewidth=sig_line_width)

#         # Place significance text
#         ax.text(np.mean(indices), y_sig + (y_max * sig_text_offset),
#                 stars, ha="center", va="bottom", fontsize=sig_marker_size)

#     plt.tight_layout()
#     plt.show()

# plot_sbbt_bargraph(sBBTResult, sbbt_plot_config)



def plot_sbbt_boxplot(sBBTResult, config):
    """
    Plot a box plot (with individual jittered points) of sBBT scores by hand with full configurability.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy.stats import ttest_rel, shapiro
    import os

    # Scaling factor and configuration parameters
    sf = config.get("scale_factor", 1.0)
    figsize = config.get("figsize", (6, 4))
    axis_label_font = config.get("axis_label_font", 14) * sf
    tick_label_font = config.get("tick_label_font", 14) * sf
    title_font = config.get("title_font", 16) * sf
    marker_size = config.get("marker_size", 50) * sf
    alpha = config.get("alpha", 0.7)
    bar_width = config.get("bar_width", 0.6)
    bar_colors = config.get("bar_colors", {"Non-dominant": "#A9A9A9", "Dominant": "#F0F0F0"})
    order = config.get("order", ("Non-dominant", "Dominant"))
    random_jitter = config.get("random_jitter", 0.04)
    show_title = config.get("show_title", False)
    show_grid = config.get("show_grid", False)
    annotate_n = config.get("annotate_n", True)
    n_loc = config.get("n_loc", "top-right")
    n_unit = config.get("n_unit", "participants")
    annotate_sig = config.get("annotate_sig", True)
    sig_text_offset = config.get("sig_text_offset", 0.02)
    sig_marker_size = config.get("sig_marker_size", 14) * sf
    sig_line = config.get("sig_line", True)
    sig_line_width = config.get("sig_line_width", 1.5) * sf
    sig_line_color = config.get("sig_line_color", "black")
    sig_y_loc = config.get("sig_y_loc", "auto")
    sig_levels = config.get("sig_levels", [(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")])
    test_type = config.get("test_type", "greater")
    
    # Y-axis label: optionally include unit text
    y_unit = config.get("y_unit", None)
    y_label_base = "sBBT score"
    y_label = f"{y_label_base} (no. of {y_unit})" if y_unit else y_label_base

    # Convert sBBTResult (assumed to be a dict with keys "non_dominant" and "dominant")
    # into a long format DataFrame.
    df = pd.DataFrame({
        "Hand": (["Non-dominant"] * len(sBBTResult["non_dominant"])) +
                (["Dominant"] * len(sBBTResult["dominant"])),
        "Score": np.concatenate([sBBTResult["non_dominant"].values,
                                 sBBTResult["dominant"].values])
    })
    # Ensure the ordering of hands follows the provided 'order'
    df["Hand"] = pd.Categorical(df["Hand"], categories=order, ordered=True)

    # Create box plot with seaborn, hiding fliers.
    fig, ax = plt.subplots(figsize=(figsize[0]*sf, figsize[1]*sf))
    sns.boxplot(x="Hand", y="Score", data=df, order=order, palette=bar_colors, 
                ax=ax, width=bar_width, showfliers=False)
    # Overlay swarm plot for individual data points with slight random jitter.
    sns.swarmplot(x="Hand", y="Score", data=df, order=order, color='black', 
                  size=marker_size/10, alpha=alpha, ax=ax)

    # Axis labels and title
    ax.set_xlabel("Hand", fontsize=axis_label_font)
    ax.set_ylabel(y_label, fontsize=axis_label_font)
    if show_title:
        ax.set_title("sBBT Box Plot", fontsize=title_font)
    ax.tick_params(axis='x', labelsize=tick_label_font)
    ax.tick_params(axis='y', labelsize=tick_label_font)

    # Set Y-axis limits (round to nearest 5)
    max_score = df["Score"].max()
    y_max = int(np.ceil(max_score / 5.0)) * 5
    if y_max < max_score:
        y_max += 5
    ax.set_ylim(0, y_max)

    # Customize spines and grid
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(show_grid)

    # Sample size annotation
    n = len(sBBTResult)
    if annotate_n:
        ax.text(0.7, 0.2, f"n = {n} {n_unit}", transform=ax.transAxes,
                ha="center", va="center", fontsize=tick_label_font)

    # Normality test on differences between dominant and non-dominant scores.
    diff_scores = np.array(sBBTResult["dominant"]) - np.array(sBBTResult["non_dominant"])
    stat_norm, p_norm = shapiro(diff_scores)
    print(f"Normality test (Shapiro-Wilk) on differences: W = {stat_norm:.4f}, p = {p_norm:.4f}")
    if p_norm < 0.05:
        print("Warning: Normality assumption is violated. The t-test results may not be reliable.")
    else:
        print("Normality assumption is satisfied.")

    # Significance annotation with paired t-test
    if annotate_sig:
        t_stat, p_val_two = ttest_rel(sBBTResult["dominant"], sBBTResult["non_dominant"])
        if test_type == "greater":
            p_val = p_val_two / 2 if t_stat > 0 else 1.0
        elif test_type == "less":
            p_val = p_val_two / 2 if t_stat < 0 else 1.0
        else:
            p_val = p_val_two

        # Determine significance stars
        for threshold, symbol in sig_levels:
            if p_val <= threshold:
                stars = symbol
                break

        y_sig = y_max * 1.05 if sig_y_loc == "auto" else sig_y_loc
        if sig_line:
            ax.plot([0, 1], [y_sig, y_sig], color=sig_line_color, linewidth=sig_line_width)
        ax.text(0.5, y_sig + (y_max * sig_text_offset), stars,
                ha="center", va="bottom", fontsize=sig_marker_size)

    plt.tight_layout()
    # Save figure as an SVG in Figure_folder
    plt.savefig(os.path.join(Figure_folder, "sbbt_boxplot.svg"), format="svg")
    plt.show()

    # Assume the following variables have been computed earlier:
    #  n                   : number of pairs,
    #  median_cond1, iqr_cond1 : median and IQR for condition 1,
    #  median_cond2, iqr_cond2 : median and IQR for condition 2.
    df_deg = n - 1
    dz = t_stat / np.sqrt(n)
    print(f"Paired-samples t-test: t({df_deg}) = {t_stat:.4f}, p = {p_val:.4f} (one-tailed), dz = {dz:.4f}")
    print(f"Median (Non-dominant): {np.median(sBBTResult['non_dominant']):.2f}, IQR: {np.percentile(sBBTResult['non_dominant'], 75) - np.percentile(sBBTResult['non_dominant'], 25):.2f}")
    print(f"Median (Dominant): {np.median(sBBTResult['dominant']):.2f}, IQR: {np.percentile(sBBTResult['dominant'], 75) - np.percentile(sBBTResult['dominant'], 25):.2f}")

plot_sbbt_boxplot(sBBTResult, sbbt_plot_config)
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# 2. tBBT task performance: Duration and Distance Trade-off

plot_config_summary = dict(
    # -----------------------------
    # General Plot Settings
    # -----------------------------
    general=dict(
        figsize=(5, 4),
        scale_factor=1,
        axis_label_font=14,
        tick_label_font=14,
        title_font=16,
        show_title=False,
        show_grid=False,
        x_ticks=True,
        y_ticks=True,
        tick_direction="out",   # always outwards
        annotate_n=True,
        n_loc=(0.95, 1.05),
        n_unit="participants",  # can be "blocks", "participants", "cm", "locations", or None
        random_jitter=0.04,
        show_whiskers=False,
        show_points=True,
        marker_size=50,
        alpha=0.4,
        label_offset=0.09,
        hide_spines=True,
        showGrid=False
    ),

    # -----------------------------
    # Axis Labeling Rules (common style)
    # -----------------------------
    axis_labels=dict(
        duration="Duration (s)",
        distance="Error (mm)",
        correlation="Correlation",
        y_unit="blocks"  # added from sbbt_plot_config
    ),
    axis_colors=dict(
        x={
            "Duration (s)": {"start": "fast", "end": "slow", "colors": ["green", "red"]}
        },
        y={
            "Error (mm)": {"start": "accurate", "end": "inaccurate", "colors": ["green", "red"]}
        }
    ),

    # -----------------------------
    # Plot-Type Specific Options
    # -----------------------------
    scatter=dict(
        show_points=True,
        annotate_corr=True,
        corr_line_y0=True,
        ylim_centered_at_zero=True,
        use_axis_colors=True,
        corr_display="rho_only"
    ),
    line=dict(
        show_markers=False,
        show_error_shade=False,
        linewidth=4,
        use_axis_colors=True
    ),
    heatmap=dict(
        colormap="coolwarm",
        show_colorbar=True,
        colorbar_label="Correlation",
        center_zero=True,
        use_axis_colors=True
    ),
    box=dict(
        bar_width=0.5,
        bar_edge_width=1.5,
        bar_colors={"Non-dominant": "#A9A9A9", "Dominant": "#F0F0F0"},
        order=("Non-dominant", "Dominant"),
        bar_spacing=0.1,
        show_whiskers=False,
        show_points=True,
        annotate_sig=True,
        sig_levels=[(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")],
        test_type="greater",
        sig_y_loc=90,
        sig_line=True,
        sig_line_width=1.5,
        sig_line_color="black",
        sig_marker_size=40,
        sig_text_offset=-0.05,
        use_axis_colors=True
    ),

    # -----------------------------
    # Statistical Options
    # -----------------------------
    stats=dict(
        compare_correlation="fisher_z",
        test_type_options=["greater", "less", "two-sided"]
    ),

    # -----------------------------
    # Optional / Misc Features
    # -----------------------------
    misc=dict(
        bar_spacing=0.1,
        placement_icon=True,
        annotate_sig=True
    )
)

# -------------------------------------------------------------------------------------------------------------------
### Plot hand trajectory with velocity-coded coloring and highlighted segments
def plot_trajectory(results, subject='07/22/HW', hand='right', trial=1,
                    file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT53.csv',
                    overlay_trial=0, velocity_segment_only=False, plot_mode='all'):
    """
    Plots the instantaneous velocity and a 3D trajectory for the specified trial.
    Colors each trajectory point based on the instantaneous velocity.
    Points outside highlighted segments are colored lightgrey when velocity_segment_only is True.
    
    Options:
      - plot_mode: 'all' to plot the entire trial or 'segment' to plot only from the first to the last highlight.
    
    Parameters:
        results (dict): The results dictionary containing trajectory data.
        subject (str): Subject key in the results dictionary.
        hand (str): Hand key ('right' or 'left') in the results dictionary.
        trial (int): The trial index to use for the main trajectory data.
        file_path (str): The file key for selecting trajectory data.
        overlay_trial (int): The trial index used to extract overlay indices for highlighting.
        velocity_segment_only (bool): If True, apply velocity-coded color only within highlighted segments.
        plot_mode (str): 'all' to plot the entire trial or 'segment' to plot only from the first to the last highlight.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as mcolors

    # Extract trajectory data for the given trial
    traj_data = results[subject][hand][trial][file_path]['traj_data']
    coord_prefix = "RFIN_" if hand == "right" else "LFIN_"
    coord_x = np.array(traj_data[coord_prefix + "X"])
    coord_y = np.array(traj_data[coord_prefix + "Y"])
    coord_z = np.array(traj_data[coord_prefix + "Z"])
    
    # Extract overlay index (or indices) from the overlay_trial
    overlay_index = results[subject][hand][overlay_trial][file_path]
    highlight_indices = overlay_index if isinstance(overlay_index, (list, np.ndarray)) else [overlay_index]
    highlight_indices = sorted(highlight_indices)
    
    n_points = len(coord_x)
    marker = "RFIN" if hand == "right" else "LFIN"

    # Extract instantaneous velocity from trajectory space (assume constant sampling rate = 200Hz)
    vel = results[subject][hand][trial][file_path]['traj_space'][marker][1]
    
    # Normalize velocities between 0 and 1
    v_min = np.min(vel)
    v_max = np.max(vel)
    if v_max - v_min > 0:
        v_norm = (vel - v_min) / (v_max - v_min)
    else:
        v_norm = np.ones_like(vel)
    
    # Map velocity to colors using the viridis colormap with an exponential scaling for contrast
    point_colors = [plt.cm.viridis(1 - (v_norm[i]**2)) for i in range(n_points)]
    
    # If velocity_segment_only is True, only retain velocity colors within highlighted segments.
    if velocity_segment_only and highlight_indices:
        segments = []
        for i in range(0, len(highlight_indices) - 1, 2):
            segments.append((highlight_indices[i], highlight_indices[i+1]))
        for i in range(n_points):
            in_segment = any(min(seg) <= i <= max(seg) for seg in segments)
            if not in_segment:
                point_colors[i] = mcolors.to_rgba('lightgrey')
    
    # Determine the indices to plot based on the plot_mode option
    if plot_mode == 'segment' and highlight_indices:
        start_idx = min(highlight_indices)
        end_idx = max(highlight_indices)
    else:
        start_idx = 0
        end_idx = n_points - 1

    plot_indices = np.arange(start_idx, end_idx + 1)
    coord_x_plot = coord_x[plot_indices]
    coord_y_plot = coord_y[plot_indices]
    coord_z_plot = coord_z[plot_indices]
    vel_plot    = np.array(vel)[plot_indices]
    colors_plot = [point_colors[i] for i in plot_indices]
    time_points = plot_indices / 200  # Time axis in seconds

    # Create the plot layout: two subplots arranged vertically (one top, one bottom)
    # Increase the bottom (3D) subplot by adjusting height_ratios and reduce vertical gap.
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 2.5])
    plt.subplots_adjust(hspace=0.01)  # reduce gap between subplots
    
    # Top subplot: Plot instantaneous velocity over time.
    ax_vel = fig.add_subplot(gs[0, 0])
    ax_vel.scatter(time_points, vel_plot, c=colors_plot, marker='o', s=5)
    ax_vel.set_xlabel('Time (s)', fontsize=16)
    ax_vel.set_ylabel('velocity\nmagnitude\n(mm/s)', fontsize=16)
    ax_vel.set_ylim([0, 1500])
    ax_vel.set_yticks([0, 750, 1500])
    ax_vel.set_yticklabels([0, 750, 1500], fontsize=16)
    ax_vel.spines['top'].set_visible(False)
    ax_vel.spines['right'].set_visible(False)
    ax_vel.grid(False)
    # Set the velocity axis on top
    ax_vel.set_zorder(2)
    ax_vel.patch.set_alpha(0.0)
    # Add subplot label (A)
    # ax_vel.text(-0.2, 0.95, "(A)", transform=ax_vel.transAxes, fontsize=20)
    # ax_vel.text(-0.2, -0.45, "(B)", transform=ax_vel.transAxes, fontsize=20)

    
    # Overlay markers at the designated highlight indices (if they fall within the plot range).
    for order, idx in enumerate(highlight_indices, start=1):
        if start_idx <= idx <= end_idx:
            t_val = idx / 200
            color = 'black' if order % 2 == 1 else 'grey'
            marker_sym = 'o'
            ax_vel.scatter(t_val, vel[idx], color=color, marker=marker_sym, s=40)
    
    # Bottom subplot: Plot the 3D trajectory.
    ax3d = fig.add_subplot(gs[1, 0], projection='3d')
    ax3d.scatter(coord_x_plot, coord_y_plot, coord_z_plot, c=colors_plot, marker='o', s=5)
    ax3d.set_xlabel("X (mm)", fontsize=16)
    ax3d.set_ylabel("Y (mm)", fontsize=16)
    ax3d.set_zlabel("Z (mm)", fontsize=16)
    ax3d.set_xticks([-250, 0, 250])
    ax3d.set_xticklabels([-250, 0, 250])
    ax3d.set_yticks([-50, 50, 150])
    ax3d.set_yticklabels([-50, 50, 150])
    ax3d.set_zticks([800, 950, 1100])
    ax3d.set_zticklabels([800, 950, 1100])
    ax3d.set_xlim([-250, 250])
    ax3d.set_ylim([-50, 150])
    ax3d.set_zlim([800, 1100])
    ax3d.set_box_aspect([10, 4, 6])
    # Set the 3D axis behind the velocity plot
    ax3d.set_zorder(1)
    
    # Overlay markers on the 3D plot corresponding to highlight indices.
    for order, idx in enumerate(highlight_indices, start=1):
        if start_idx <= idx <= end_idx:
            color = 'black' if order % 2 == 1 else 'grey'
            marker_sym = 'o'
            ax3d.scatter(coord_x[idx], coord_y[idx], coord_z[idx], color=color, marker=marker_sym, s=30)

    plt.tight_layout()
    plt.show()
    
    # Optional: Create an additional 3D plot for a selected segment if at least 4 highlight indices exist.
    if len(highlight_indices) >= 4:
        seg_start = highlight_indices[2]
        seg_end = highlight_indices[3]
        seg_indices = np.arange(seg_start, seg_end + 1)
        seg_coord_x = coord_x[seg_indices]
        seg_coord_y = coord_y[seg_indices]
        seg_coord_z = coord_z[seg_indices]
        seg_colors = [point_colors[i] for i in seg_indices]
        
        fig2 = plt.figure(figsize=(10, 8))
        ax3d_seg = fig2.add_subplot(111, projection='3d')
        ax3d_seg.scatter(seg_coord_x, seg_coord_y, seg_coord_z, c=seg_colors, marker='o', s=5)
        ax3d_seg.scatter(coord_x[seg_start], coord_y[seg_start], coord_z[seg_start],
                         color='green', marker='o', s=50, label='start')
        ax3d_seg.scatter(coord_x[seg_end], coord_y[seg_end], coord_z[seg_end],
                         color='blue', marker='X', s=50, label='end')
        ax3d_seg.set_xlabel(f"{coord_prefix}X (mm)", fontsize=14)
        ax3d_seg.set_ylabel(f"{coord_prefix}Y (mm)", fontsize=14)
        ax3d_seg.set_zlabel(f"{coord_prefix}Z (mm)", fontsize=14)
        ax3d_seg.legend()
        plt.tight_layout()
        plt.show()

plot_trajectory(results, subject='07/22/HW', hand='left', trial=1,
                file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv',
                overlay_trial=0, velocity_segment_only=True, plot_mode='segment')

def plot_trajectory(results, subject='07/22/HW', hand='right', trial=1,
                    file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT53.csv',
                    overlay_trial=0, velocity_segment_only=False, plot_mode='all'):
    """
    Plots the instantaneous velocity and a 3D trajectory for the specified trial.
    Colors each trajectory point based on the instantaneous velocity.
    Points outside highlighted segments are colored lightgrey when velocity_segment_only is True.
    
    Options:
      - plot_mode: 'all' to plot the entire trial or 'segment' to plot only from the first to the last highlight.
    
    Parameters:
        results (dict): The results dictionary containing trajectory data.
        subject (str): Subject key in the results dictionary.
        hand (str): Hand key ('right' or 'left') in the results dictionary.
        trial (int): The trial index to use for the main trajectory data.
        file_path (str): The file key for selecting trajectory data.
        overlay_trial (int): The trial index used to extract overlay indices for highlighting.
        velocity_segment_only (bool): If True, apply velocity-coded color only within highlighted segments.
        plot_mode (str): 'all' to plot the entire trial or 'segment' to plot only from the first to the last highlight.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as mcolors

    # Extract trajectory data for the given trial
    traj_data = results[subject][hand][trial][file_path]['traj_data']
    coord_prefix = "RFIN_" if hand == "right" else "LFIN_"
    coord_x = np.array(traj_data[coord_prefix + "X"])
    coord_y = np.array(traj_data[coord_prefix + "Y"])
    coord_z = np.array(traj_data[coord_prefix + "Z"])
    
    # Extract overlay index (or indices) from the overlay_trial
    overlay_index = results[subject][hand][overlay_trial][file_path]
    highlight_indices = overlay_index if isinstance(overlay_index, (list, np.ndarray)) else [overlay_index]
    highlight_indices = sorted(highlight_indices)
    
    n_points = len(coord_x)
    marker = "RFIN" if hand == "right" else "LFIN"

    # Extract instantaneous velocity from trajectory space (assume constant sampling rate = 200Hz)
    vel = results[subject][hand][trial][file_path]['traj_space'][marker][1]
    
    # Normalize velocities between 0 and 1
    v_min = np.min(vel)
    v_max = np.max(vel)
    if v_max - v_min > 0:
        v_norm = (vel - v_min) / (v_max - v_min)
    else:
        v_norm = np.ones_like(vel)
    
    # Map velocity to colors using the viridis colormap with an exponential scaling for contrast
    point_colors = [plt.cm.viridis(1 - (v_norm[i]**2)) for i in range(n_points)]
    
    # If velocity_segment_only is True, only retain velocity colors within highlighted segments.
    if velocity_segment_only and highlight_indices:
        segments = []
        for i in range(0, len(highlight_indices) - 1, 2):
            segments.append((highlight_indices[i], highlight_indices[i+1]))
        for i in range(n_points):
            in_segment = any(min(seg) <= i <= max(seg) for seg in segments)
            if not in_segment:
                point_colors[i] = mcolors.to_rgba('lightgrey')
    
    # Determine the indices to plot based on the plot_mode option
    if plot_mode == 'segment' and highlight_indices:
        start_idx = min(highlight_indices)
        end_idx = max(highlight_indices)
    else:
        start_idx = 0
        end_idx = n_points - 1

    plot_indices = np.arange(start_idx, end_idx + 1)
    coord_x_plot = coord_x[plot_indices]
    coord_y_plot = coord_y[plot_indices]
    coord_z_plot = coord_z[plot_indices]
    vel_plot    = np.array(vel)[plot_indices]
    colors_plot = [point_colors[i] for i in plot_indices]
    time_points = plot_indices / 200  # Time axis in seconds

    # Create the plot layout: two subplots arranged vertically (one top, one bottom)
    # Increase the bottom (3D) subplot by adjusting height_ratios and reduce vertical gap.
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 2.5])
    # Update subplot margins so both subplots align horizontally
    plt.subplots_adjust(left=0.1, right=0.9, hspace=0.01)
    
    # Top subplot: Plot instantaneous velocity over time.
    ax_vel = fig.add_subplot(gs[0, 0])
    ax_vel.scatter(time_points, vel_plot, c=colors_plot, marker='o', s=5)
    ax_vel.set_xlabel('Time (s)', fontsize=16)
    ax_vel.set_ylabel('Velocity\nmagnitude\n(mm/s)', fontsize=16)
    ax_vel.set_ylim([0, 1500])
    ax_vel.set_yticks([0, 750, 1500])
    ax_vel.set_yticklabels([0, 750, 1500], fontsize=16)
    ax_vel.spines['top'].set_visible(False)
    ax_vel.spines['right'].set_visible(False)
    ax_vel.grid(False)
    # Set the velocity axis on top
    ax_vel.set_zorder(2)
    ax_vel.patch.set_alpha(0.0)
    
    # Overlay markers at the designated highlight indices (if they fall within the plot range).
    for order, idx in enumerate(highlight_indices, start=1):
        if start_idx <= idx <= end_idx:
            t_val = idx / 200
            color = 'black' if order % 2 == 1 else 'grey'
            marker_sym = 'o'
            ax_vel.scatter(t_val, vel[idx], color=color, marker=marker_sym, s=40)
    
    # Bottom subplot: Plot the 3D trajectory.
    ax3d = fig.add_subplot(gs[1, 0], projection='3d')
    # Move the 3D plot more to the right side.
    pos = ax3d.get_position()
    ax3d.set_position([pos.x0 + 0.08, pos.y0, pos.width, pos.height])
    
    ax3d.scatter(coord_x_plot, coord_y_plot, coord_z_plot, c=colors_plot, marker='o', s=5)
    ax3d.set_xlabel("X (mm)", fontsize=16)
    ax3d.set_ylabel("Y (mm)", fontsize=16)
    ax3d.set_zlabel("Z (mm)", fontsize=16)
    ax3d.set_xticks([-250, 0, 250])
    ax3d.set_xticklabels([-250, 0, 250])
    ax3d.set_yticks([-50, 50, 150])
    ax3d.set_yticklabels([-50, 50, 150])
    ax3d.set_zticks([800, 950, 1100])
    ax3d.set_zticklabels([800, 950, 1100])
    ax3d.set_xlim([-250, 250])
    ax3d.set_ylim([-50, 150])
    ax3d.set_zlim([800, 1100])
    ax3d.set_box_aspect([10, 4, 6])
    # Set the 3D axis behind the velocity plot
    ax3d.set_zorder(1)
    
    # Overlay markers on the 3D plot corresponding to highlight indices.
    for order, idx in enumerate(highlight_indices, start=1):
        if start_idx <= idx <= end_idx:
            color = 'black' if order % 2 == 1 else 'grey'
            marker_sym = 'o'
            ax3d.scatter(coord_x[idx], coord_y[idx], coord_z[idx], color=color, marker=marker_sym, s=30)

    plt.tight_layout()
    plt.show()
    
    # Optional: Create an additional 3D plot for a selected segment if at least 4 highlight indices exist.
    if len(highlight_indices) >= 4:
        seg_start = highlight_indices[2]
        seg_end = highlight_indices[3]
        seg_indices = np.arange(seg_start, seg_end + 1)
        seg_coord_x = coord_x[seg_indices]
        seg_coord_y = coord_y[seg_indices]
        seg_coord_z = coord_z[seg_indices]
        seg_colors = [point_colors[i] for i in seg_indices]
        
        fig2 = plt.figure(figsize=(10, 8))
        ax3d_seg = fig2.add_subplot(111, projection='3d')
        ax3d_seg.scatter(seg_coord_x, seg_coord_y, seg_coord_z, c=seg_colors, marker='o', s=5)
        ax3d_seg.scatter(coord_x[seg_start], coord_y[seg_start], coord_z[seg_start],
                         color='green', marker='o', s=50, label='start')
        ax3d_seg.scatter(coord_x[seg_end], coord_y[seg_end], coord_z[seg_end],
                         color='blue', marker='X', s=50, label='end')
        ax3d_seg.set_xlabel(f"{coord_prefix}X (mm)", fontsize=14)
        ax3d_seg.set_ylabel(f"{coord_prefix}Y (mm)", fontsize=14)
        ax3d_seg.set_zlabel(f"{coord_prefix}Z (mm)", fontsize=14)
        ax3d_seg.legend()
        plt.tight_layout()
        plt.show()

plot_trajectory(results, subject='07/22/HW', hand='left', trial=1,
                file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv',
                overlay_trial=0, velocity_segment_only=True, plot_mode='segment')





# Hypothesis: Participants will be mmore accuracy when they take longer to complete placements, demonstrating a speed-accuracy trade-off.
def plot_speed_accuracy_tradeoff(cfg):
    """
    Plots a speed-accuracy tradeoff line plot using formatting directly from plot_config_summary.
    
    Parameters
    ----------
    x : array-like
        X-axis data (duration)
    y : array-like
        Y-axis data (error)
    cfg : dict
        Plot configuration dictionary (plot_config_summary)
    """
    x = np.linspace(0.5, 2.0, 100)   # Duration (s)
    y = -10 * x + 20                 # Error (mm)

    # Load settings from config
    general = cfg["general"]
    line_cfg = cfg["line"]
    axis_labels = cfg["axis_labels"]
    axis_colors = cfg["axis_colors"]
    label_offset = general.get("label_offset", 0.08)
    showGrid = general.get("showGrid", False)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=general["figsize"])

    # Plot line
    ax.plot(
        x,
        y,
        color='black',
        linewidth=line_cfg.get("linewidth", 2),
        marker='o' if line_cfg.get("show_markers", False) else None
    )

    # Set axis labels
    ax.set_xlabel(axis_labels.get("duration", "X"), fontsize=general["axis_label_font"])
    ax.set_ylabel(axis_labels.get("distance", "Y"), fontsize=general["axis_label_font"])

    # Remove numeric ticks if configured
    ax.set_xticks([])
    ax.set_yticks([]) 

    # Restore x/y spines
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Axis ranges for offsetting labels
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    # X-axis start/end labels
    x_cfg = axis_colors.get("x", {}).get(axis_labels["duration"], {})
    if x_cfg:
        ax.text(x[0], y.min() - label_offset*y_range, x_cfg["start"],
                color=x_cfg["colors"][0], ha="center", va="top",
                fontsize=general["tick_label_font"])
        ax.text(x[-1], y.min() - label_offset*y_range, x_cfg["end"],
                color=x_cfg["colors"][1], ha="center", va="top",
                fontsize=general["tick_label_font"])

    # Y-axis start/end labels
    y_cfg = axis_colors.get("y", {}).get(axis_labels["distance"], {})
    if y_cfg:
        ax.text(x.min() - label_offset*x_range, y[-1], y_cfg["start"],
                color=y_cfg["colors"][0], ha="right", va="center",
                fontsize=general["tick_label_font"])
        ax.text(x.min() - label_offset*x_range, y[0], y_cfg["end"],
                color=y_cfg["colors"][1], ha="right", va="center",
                fontsize=general["tick_label_font"])

    plt.tight_layout()
    plt.grid(showGrid)
    plt.show()

plot_speed_accuracy_tradeoff(plot_config_summary)

# # 2.1 within each placemnet location

def plot_reach_scatter_and_spearman(subject, hand, reach_index, config=plot_config_summary):
    """
    Plots a scatter plot of durations vs. distances for a given subject, hand, and reach index
    across trials and calculates the Spearman correlation and p-value.
    
    Also performs a Shapiro-Wilk normality test on the x (durations) and y (distances) data before calculating
    the Spearman correlation. A linear regression line is overlaid on the scatter plot.
    
    Parameters:
        subject (str): Subject identifier (e.g., "07/22/HW")
        hand (str): Hand identifier (e.g., "non_dominant")
        reach_index (int): Index of the reach (0-indexed)
        config (dict): Plot configuration dictionary
        
    Returns:
        tuple: Spearman correlation coefficient and p-value
    """

    durations = []
    distances = []

    # Gather duration and distance values
    for trial, rep_durations in updated_metrics_acorss_phases[subject][hand]['durations'].items():
        duration = rep_durations[reach_index]
        distance = updated_metrics_acorss_phases[subject][hand]['distance'][trial][reach_index]
        durations.append(duration)
        distances.append(distance)
    
    # Perform normality tests
    stat_dur, p_dur = shapiro(durations)
    stat_dist, p_dist = shapiro(distances)
    print(f"Normality test for durations: W = {stat_dur:.4f}, p-value = {p_dur:.4f}")
    print(f"Normality test for distances: W = {stat_dist:.4f}, p-value = {p_dist:.4f}")

    # Calculate Spearman correlation
    corr, pval = spearmanr(durations, distances)
    
    # Determine significance stars based on p-value
    if pval < 0.001:
        stars = "***"
    elif pval < 0.01:
        stars = "**"
    elif pval < 0.05:
        stars = "*"
    else:
        stars = "ns"

    # General settings
    gen = config['general']
    scatter_cfg = config['scatter']
    axis_labels = config['axis_labels']
    axis_colors = config['axis_colors']
    tick_direction = gen.get('tick_direction', 'out')
    
    fig, ax = plt.subplots(figsize=gen['figsize'])

    # Scatter points
    ax.scatter(
        durations,
        distances,
        s=gen['marker_size'],
        alpha=gen['alpha'],
        color="black"
    )
    
    # Overlay linear regression line based on the original durations (without jitter)
    durations_arr = np.array(durations)
    distances_arr = np.array(distances)
    if len(durations_arr) > 1:
        slope, intercept = np.polyfit(durations_arr, distances_arr, 1)
        # Create line values for the regression line
        x_line = np.linspace(min(durations_arr), max(durations_arr), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="black", linewidth=2, label="Linear regression")
        # ax.legend(fontsize=gen['tick_label_font'])
    
    # Axis labels
    ax.set_xlabel(axis_labels['duration'], fontsize=gen['axis_label_font'])
    ax.set_ylabel(axis_labels['distance'], fontsize=gen['axis_label_font'])
    
    # Axis ticks
    if gen['x_ticks']:
        ax.tick_params(axis='x', labelsize=gen['tick_label_font'], direction=tick_direction)
    else:
        ax.set_xticks([])
    if gen['y_ticks']:
        ax.tick_params(axis='y', labelsize=gen['tick_label_font'], direction=tick_direction)
    else:
        ax.set_yticks([])
    
    # Annotate Spearman correlation along with significance stars
    if scatter_cfg.get('annotate_corr', True):
        ax.text(
            0.55, 0.95,
            f"ρ = {corr:.2f} {stars}",
            transform=ax.transAxes,
            fontsize=gen['tick_label_font'],
            verticalalignment='top'
        )
        
    # Sample size annotation with unit as placements
    n = len(durations)
    ax.text(
        0.55, 0.75,
        f"n = {n} placements",
        transform=ax.transAxes,
        fontsize=gen['tick_label_font'],
        verticalalignment='bottom'
    )
    
    # Grid
    ax.grid(gen['show_grid'])
    
    # Always hide top and right spines
    if gen.get('hide_spines', True):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    # Apply axis color ramps
    if scatter_cfg.get('use_axis_colors', True):
        # X-axis color bar
        x_colors = axis_colors['x'].get(axis_labels['duration'], None)
        if x_colors:
            ax.annotate(
                x_colors['start'],
                xy=(0, -gen['label_offset']),
                xycoords=('axes fraction', 'axes fraction'),
                fontsize=gen['tick_label_font'],
                ha='left',
                va='top',
                color=x_colors['colors'][0]
            )
            ax.annotate(
                x_colors['end'],
                xy=(1, -gen['label_offset']),
                xycoords=('axes fraction', 'axes fraction'),
                fontsize=gen['tick_label_font'],
                ha='right',
                va='top',
                color=x_colors['colors'][-1]
            )
        
        # Y-axis color bar
        y_colors = axis_colors['y'].get(axis_labels['distance'], None)
        if y_colors:
            ax.annotate(
                y_colors['start'],
                xy=(-gen['label_offset'], 0),
                xycoords=('axes fraction', 'axes fraction'),
                fontsize=gen['tick_label_font'],
                ha='right',
                va='bottom',
                color=y_colors['colors'][0]
            )
            ax.annotate(
                y_colors['end'],
                xy=(-gen['label_offset'], 1),
                xycoords=('axes fraction', 'axes fraction'),
                fontsize=gen['tick_label_font'],
                ha='right',
                va='top',
                color=y_colors['colors'][-1]
            )
    
    plt.tight_layout()
    plt.show()
    
    return corr, pval

corr_value, p_value = plot_reach_scatter_and_spearman("07/22/HW", "non_dominant", 0, config=plot_config_summary)

# Calculate and return Spearman correlation, p-value, data points, and hyperbolic fit parameters (a, b) for durations vs distances for each subject, hand, and reach index
def calculate_duration_distance_reach_indices(updated_metrics):
    """
    Calculates Spearman correlation, p-value, data points, and hyperbolic fit parameters (a, b)
    for durations vs distances for each subject, hand, and reach index.

    Parameters:
        updated_metrics (dict): Updated metrics data.

    Returns:
        dict: Dictionary containing results for each subject, hand, and reach index.
    """
    results = {}

    for subject in updated_metrics.keys():
        results[subject] = {}
        for hand in ['non_dominant', 'dominant']:
            results[subject][hand] = {}
            for reach_index in range(16):
                x_values = []
                y_values = []

                trials = updated_metrics[subject][hand]['durations'].keys()

                for trial in trials:
                    trial_x = np.array(updated_metrics[subject][hand]['durations'][trial])
                    trial_y = np.array(updated_metrics[subject][hand]['distance'][trial])

                    if reach_index < len(trial_x) and reach_index < len(trial_y):
                        x_values.append(trial_x[reach_index])
                        y_values.append(trial_y[reach_index])

                # Remove NaN values
                x_values = np.array(x_values)
                y_values = np.array(y_values)
                valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
                x_values = x_values[valid_indices]
                y_values = y_values[valid_indices]

                # Calculate Spearman correlation
                if len(x_values) > 1 and len(y_values) > 1:
                    spearman_corr, p_value = spearmanr(x_values, y_values)
                else:
                    spearman_corr, p_value = np.nan, np.nan

                # Store results
                results[subject][hand][reach_index] = {
                    "spearman_corr": spearman_corr,
                    "p_value": p_value,
                    "data_points": len(x_values)
                }

    return results

SAT_corr_within_results = calculate_duration_distance_reach_indices(updated_metrics_acorss_phases)


def heatmap_spearman_correlation_reach_indices(results, hand="both", simplified=False, 
                                                return_medians=False, overlay_median=False, config=None):
    """
    Plots a heatmap of Spearman correlations for the specified hand(s) and optionally returns the column and row medians.
    Optionally overlays a green square on each row at the cell closest to the row median.
    
    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.
        hand (str): Which hand to plot; "non_dominant", "dominant", or "both". Default is "both".
        simplified (bool): If True, plots a compact version with no annotations and no subject labels.
                           When hand == "both", each hand is plotted as a subplot.
        return_medians (bool): If True, returns a dictionary containing column and row medians.
        overlay_median (bool): If True, overlays a green square on each row at the cell closest to the row median.
        config (dict): Plot configuration dictionary. If provided, the "heatmap" and "general" sub-dictionaries will be used.
        
    Returns:
        dict or None: If return_medians is True, returns a dictionary with keys corresponding to each hand 
                      (or the chosen hand) and values as dictionaries with 'column_medians' and 'row_medians'.
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    reach_indices = list(range(16))
    medians = {}

    # Get heatmap settings from config if provided, else use defaults
    if config is None:
        h_cfg = {}
        general_cfg = {"figsize": (5, 4), "axis_label_font": 14, "tick_label_font": 12}
    else:
        h_cfg = config.get("heatmap", {})
        general_cfg = config.get("general", {"figsize": (5, 4), "axis_label_font": 14, "tick_label_font": 12})
    # cmap = h_cfg.get("colormap", "coolwarm")
    # cmap = LinearSegmentedColormap.from_list("custom_diverging", [(223/255, 148/255, 157/255), (1, 1, 1), (111/255, 94/255, 79/255)])
    # User-provided colors
    end_left = "#ffd6e9"   # light pink
    mid_left = "#ffbcda"   # medium pink
    center   = "#ffffff"   # neutral beige
    mid_right = "#c79274"  # warm brown
    end_right = "#946656"  # dark brown

    # Arrange symmetrically around the center
    symmetric_colors = [end_left, mid_left, center, mid_right, end_right]

    # Create a diverging colormap
    cmap = mpl.colors.LinearSegmentedColormap.from_list("SymmetricPalette", symmetric_colors)

    show_colorbar = h_cfg.get("show_colorbar", True)
    colorbar_label = h_cfg.get("colorbar_label", "Correlation")
    center_zero = h_cfg.get("center_zero", True)
    vmin = -1 if center_zero else None
    vmax = 1 if center_zero else None
    # Prepare cbar_kws with an additional fontsize setting
    cbar_kws = {"label": colorbar_label, "ticks": np.linspace(vmin, vmax, 5)} if show_colorbar else None
    fig_size = (8, 6)
    axis_label_font = general_cfg.get("axis_label_font", 14)
    tick_label_font = general_cfg.get("tick_label_font", 12)

    # Define a legend patch for the median overlay (black square)
    median_patch = Patch(facecolor='none', edgecolor='black', lw=2, label='Participant\nmedian\ncorrelation')
    
    if hand == "both":
        if simplified:
            fig, axes = plt.subplots(1, 2, figsize=fig_size)
        else:
            fig, axes = plt.subplots(2, 1, figsize=fig_size)
        
        for idx, h in enumerate(["non_dominant", "dominant"]):
            subjects = list(results.keys())
            data = []
            for subject in subjects:
                if h in results[subject]:
                    correlations = [
                        results[subject][h].get(ri, {}).get("spearman_corr", np.nan)
                        for ri in reach_indices
                    ]
                    data.append(correlations)
            df = pd.DataFrame(data, index=subjects, columns=reach_indices)
            ax = axes[idx] if isinstance(axes, (list, np.ndarray)) else axes
            sns.heatmap(
                df,
                annot=not simplified,
                fmt=".2f",
                cmap=cmap,
                cbar=show_colorbar,
                cbar_kws=cbar_kws,
                xticklabels=list(range(1, 17)),
                yticklabels=[] if simplified else subjects,
                vmin=vmin,
                vmax=vmax,
                ax=ax
            )
            # Adjust colorbar tick label font size
            if show_colorbar and ax.collections:
                cbar = ax.collections[0].colorbar
                if cbar is not None:
                    cbar.ax.tick_params(labelsize=tick_label_font)
                    cbar.ax.yaxis.label.set_size(tick_label_font)

            ax.set_xlabel("Location", fontsize=axis_label_font)
            ax.set_xticklabels(range(1, 17), fontsize=tick_label_font, rotation=0)
            ax.set_ylabel("Subjects", fontsize=axis_label_font)
            if overlay_median:
                import matplotlib.patches as patches
                for i, subject in enumerate(df.index):
                    row_data = df.loc[subject].dropna()
                    if row_data.empty:
                        continue
                    median_val = np.median(row_data.values)
                    col_idx = np.argmin(np.abs(df.loc[subject].values - median_val))
                    ax.add_patch(patches.Rectangle((col_idx, i), 1, 1, fill=False, edgecolor='black', lw=2))
                # Add legend explaining the black square
                ax.legend(handles=[median_patch], loc='upper right', fontsize=tick_label_font)
            if return_medians:
                medians[h] = {
                    "column_medians": df.median(axis=0).to_dict(),
                    "row_medians": df.median(axis=1).to_dict()
                }
        plt.tight_layout()
        plt.show()
    
    else:
        subjects = list(results.keys())
        data = []
        for subject in subjects:
            if hand in results[subject]:
                correlations = [
                    results[subject][hand].get(ri, {}).get("spearman_corr", np.nan)
                    for ri in reach_indices
                ]
                data.append(correlations)
        fig, ax = plt.subplots(figsize=fig_size)
        df = pd.DataFrame(data, index=subjects, columns=reach_indices)
        sns.heatmap(
            df,
            annot=not simplified,
            fmt=".2f",
            cmap=cmap,
            cbar=show_colorbar,
            cbar_kws=cbar_kws,
            xticklabels=list(range(1, 17)),
            yticklabels=[] if simplified else subjects,
            vmin=vmin,
            vmax=vmax,
            ax=ax
        )
        # Adjust colorbar tick label font size
        if show_colorbar and ax.collections:
            cbar = ax.collections[0].colorbar
            if cbar is not None:
                cbar.ax.tick_params(labelsize=tick_label_font)
                cbar.ax.yaxis.label.set_size(tick_label_font)

        ax.set_xlabel("Location", fontsize=axis_label_font)
        ax.set_xticklabels(range(1, 17), fontsize=tick_label_font, rotation=0)
        ax.set_ylabel("Participant", fontsize=axis_label_font)
        ax.set_yticklabels([] if simplified else ax.get_yticklabels())
        
        # Insert the placement location icon on the right side of the plot
        try:
            icon_img = mpimg.imread('/Users/yilinwu/Desktop/Thesis/PlacementLocationIcon_RBOX.png')
            imagebox = OffsetImage(icon_img, zoom=0.2)
            ab = AnnotationBbox(imagebox, (1.25, -0.18), xycoords='axes fraction', frameon=False)
            ax.add_artist(ab)
        except Exception as e:
            print("Error loading icon image:", e)
        
        if overlay_median:
            import matplotlib.patches as patches
            for i, subject in enumerate(df.index):
                row_data = df.loc[subject].dropna()
                if row_data.empty:
                    continue
                median_val = np.median(row_data.values)
                col_idx = np.argmin(np.abs(df.loc[subject].values - median_val))
                ax.add_patch(patches.Rectangle((col_idx, i), 1, 1, fill=False, edgecolor='black', lw=2))
            # Add legend for the black square overlay
            ax.legend(handles=[median_patch], loc=(0.8, -0.27), fontsize=tick_label_font, frameon=False)
        plt.tight_layout()
        plt.show()
        if return_medians:
            medians[hand] = {
                "column_medians": df.median(axis=0).to_dict(),
                "row_medians": df.median(axis=1).to_dict()
            }
    
    if return_medians:
        return medians

heatmap_medians = heatmap_spearman_correlation_reach_indices(
    SAT_corr_within_results, hand="non_dominant", simplified=True, 
    return_medians=True, overlay_median=True, config=plot_config_summary
)

def boxplot_spearman_corr_with_stats_reach_indices_by_subject(results, config=plot_config_summary):
    """
    Creates a box plot of median Spearman correlations for each subject,
    separated by non_dominant and dominant hands, using the formatting defined in plot_config_summary.
    Also annotates significance (using stars) by drawing a common horizontal line above both boxes for
    hand vs 0 comparisons and a line across both boxes for paired between-hand comparisons with stars indicating significance.
    
    Multiple comparisons are corrected using the Benjamini–Hochberg FDR procedure.
    
    Additionally, it reports the Wilcoxon signed‐rank test for each comparison with the median scores for each group,
    the sample size (N), the degrees of freedom (N - 1), the Z-statistic (rounded to two decimal places),
    the exact p-value, and the effect size (r).
    
    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.
        config (dict): Plot configuration dictionary.
    """
    import matplotlib.pyplot as plt

    # Extract configuration settings.
    general_cfg = config.get("general", {"figsize": (5, 4), "axis_label_font": 14, "tick_label_font": 14, "alpha": 0.4})
    box_cfg = config.get("box", {"bar_colors": {"Non-dominant": "#A9A9A9", "Dominant": "#F0F0F0"},
                                 "sig_levels": [(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")],
                                 "sig_y_loc": 90,
                                 "sig_line": True,
                                 "sig_line_width": 1.5,
                                 "sig_line_color": "black",
                                 "sig_marker_size": 40,
                                 "sig_text_offset": -0.05,
                                 "hand_sig_offset": 0.2,
                                 "group_sig_offset": 0.4})
    figsize = general_cfg.get("figsize", (5, 4))
    axis_label_font = general_cfg.get("axis_label_font", 14)
    tick_label_font = general_cfg.get("tick_label_font", 14)
    alpha = general_cfg.get("alpha", 0.4)
    sig_marker_size = box_cfg.get("sig_marker_size", 40)
    sig_text_offset = box_cfg.get("sig_text_offset", -0.05)
    
    # New offset parameters for significance lines and text
    hand_sig_offset = box_cfg.get("hand_sig_offset", 0.2)
    group_sig_offset = box_cfg.get("group_sig_offset", 0.4)
    
    # Collect median correlations for each subject for both hands.
    median_corr_non = []
    median_corr_dom = []
    paired_non = []
    paired_dom = []
    for subject in results.keys():
        subject_non = None
        subject_dom = None
        for hand in ['non_dominant', 'dominant']:
            if hand in results[subject]:
                # Gather correlations for reach indices 0-15.
                correlations = [
                    results[subject][hand].get(reach_index, {}).get("spearman_corr", float('nan'))
                    for reach_index in range(16)
                ]
                # Remove NaNs.
                correlations = [r for r in correlations if r is not None and not (isinstance(r, float) and np.isnan(r))]
                if correlations:
                    med_corr = np.median(correlations)
                    if hand == 'non_dominant':
                        median_corr_non.append(med_corr)
                        subject_non = med_corr
                    elif hand == 'dominant':
                        median_corr_dom.append(med_corr)
                        subject_dom = med_corr
        # Collect only paired subjects.
        if (subject_non is not None) and (subject_dom is not None):
            paired_non.append(subject_non)
            paired_dom.append(subject_dom)
    
    # Create a DataFrame for plotting.
    data = []
    for x in median_corr_non:
        data.append({"Hand": "Non-dominant", "Correlation": x})
    for x in median_corr_dom:
        data.append({"Hand": "Dominant", "Correlation": x})
    df_plot = pd.DataFrame(data)
    
    # Define hand order and retrieve the custom palette from box config.
    hand_order = ["Non-dominant", "Dominant"]
    palette = box_cfg.get("bar_colors", {"Non-dominant": "#A9A9A9", "Dominant": "#F0F0F0"})
    
    # Plotting using the configuration settings.
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(x="Hand", y="Correlation", data=df_plot,
                order=hand_order, palette=palette, ax=ax, linewidth=1.5)

    sns.swarmplot(x="Hand", y="Correlation", data=df_plot,
                  order=hand_order, color='black', size=6, alpha=alpha, ax=ax)

    ax.set_xlabel("Hand", fontsize=axis_label_font)
    # Set y-axis label to just "Correlation"
    ax.set_ylabel("Correlation", fontsize=axis_label_font)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([-0.5, 0, 0.5])
    ax.tick_params(axis='y', labelsize=tick_label_font)
    # Set x tick label size as 14
    ax.tick_params(axis='x', labelsize=14)
    
    # Annotate sample size (n) at position (0.75, 0.95) with unit participants.
    n = len(results)
    ax.text(0.75, 0.95, f"n = {n} participants", transform=ax.transAxes,
            ha="center", va="center", fontsize=tick_label_font)

    # ---------------------------
    # Compute individual tests
    # ---------------------------
    computed_p = {}
    group_stars = "ns"
    stars_non = "ns"
    stars_dom = "ns"
    
    # Group comparison: paired test between non-dominant and dominant.
    if (len(paired_non) > 0) and (len(paired_dom) > 0):
        try:
            stat_group, p_group = wilcoxon(paired_non, paired_dom)
            computed_p["group"] = p_group
            
            med_nd_group = np.median(paired_non) if paired_non else 0
            med_dom_group = np.median(paired_dom) if paired_dom else 0
            group_line_y = max(med_nd_group, med_dom_group) + group_sig_offset
        except Exception:
            ax.text(0.5, 0.95, "Group comparison failed", ha="center", va="bottom", 
                    fontsize=sig_marker_size, transform=ax.get_xaxis_transform())
    # Non-dominant vs 0.
    if median_corr_non:
        try:
            stat_non, p_non = wilcoxon(median_corr_non)
            computed_p["non_dominant"] = p_non
        except Exception:
            ax.text(0, 0.95, "Non-dominant: NA", ha="center", va="bottom", 
                    fontsize=sig_marker_size, transform=ax.get_xaxis_transform())
    else:
        ax.text(0, 0.95, "No data (Non-dominant)", ha="center", va="bottom", 
                fontsize=sig_marker_size, transform=ax.get_xaxis_transform())
    # Dominant vs 0.
    if median_corr_dom:
        try:
            stat_dom, p_dom = wilcoxon(median_corr_dom)
            computed_p["dominant"] = p_dom
        except Exception:
            ax.text(1, 0.95, "Dominant: NA", ha="center", va="bottom", 
                    fontsize=sig_marker_size, transform=ax.get_xaxis_transform())
    else:
        ax.text(1, 0.95, "No data (Dominant)", ha="center", va="bottom", 
                fontsize=sig_marker_size, transform=ax.get_xaxis_transform())
    
    # Apply Benjamini–Hochberg FDR correction to the collected p-values.
    if computed_p:
        keys = list(computed_p.keys())
        pvals = [computed_p[k] for k in keys]
        _, pvals_adj, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        computed_p_adj = dict(zip(keys, pvals_adj))
    else:
        computed_p_adj = {}

    # Determine significance stars using adjusted p-values.
    def get_stars(pvalue, levels):
        for threshold, symbol in levels:
            if pvalue < threshold:
                return symbol
        return "ns"
    
    if "group" in computed_p_adj:
        group_stars = get_stars(computed_p_adj["group"], box_cfg.get("sig_levels", [(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")]))
    if "non_dominant" in computed_p_adj:
        stars_non = get_stars(computed_p_adj["non_dominant"], box_cfg.get("sig_levels", [(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")]))
    if "dominant" in computed_p_adj:
        stars_dom = get_stars(computed_p_adj["dominant"], box_cfg.get("sig_levels", [(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")]))
    
    # ---------------------------
    # Annotate Group Comparison
    # ---------------------------
    if ("group" in computed_p_adj) and (len(paired_non) > 0):
        effective_group_size = sig_marker_size if group_stars != "ns" else 16
        med_nd_group = np.median(paired_non) if paired_non else 0
        med_dom_group = np.median(paired_dom) if paired_dom else 0
        line_y = max(med_nd_group, med_dom_group) + group_sig_offset
        if box_cfg.get("sig_line", True):
            ax.plot([0, 1], [line_y, line_y], color=box_cfg.get("sig_line_color", "black"),
                    linewidth=box_cfg.get("sig_line_width", 1.5))
        text_offset = abs(sig_text_offset) if group_stars == "ns" else sig_text_offset
        ax.text(0.5, line_y + text_offset, group_stars, ha="center", va="bottom", fontsize=effective_group_size)
    
    # ---------------------------
    # Annotate Non-dominant vs 0
    # ---------------------------
    if median_corr_non:
        effective_non_size = sig_marker_size if stars_non != "ns" else 16
        text_offset_non = abs(sig_text_offset) if stars_non == "ns" else sig_text_offset
        ax.text(0, np.median(median_corr_non) + hand_sig_offset + text_offset_non, 
                stars_non, ha="center", va="bottom", fontsize=effective_non_size)
    # ---------------------------
    # Annotate Dominant vs 0
    # ---------------------------
    if median_corr_dom:
        effective_dom_size = sig_marker_size if stars_dom != "ns" else 16
        text_offset_dom = abs(sig_text_offset) if stars_dom == "ns" else sig_text_offset
        ax.text(1, np.median(median_corr_dom) + hand_sig_offset + text_offset_dom,
                stars_dom, ha="center", va="bottom", fontsize=effective_dom_size)
        
    for spine in ["right", "top"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(top=False)
    ax.axhline(0.5, color='white', linestyle='-', linewidth=2)

    plt.tight_layout()
    plt.show()
    
    # ---- Reporting Wilcoxon Test Details ----
    print("\nWilcoxon Signed-Rank Test Reports:")
    # For paired between-hand comparisons.
    if (len(paired_non) > 0) and (len(paired_dom) > 0) and ("group" in computed_p_adj):
        N_group = len(paired_non)
        df_group = N_group - 1
        med_nd_group = np.median(paired_non)
        med_dom_group = np.median(paired_dom)
        expected_group = N_group * (N_group + 1) / 4
        sd_group = math.sqrt(N_group * (N_group + 1) * (2 * N_group + 1) / 24)
        # Recalculate original group test for reporting if possible.
        try:
            stat_group, _ = wilcoxon(paired_non, paired_dom)
        except Exception:
            stat_group = float('nan')
        Z_group = (stat_group - expected_group) / sd_group if sd_group > 0 else float('nan')
        r_group = abs(Z_group) / math.sqrt(N_group)
        print(f"Paired Comparison (Non-dominant vs Dominant):")
        print(f"  Non-dominant median = {med_nd_group}, Dominant median = {med_dom_group}, N = {N_group}, df = {df_group}")
        print(f"  Adjusted p = {computed_p_adj['group']}, Z = {Z_group:.2f}, Effect Size (r) = {r_group:.2f}")
    else:
        print("Not enough paired data for between-hand comparison.")

    # For non-dominant vs 0.
    if median_corr_non and ("non_dominant" in computed_p_adj):
        N_nd = len(median_corr_non)
        df_nd = N_nd - 1
        med_nd = np.median(median_corr_non)
        expected_nd = N_nd*(N_nd+1)/4
        sd_nd = math.sqrt(N_nd*(N_nd+1)*(2*N_nd+1)/24)
        try:
            stat_non, _ = wilcoxon(median_corr_non)
        except Exception:
            stat_non = float('nan')
        Z_nd = (stat_non - expected_nd) / sd_nd if sd_nd > 0 else float('nan')
        r_nd = abs(Z_nd) / math.sqrt(N_nd)
        print(f"\nNon-dominant vs 0:")
        print(f"  Median = {med_nd}, N = {N_nd}, df = {df_nd}, IQR = {np.percentile(median_corr_non, 75) - np.percentile(median_corr_non, 25)}")
        print(f"  Adjusted p = {computed_p_adj['non_dominant']}, Z = {Z_nd:.2f}, Effect Size (r) = {r_nd:.2f}")
    else:
        print("\nNo data for non-dominant vs 0 comparison.")

    # For dominant vs 0.
    if median_corr_dom and ("dominant" in computed_p_adj):
        N_dom = len(median_corr_dom)
        df_dom = N_dom - 1
        med_dom = np.median(median_corr_dom)
        expected_dom = N_dom*(N_dom+1)/4
        sd_dom = math.sqrt(N_dom*(N_dom+1)*(2*N_dom+1)/24)
        try:
            stat_dom, _ = wilcoxon(median_corr_dom)
        except Exception:
            stat_dom = float('nan')
        Z_dom = (stat_dom - expected_dom) / sd_dom if sd_dom > 0 else float('nan')
        r_dom = abs(Z_dom) / math.sqrt(N_dom)
        print(f"\nDominant vs 0:")
        print(f"  Median = {med_dom}, N = {N_dom}, df = {df_dom}, IQR = {np.percentile(median_corr_dom, 75) - np.percentile(median_corr_dom, 25)}")
        print(f"  Adjusted p = {computed_p_adj['dominant']}, Z = {Z_dom:.2f}, Effect Size (r) = {r_dom:.2f}")
    else:
        print("\nNo data for dominant vs 0 comparison.")

boxplot_spearman_corr_with_stats_reach_indices_by_subject(SAT_corr_within_results, config=plot_config_summary)
# -------------------------------------------------------------------------------------------------------------------
# # 2.2 across placemnet location
# # -------------------------------------------------------------------------------------------------------------------
def calculate_trials_mean_median_of_reach_indices(updated_metrics, metric_x, metric_y):
    """
    Calculates average and median 'durations', and average and median 'distance' for each reach_index
    for all subjects and hands, and returns two dictionaries: one for mean and one for median.

    Parameters:
        updated_metrics (dict): Updated metrics data.
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.

    Returns:
        tuple: (mean_statistics, median_statistics) dictionaries containing statistics for all subjects and hands.
    """
    mean_statistics = {}
    median_statistics = {}

    for subject, hands_data in updated_metrics.items():
        mean_statistics[subject] = {}
        median_statistics[subject] = {}
        for hand, metrics in hands_data.items():
            mean_statistics[subject][hand] = []
            median_statistics[subject][hand] = []

            for reach_index in range(16):
                x_values = []
                y_values = []

                trials = metrics[metric_x].keys()

                for trial in trials:
                    trial_x = np.array(metrics[metric_x][trial])
                    trial_y = np.array(metrics[metric_y][trial])

                    # Collect data for the specified reach index
                    if reach_index < len(trial_x) and reach_index < len(trial_y):
                        x_values.append(trial_x[reach_index])
                        y_values.append(trial_y[reach_index])
                
                # Remove NaN values
                x_values = np.array(x_values)
                y_values = np.array(y_values)
                valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
                x_values = x_values[valid_indices]
                y_values = y_values[valid_indices]

                # Calculate statistics
                avg_duration = np.mean(x_values) if len(x_values) > 0 else np.nan
                median_duration = np.median(x_values) if len(x_values) > 0 else np.nan
                avg_distance = np.mean(y_values) if len(y_values) > 0 else np.nan
                median_distance = np.median(y_values) if len(y_values) > 0 else np.nan

                mean_statistics[subject][hand].append({
                    "reach_index": reach_index,
                    "avg_duration": avg_duration,
                    "avg_distance": avg_distance
                })

                median_statistics[subject][hand].append({
                    "reach_index": reach_index,
                    "median_duration": median_duration,
                    "median_distance": median_distance
                })

    return mean_statistics, median_statistics
mean_stats, median_stats = calculate_trials_mean_median_of_reach_indices(updated_metrics_acorss_phases, 'durations', 'distance')

def plot_trials_mean_median_of_reach_indices(stats, subject, hand, metric_x, metric_y, stat_type="avg", use_unique_colors=False, config=plot_config_summary, marker_style='s'):
    """
    Overlays scatter plots for all reach indices in a single plot using either mean or median statistics.
    For each reach index (1 to 16), it uses the colors provided by placement_location_colors.
    Calculates and returns the Pearson correlation for the overlayed points.
    
    Parameters:
        stats (dict): Statistics (mean or median) for all subjects and hands.
        subject (str): Subject identifier.
        hand (str): Hand ('non_dominant' or 'dominant').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        stat_type (str): Type of statistics to use ("mean" or "median").
        use_unique_colors (bool): Ignored in this implementation.
        config (dict): Plot configuration dictionary.
        marker_style (str): Marker style for points (default: 's' for square).
        
    Returns:
        tuple: Pearson correlation and p-value for the overlayed points.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.image as mpimg
    from scipy.stats import pearsonr  # Changed: using Pearson correlation

    # Extract configuration values
    general_cfg = config.get("general", {})
    axis_labels = config.get("axis_labels", {})
    figsize = general_cfg.get("figsize", (5, 4))
    axis_label_font = general_cfg.get("axis_label_font", 14)
    tick_label_font = general_cfg.get("tick_label_font", 14)
    showGrid = general_cfg.get("showGrid", False)
    tick_direction = general_cfg.get("tick_direction", "out")
    
    plt.figure(figsize=figsize)
    
    x_values = []
    y_values = []
    
    # For each reach index (0 to 15), use placement_location_colors from globals.
    for reach_index in range(16):
        duration = stats[subject][hand][reach_index][f"{stat_type}_duration"]
        distance = stats[subject][hand][reach_index][f"{stat_type}_distance"]
        if not np.isnan(duration) and not np.isnan(distance):
            x_values.append(duration)
            y_values.append(distance)
            # Get the color from placement_location_colors 
            color = placement_location_colors[reach_index]
            # Plot the scatter with a colored square (no annotation text)
            plt.scatter(duration, distance, facecolors=color, edgecolors=color, s=100,
                        zorder=5, alpha=1.0, marker=marker_style)
    
    # Add linear regression line in black if sufficient data points exist.
    if len(x_values) > 1 and len(y_values) > 1:
        slope, intercept = np.polyfit(x_values, y_values, 1)
        line_x = np.linspace(min(x_values), max(x_values), 100)
        line_y = slope * line_x + intercept
        plt.plot(line_x, line_y, color='black', linestyle='-', linewidth=2)
    
    # Calculate Pearson correlation if enough points are available
    if len(x_values) > 1 and len(y_values) > 1:
        pearson_corr, p_value = pearsonr(x_values, y_values)
    else:
        pearson_corr, p_value = np.nan, np.nan

    # Determine significance stars based on p_value using sig_levels=[(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")]
    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = "ns"

    # Set axis labels from configuration
    xlabel = axis_labels.get("duration", "Duration (s)")
    ylabel = axis_labels.get("distance", "Error (mm)")
    plt.xlabel(xlabel, fontsize=axis_label_font)
    plt.ylabel(ylabel, fontsize=axis_label_font)
    
    # Apply axis color annotations similar to plot_reach_scatter_and_spearman
    ax = plt.gca()
    axis_colors = config.get("axis_colors", {})
    # X-axis annotations (unchanged)
    x_cfg = axis_colors.get("x", {}).get(xlabel, None)
    if x_cfg:
        label_offset = general_cfg.get("label_offset", 4)
        ax.annotate(
            x_cfg["start"],
            xy=(0, -label_offset),
            xycoords=('axes fraction', 'axes fraction'),
            fontsize=tick_label_font,
            ha="left",
            va="top",
            color=x_cfg["colors"][0]
        )
        ax.annotate(
            x_cfg["end"],
            xy=(1, -label_offset),
            xycoords=('axes fraction', 'axes fraction'),
            fontsize=tick_label_font,
            ha="right",
            va="top",
            color=x_cfg["colors"][-1]
        )
    # Y-axis annotations: swap to put "inaccurate" at the top and "accurate" at the bottom.
    y_cfg = axis_colors.get("y", {}).get(ylabel, None)
    if y_cfg:
        label_offset = general_cfg.get("label_offset", 4)
        ax.annotate(
            y_cfg["end"],
            xy=(-label_offset, 1-0.07),
            xycoords=('axes fraction', 'axes fraction'),
            fontsize=tick_label_font,
            ha="right",
            va="top",
            color=y_cfg["colors"][-1]
        )
        ax.annotate(
            y_cfg["start"],
            xy=(-label_offset, 0+0.07),
            xycoords=('axes fraction', 'axes fraction'),
            fontsize=tick_label_font,
            ha="right",
            va="bottom",
            color=y_cfg["colors"][0]
        )
    
    # Set fixed axis limits and ticks (as per original design)
    plt.xlim(0.75, 0.95)
    plt.xticks([0.75, 0.85, 0.95], fontsize=tick_label_font)
    plt.ylim(1.7, 3.5)
    plt.yticks([1.7, 2.6, 3.5], fontsize=tick_label_font)
    
    # Adjust tick direction
    plt.tick_params(axis='both', which='both', direction=tick_direction)
    
    # Optionally hide grid and spines based on config
    plt.grid(showGrid)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    
    # Insert the placement location icon on the right side of the plot
    try:
        icon_img = mpimg.imread('/Users/yilinwu/Desktop/Thesis/PlacementLocationIcon_RBOX.png')
        imagebox = OffsetImage(icon_img, zoom=0.2)
        ab = AnnotationBbox(imagebox, (0.85, 0.15), xycoords='axes fraction', frameon=False)
        ax.add_artist(ab)
    except Exception as e:
        print("Error loading icon image:", e)
    
    # Only label the correlation if significant
    if stars == "ns":
        r_text = f"r = {pearson_corr:.2f}"
    else:
        r_text = f"r = {pearson_corr:.2f} {stars}"
    ax.text(0.7, 0.4, r_text, transform=ax.transAxes, fontsize=tick_label_font)
    ax.text(0.7, 0.3, f"n = {len(x_values)} locations", transform=ax.transAxes, fontsize=tick_label_font)
    
    # Print correlation info in console as well
    print(f"Overlay of Reach Statistics ({subject}, {hand.capitalize()}, {stat_type.capitalize()}):")
    print(f"Pearson Corr: {pearson_corr:.2f} {stars if stars!='ns' else ''}, P-value: {p_value:.2f}")
    
    plt.show()
    
    return pearson_corr, p_value
plot_trials_mean_median_of_reach_indices(median_stats, '07/22/HW', 'non_dominant', 'durations', 'distance', stat_type="median", use_unique_colors=True, config=plot_config_summary, marker_style='s')


# Calculate Pearson correlation, p-value, data points, and hyperbolic fit parameters (a, b) for durations vs distances for each subject and hand across reach indices
def calculate_duration_distance_trials_mean_median_of_reach_indices(stats, selected_subjects=None, stat_type="avg"):
    """
    Calculates Pearson correlation, p-value, data points, and hyperbolic fit parameters (a, b)
    for reach statistics (e.g., durations vs distances) for each subject and hand.

    Parameters:
        stats (dict): Statistics (mean or median) for all subjects and hands.
        selected_subjects (list or None): List of subjects to include. If None, includes all subjects.
        stat_type (str): Type of statistics to use ("mean" or "median").

    Returns:
        dict: Dictionary containing results for each subject and hand.
    """
    if selected_subjects is None:
        selected_subjects = stats.keys()

    results = {}

    for subject in selected_subjects:
        if subject in stats:
            results[subject] = {}
            for hand in ['non_dominant', 'dominant']:
                x_values = []
                y_values = []

                for reach_index in range(16):
                    # Get statistics for the current reach index
                    duration = stats[subject][hand][reach_index].get(f"{stat_type}_duration", np.nan)
                    distance = stats[subject][hand][reach_index].get(f"{stat_type}_distance", np.nan)

                    if not np.isnan(duration) and not np.isnan(distance):
                        x_values.append(duration)
                        y_values.append(distance)

                # Calculate Pearson correlation only if there are enough valid data points
                if len(x_values) > 1 and len(y_values) > 1:
                    pearson_corr, p_value = pearsonr(x_values, y_values)
                else:
                    pearson_corr, p_value = np.nan, np.nan

                # Fit a hyperbolic curve
                def hyperbolic_func(x, a, b):
                    return a / (x + b)

                try:
                    params, _ = curve_fit(hyperbolic_func, x_values, y_values)
                    a, b = params
                except Exception:
                    a, b = np.nan, np.nan

                # Store results
                results[subject][hand] = {
                    "pearson_corr": pearson_corr,
                    "p_value": p_value,
                    "data_points": len(x_values),
                    "hyperbolic_fit_a": a,
                    "hyperbolic_fit_b": b
                }

    return results

# Calculate Pearson correlation, p-value, data points, and hyperbolic fit parameters
SAT_corr_acorss_results = calculate_duration_distance_trials_mean_median_of_reach_indices(median_stats, stat_type='median')

def boxplot_pearson_corr_trials_mean_median_of_reach_indices(results, config=None, orientation="vertical"):
    """
    Creates a box plot of Pearson correlations for durations vs distances across all subjects,
    separated by non-dominant and dominant hands. Annotates significance:
      - Each hand vs 0 (Wilcoxon signed‐rank)
      - Between‐hand comparison (paired Wilcoxon on Fisher z-transformed correlations)
    Multiple comparisons are corrected using the Benjamini–Hochberg FDR procedure.
    
    Also prints statistics results including Median, n, df, IQR, z, p, and effect size.
    
    Parameters:
        results (dict): Results containing Pearson correlations for each subject and hand.
        config (dict): Plot configuration dictionary.
        orientation (str): 'vertical' or 'horizontal' orientation of the box plot.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests
    import math

    if config is None:
        config = {}
    
    # ---------------------
    # Configuration
    # ---------------------
    general_cfg = config.get("general", {"figsize": (5, 4),
                                         "axis_label_font": 14,
                                         "tick_label_font": 14,
                                         "alpha": 0.4})
    box_cfg = config.get("box", {
        "bar_colors": {"Non-dominant": "#A9A9A9", "Dominant": "#F0F0F0"},
        "sig_levels": [(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")],
        "sig_y_loc": 90,
        "sig_line": True,
        "sig_line_width": 1.5,
        "sig_line_color": "black",
        "sig_marker_size": 40,
        "sig_text_offset": -0.05,
        "hand_sig_offset": 0.2,
        "group_sig_offset": 0.4
    })
    
    figsize = general_cfg.get("figsize", (5, 4))
    axis_label_font = general_cfg.get("axis_label_font", 14)
    tick_label_font = general_cfg.get("tick_label_font", 14)
    alpha = general_cfg.get("alpha", 0.4)
    sig_marker_size = box_cfg.get("sig_marker_size", 40)
    sig_text_offset = box_cfg.get("sig_text_offset", 0.2)
    hand_sig_offset = box_cfg.get("hand_sig_offset", 0.4)
    group_sig_offset = box_cfg.get("group_sig_offset", 0.7)

    # ---------------------
    # Gather median correlations per subject per hand
    # ---------------------
    median_corr_non = []
    median_corr_dom = []
    for subject, hands_data in results.items():
        for hand in ['non_dominant', 'dominant']:
            if hand in hands_data:
                corr = hands_data[hand].get("pearson_corr", float('nan'))
                if corr is not None and not (isinstance(corr, float) and np.isnan(corr)):
                    if hand == 'non_dominant':
                        median_corr_non.append(corr)
                    else:
                        median_corr_dom.append(corr)

    # Build DataFrame for plotting
    data = ([{"Hand": "Non-dominant", "Correlation": x} for x in median_corr_non] +
            [{"Hand": "Dominant", "Correlation": x} for x in median_corr_dom])
    df_plot = pd.DataFrame(data)

    hand_order = ["Non-dominant", "Dominant"]
    palette = box_cfg.get("bar_colors", {"Non-dominant": "#A9A9A9", "Dominant": "#F0F0F0"})

    # ---------------------
    # Plotting
    # ---------------------
    fig, ax = plt.subplots(figsize=figsize)
    if orientation.lower() == "horizontal":
        sns.boxplot(y="Hand", x="Correlation", data=df_plot, order=hand_order,
                    palette=palette, ax=ax, linewidth=1.5)
        sns.swarmplot(y="Hand", x="Correlation", data=df_plot,
                      order=hand_order, color='black', size=6, alpha=alpha, ax=ax)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Correlation", fontsize=axis_label_font)
        ax.set_ylabel("Hand", fontsize=axis_label_font)
        ax.set_xticks([-1, 0, 1])
        ax.tick_params(axis='x', labelsize=tick_label_font)
        ax.tick_params(axis='y', labelsize=14)
    else:
        sns.boxplot(x="Hand", y="Correlation", data=df_plot, order=hand_order,
                    palette=palette, ax=ax, linewidth=1.5)
        sns.swarmplot(x="Hand", y="Correlation", data=df_plot,
                      order=hand_order, color='black', size=6, alpha=alpha, ax=ax)
        ax.set_ylim(-1, 1)
        ax.set_ylabel("Correlation", fontsize=axis_label_font)
        ax.set_xlabel("Hand", fontsize=axis_label_font)
        ax.set_yticks([-1, 0, 1])
        ax.tick_params(axis='y', labelsize=tick_label_font)
        ax.tick_params(axis='x', labelsize=14)

    # Annotate sample size
    n = len(median_corr_non)
    ax.text(0.75, 0.15, f"n = {n} participants", transform=ax.transAxes,
            ha="center", va="center", fontsize=tick_label_font)

    # ---------------------
    # Between-hand comparison (paired Wilcoxon on Fisher z)
    # ---------------------
    # Only if sample sizes match and at least 4 pairs
    if len(median_corr_non) == len(median_corr_dom) and len(median_corr_non) > 3:
        z_non = np.arctanh(median_corr_non)
        z_dom = np.arctanh(median_corr_dom)
        try:
            stat_group, p_group = wilcoxon(z_dom, z_non)
        except Exception:
            p_group = np.nan
            stat_group = np.nan
    else:
        p_group = np.nan

    # Compute statistics for group comparison if available
    if not np.isnan(p_group):
        # Compute differences on Fisher-z scale
        diffs = np.array(z_dom) - np.array(z_non)
        n_group = len(diffs)
        df_group = n_group - 1
        median_diff = np.median(diffs)
        IQR_diff = np.percentile(diffs, 75) - np.percentile(diffs, 25)
        expected_group = n_group * (n_group + 1) / 4
        sd_group = math.sqrt(n_group * (n_group + 1) * (2 * n_group + 1) / 24)
        z_stat = (stat_group - expected_group) / sd_group if sd_group > 0 else float('nan')
        effect_group = abs(z_stat) / math.sqrt(n_group)
        # Corrected p-value will be annotated later; here we print full stats:
        print("Between-hand comparison (paired Wilcoxon on Fisher z):")
        print(f"  Median difference = {median_diff:.2f}, n = {n_group}, df = {df_group}, IQR = {IQR_diff:.2f}")
        print(f"  z = {z_stat:.2f}, p = {p_group:.4f}, effect size = {effect_group:.2f}")
    else:
        print("Not enough data for between-hand comparison.")

    # ---------------------
    # Wilcoxon vs 0 for each hand (uncorrected p-values) and print stats for each hand
    # ---------------------
    hand_p_values = {}
    for hand_name, vals in zip(hand_order, [median_corr_non, median_corr_dom]):
        if len(vals) > 0:
            try:
                stat_hand, p_val = wilcoxon(vals)
            except Exception:
                p_val = np.nan
                stat_hand = np.nan
            n_hand = len(vals)
            df_hand = n_hand - 1
            median_val = np.median(vals)
            IQR_val = np.percentile(vals, 75) - np.percentile(vals, 25)
            expected_hand = n_hand * (n_hand + 1) / 4
            sd_hand = math.sqrt(n_hand * (n_hand + 1) * (2 * n_hand + 1) / 24)
            z_hand = (stat_hand - expected_hand) / sd_hand if sd_hand > 0 else float('nan')
            effect_hand = abs(z_hand) / math.sqrt(n_hand)
            hand_p_values[hand_name] = p_val
            print(f"Wilcoxon test for {hand_name} vs 0:")
            print(f"  Median = {median_val:.2f}, n = {n_hand}, df = {df_hand}, IQR = {IQR_val:.2f}")
            print(f"  z = {z_hand:.2f}, p = {p_val:.4f}, effect size = {effect_hand:.2f}")
        else:
            hand_p_values[hand_name] = np.nan

    # ---------------------
    # Apply Benjamini–Hochberg FDR correction across the group and hand tests
    # ---------------------
    p_list = []
    test_labels = []
    if not np.isnan(p_group):
        p_list.append(p_group)
        test_labels.append("Group")
    for hand_name in hand_order:
        if hand_name in hand_p_values and not np.isnan(hand_p_values[hand_name]):
            p_list.append(hand_p_values[hand_name])
            test_labels.append(hand_name)

    if p_list:
        _, p_corrected, _, _ = multipletests(p_list, alpha=0.05, method="fdr_bh")
        # Assign corrected p-values
        p_group_corr = p_corrected[0] if test_labels[0] == "Group" else np.nan
        hand_corr = {}
        # If group test was included, hand tests start at index 1; otherwise from 0.
        start_idx = 1 if test_labels[0] == "Group" else 0
        for i, hand_name in enumerate(hand_order):
            hand_corr[hand_name] = p_corrected[start_idx + i]
    else:
        p_group_corr = np.nan
        hand_corr = {h: np.nan for h in hand_order}

    # Annotate between-hand group comparison if significant
    if not np.isnan(p_group_corr):
        group_stars = "ns"
        for threshold, symbol in box_cfg.get("sig_levels", [(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")]):
            if p_group_corr < threshold:
                group_stars = symbol
                break
        if group_stars != "ns":
            med_nd = np.median(median_corr_non)
            med_dom = np.median(median_corr_dom)
            line_y = max(med_nd, med_dom) + group_sig_offset
            if box_cfg.get("sig_line", True):
                ax.plot([0, 1], [line_y, line_y], color=box_cfg.get("sig_line_color", "black"),
                        linewidth=box_cfg.get("sig_line_width", 1.5))
            ax.text(0.5, line_y + sig_text_offset + 0.04, group_stars,
                    ha="center", va="bottom", fontsize=sig_marker_size)
            print(f"Between-hand comparison (paired Wilcoxon on Fisher z): corrected p = {p_group_corr:.4f}, stars = {group_stars}")
        else:
            print(f"Between-hand comparison (paired Wilcoxon on Fisher z): corrected p = {p_group_corr:.4f} (ns)")
    else:
        ax.text(0.5, 0.95, "Not enough data for group comparison", ha="center", va="bottom",
                fontsize=sig_marker_size, transform=ax.get_xaxis_transform())
        print("Not enough data for between-hand comparison.")

    # Annotate each hand vs 0 using the corrected p-values
    common_line_y = max(np.median(median_corr_non + [0]), np.median(median_corr_dom + [0])) + hand_sig_offset
    for i, hand_name in enumerate(hand_order):
        p_val_corr = hand_corr.get(hand_name, np.nan)
        if not np.isnan(p_val_corr):
            stars = "ns"
            for threshold, symbol in box_cfg.get("sig_levels", [(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")]):
                if p_val_corr < threshold:
                    stars = symbol
                    break
            ax.text(i, common_line_y + sig_text_offset, stars,
                    ha="center", va="bottom", fontsize=sig_marker_size)
            print(f"Corrected Wilcoxon test for {hand_name} vs 0: corrected p = {p_val_corr:.4f}, stars = {stars}")
        else:
            ax.text(i, 0.95, f"{hand_name}: NA", ha="center", va="bottom",
                    fontsize=sig_marker_size, transform=ax.get_xaxis_transform())
    # ---------------------
    # Final formatting
    # ---------------------
    for spine in ["right", "top"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(top=False)
    ax.axhline(1, color='white', linestyle='-', linewidth=2)
    plt.tight_layout()
    plt.show()

    # ---------------------
    # Print final medians for box plot
    # ---------------------
    print(f"Median Non-dominant: {np.median(median_corr_non) if median_corr_non else 'NA'}, "
          f"Median Dominant: {np.median(median_corr_dom) if median_corr_dom else 'NA'}")

boxplot_pearson_corr_trials_mean_median_of_reach_indices(SAT_corr_acorss_results, config=plot_config_summary, orientation="vertical")

# -------------------------------------------------------------------------------------------------------------------
# Global dictionary to store the z-scored data.
updated_metrics_zscore = {}

def process_and_plot_scatter_for_subject_hand(updated_metrics, subject, hand, show_plots=True):
    global updated_metrics_zscore

    durations_dict = updated_metrics[subject][hand]['durations']
    distance_dict = updated_metrics[subject][hand]['distance']

    trial_keys = sorted(durations_dict.keys())
    num_reaches = len(durations_dict[trial_keys[0]])

    durations_matrix = np.array([durations_dict[trial] for trial in trial_keys])
    distance_matrix  = np.array([distance_dict[trial] for trial in trial_keys])

    # --- Original Scatter Plots ---
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
        plt.show()  # remove tight_layout to avoid conflicts

    # --- Z-score computation ---
    z_durations_matrix = (durations_matrix - np.nanmean(durations_matrix, axis=0)) / np.nanstd(durations_matrix, axis=0, ddof=0)
    z_distance_matrix  = (distance_matrix - np.nanmean(distance_matrix, axis=0)) / np.nanstd(distance_matrix, axis=0, ddof=0)

    raw_durations = {}
    raw_distance = {}
    zscore_durations = {}
    zscore_distance  = {}
    zscore_perp_distances = {}

    for i in range(num_reaches):
        raw_durations[i + 1] = durations_matrix[:, i].tolist()
        raw_distance[i + 1] = distance_matrix[:, i].tolist()
        zscore_durations[i + 1] = z_durations_matrix[:, i].tolist()
        zscore_distance[i + 1] = z_distance_matrix[:, i].tolist()

    # --- Z-scored Scatter Plots with color-coded MotorAcuity ---
    if show_plots:
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
        for i in range(num_reaches):
            ax = axs[i // cols][i % cols]
            ax.scatter(z_durations_matrix[:, i], z_distance_matrix[:, i], color='purple', alpha=0.5)
            ax.set_title(f"Reach {i + 1}")
            ax.set_xlabel("Z-scored Duration")
            ax.set_ylabel("Z-scored Distance")
            ax.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
            ax.axvline(x=0, color='red', linestyle='--', linewidth=0.8)

            # 45° line
            xlims = ax.get_xlim()
            ylims = ax.get_ylim()
            min_lim = min(xlims[0], ylims[0])
            max_lim = max(xlims[1], ylims[1])
            x_vals = np.linspace(min_lim, max_lim, 100)
            ax.plot(x_vals, x_vals, color='green', linestyle='--', label='45° line')

            perp_distances = []
            motor_acuity_scaled = []
            for j, (x_val, y_val) in enumerate(zip(z_durations_matrix[:, i], z_distance_matrix[:, i])):
                proj_x = (x_val + y_val) / 2
                proj_y = proj_x
                Projection_point = (x_val + y_val) / math.sqrt(2)
                perp_distances.append(Projection_point)

                # invert so higher = better
                motor_acuity = -Projection_point
                motor_acuity_scaled.append(motor_acuity)

                # Color based on MotorAcuity
                norm = plt.Normalize(min(motor_acuity_scaled), max(motor_acuity_scaled))
                cmap = plt.get_cmap("RdYlGn")
                color = cmap(norm(motor_acuity))

                ax.plot([x_val, proj_x], [y_val, proj_y], color=color, linestyle=':', linewidth=0.8)
                ax.scatter(proj_x, proj_y, color=color, marker='x', s=30)

                # Print x, y, MotorAcuity
                # print(f"Reach {i+1}, Point {j+1}: x={x_val:.3f}, y={y_val:.3f}, MotorAcuity={motor_acuity:.3f}")

            # Add colorbar (one per reach)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(motor_acuity_scaled), max(motor_acuity_scaled)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("MotorAcuity (higher = better)")

            zscore_perp_distances[i + 1] = perp_distances
            ax.legend()
            ax.grid(True)
        # fig.suptitle(f"{subject} - {hand.capitalize()}: Z-scored Duration vs Distance", fontsize=16)
        plt.show()

    else:
        for i in range(num_reaches):
            perp_distances = []
            for j, (x_val, y_val) in enumerate(zip(z_durations_matrix[:, i], z_distance_matrix[:, i])):
                Projection_point = (x_val + y_val) / math.sqrt(2)
                perp_distances.append(Projection_point)
            zscore_perp_distances[i + 1] = perp_distances

    # Save data globally
    if subject not in updated_metrics_zscore:
        updated_metrics_zscore[subject] = {}
    updated_metrics_zscore[subject][hand] = {
        'durations': raw_durations,
        'distance': raw_distance,
        'zscore_durations': zscore_durations,
        'zscore_distance': zscore_distance,
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
updated_metrics_zscore = get_updated_metrics_zscore(updated_metrics_acorss_phases, show_plots=True)


def compare_raw_vs_processed(subject, hand, reach_index, show_plots=True, config=plot_config_summary):
    """
    Side-by-side comparison of raw vs processed scatter plots with color-coded MotorAcuity.
    Prints each point's x, y, and MotorAcuity value. Adds a colorbar showing scaled MotorAcuity.
    """
    global updated_metrics_zscore

    # Extract saved data
    raw_durations    = updated_metrics_zscore[subject][hand]['durations'][reach_index]
    raw_distance     = updated_metrics_zscore[subject][hand]['distance'][reach_index]
    zscore_durations = updated_metrics_zscore[subject][hand]['zscore_durations'][reach_index]
    zscore_distance  = updated_metrics_zscore[subject][hand]['zscore_distance'][reach_index]
    perp_distances   = updated_metrics_zscore[subject][hand]['MotorAcuity'][reach_index]

    # Scale and invert MotorAcuity so higher = better
    motor_acuity_scaled = [-d for d in perp_distances]

    # Print each point's x, y, MotorAcuity
    print(f"\nSubject: {subject}, Hand: {hand}, Reach: {reach_index}")
    print("Index\tX\tY\tMotorAcuity")
    for idx, (x, y, m) in enumerate(zip(zscore_durations, zscore_distance, motor_acuity_scaled), start=1):
        print(f"{idx}\t{x:.3f}\t{y:.3f}\t{m:.3f}")

    # Normalize for color mapping
    norm = plt.Normalize(min(motor_acuity_scaled), max(motor_acuity_scaled))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("RedGreen", ["red", "green"])

    # Configuration
    gen_cfg         = config.get("general", {})
    axis_labels     = config.get("axis_labels", {})
    figsize         = (9, 4)
    axis_label_font = gen_cfg.get("axis_label_font", 14)
    tick_label_font = gen_cfg.get("tick_label_font", 14)
    markersize      = gen_cfg.get("marker_size", 50)
    label_offset    = gen_cfg.get("label_offset", 0.09)
    axis_colors     = config.get("axis_colors", {})

    n = len(raw_durations)

    if show_plots:
        fig, axs = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

        # --- Left: Raw Data ---
        axs[0].scatter(raw_durations, raw_distance, color='black', s=markersize, alpha=0.4)
        axs[0].set_xlabel(axis_labels.get("duration", "Duration (s)"), fontsize=axis_label_font)
        axs[0].set_ylabel(axis_labels.get("distance", "Distance (mm)"), fontsize=axis_label_font)
        axs[0].tick_params(axis='both', labelsize=tick_label_font)
        axs[0].grid(False)
        axs[0].set_xlim(0.5, 1.2)
        axs[0].set_ylim(0, 8)
        axs[0].set_yticks([0, 4, 8])
        axs[0].set_yticklabels([0, 4, 8], fontsize=tick_label_font)
        axs[0].set_box_aspect(1)
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)

        # Axis annotations for raw plot
        raw_x_cfg = axis_colors.get("x", {}).get(axis_labels.get("duration", "Duration (s)"), None)
        raw_y_cfg = axis_colors.get("y", {}).get(axis_labels.get("distance", "Distance (mm)"), None)
        if raw_x_cfg:
            axs[0].annotate(raw_x_cfg["start"], xy=(0 - 0.04, -label_offset - 0.02), xycoords='axes fraction',
                            fontsize=tick_label_font, color=raw_x_cfg["colors"][0], ha='left', va='top')
            axs[0].annotate(raw_x_cfg["end"], xy=(1 + 0.04, -label_offset - 0.02), xycoords='axes fraction',
                            fontsize=tick_label_font, color=raw_x_cfg["colors"][1], ha='right', va='top')
        if raw_y_cfg:
            axs[0].annotate(raw_y_cfg["end"], xy=(-label_offset, 1), xycoords='axes fraction',
                            fontsize=tick_label_font, color=raw_y_cfg["colors"][1], ha='right', va='top')
            axs[0].annotate(raw_y_cfg["start"], xy=(-label_offset, 0), xycoords='axes fraction',
                            fontsize=tick_label_font, color=raw_y_cfg["colors"][0], ha='right', va='bottom')

        # --- Right: Processed Data ---
        axs[1].scatter(zscore_durations, zscore_distance, color='grey', label="Z-scored Data",
                       s=markersize, alpha=0.4)
        x_min_proc = min(zscore_durations)
        x_max_proc = max(zscore_durations)
        min_lim = min(x_min_proc, min(zscore_distance))
        max_lim = max(x_max_proc, max(zscore_distance))
        x_vals = np.linspace(min_lim, max_lim, 100)
        axs[1].plot(x_vals, x_vals, color='lightgrey', linestyle='--', label='45° line', linewidth=2, zorder=1)

        # Overlay scaled and color-coded MotorAcuity projections
        for x_val, y_val, proj in zip(zscore_durations, zscore_distance, motor_acuity_scaled):
            proj_x = (x_val + y_val) / 2
            proj_y = proj_x
            color = cmap(norm(proj))
            axs[1].plot([x_val, proj_x], [y_val, proj_y], color='lightgrey', linestyle=':', linewidth=1.5, zorder=1)
            axs[1].scatter(proj_x, proj_y, color=color, marker='x', s=markersize, zorder=2)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axs[1], fraction=0.046, pad=0.04)
        cbar.set_label("Motor acuity", fontsize=tick_label_font)
        cbar.ax.tick_params(labelsize=tick_label_font)
        # Add annotations: top for "Fast and accurate", Bottom for "Slow and inaccurate"
        cbar.ax.text(1.9, 0.78, "Fast\nand\naccurate", 
                 fontsize=tick_label_font, transform=cbar.ax.transAxes, color='green', multialignment='center')
        cbar.ax.text(1.8, -0.1, "Slow\nand\ninaccurate", va="bottom",
                 fontsize=tick_label_font, transform=cbar.ax.transAxes, color='red', multialignment='center')

        axs[1].set_xlabel("Z-scored duration (s)", fontsize=axis_label_font)
        axs[1].set_ylabel("Z-scored distance (mm)", fontsize=axis_label_font)
        axs[1].tick_params(axis='both', labelsize=tick_label_font)
        axs[1].grid(False)
        axs[1].set_aspect('equal', adjustable='box')
        axs[1].set_xticks([-2.5, 0, 2.5])
        axs[1].set_xticklabels(["-2.5", "0", "2.5"])
        axs[1].set_yticks([-2.5, 0, 2.5])
        axs[1].set_yticklabels(["-2.5", "0", "2.5"])
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["right"].set_visible(False)
        axs[1].text(1.08, 1.01, f"n = {n} placements", transform=axs[1].transAxes,
                    ha='right', va='top', fontsize=tick_label_font)

        # Axis annotations for processed plot
        proc_x_cfg = axis_colors.get("x", {}).get(axis_labels.get("duration", "Duration (s)"), None)
        proc_y_cfg = axis_colors.get("y", {}).get(axis_labels.get("distance", "Distance (mm)"), None)
        if proc_x_cfg:
            axs[1].annotate(proc_x_cfg["start"], xy=(0 - 0.08, -label_offset - 0.02), xycoords='axes fraction',
                            fontsize=tick_label_font, color=proc_x_cfg["colors"][0], ha='left', va='top')
            axs[1].annotate(proc_x_cfg["end"], xy=(1 + 0.08, -label_offset - 0.02), xycoords='axes fraction',
                            fontsize=tick_label_font, color=proc_x_cfg["colors"][1], ha='right', va='top')
        if proc_y_cfg:
            axs[1].annotate(proc_y_cfg["end"], xy=(-label_offset - 0.1, 1), xycoords='axes fraction',
                            fontsize=tick_label_font, color=proc_y_cfg["colors"][1], ha='right', va='top')
            axs[1].annotate(proc_y_cfg["start"], xy=(-label_offset - 0.1, 0), xycoords='axes fraction',
                            fontsize=tick_label_font, color=proc_y_cfg["colors"][0], ha='right', va='bottom')

        # Spearman correlation
        spearman_corr, p_value = spearmanr(zscore_durations, zscore_distance)
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
compare_raw_vs_processed('07/22/HW', 'non_dominant', reach_index=14, show_plots=True, config=plot_config_summary)


# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
def convert_updated_metrics_zscore(original_zscore, cross_metrics):
    """
    Converts the structure of updated_metrics_zscore from being keyed by reach indices (1 to 16)
    into being keyed by trial names. The trial names are obtained from the "durations"
    metric from cross_metrics.

    For each subject and hand, for each metric (e.g. 'durations', 'distance',
    'MotorAcuity'), the original structure is assumed to be:
       {reach_index (as int): [trial0_value, trial1_value, ...]}
    and the new structure will be:
       {trial_name: [value_for_reach1, value_for_reach2, ..., value_for_reach16]}

    Parameters:
        original_zscore (dict): The original updated_metrics_zscore dictionary with reach keys.
        cross_metrics (dict): A dictionary (e.g., updated_metrics_acorss_TWs) from which to extract
                              trial names via cross_metrics[subject][hand]['durations'].keys()

    Returns:
        new_zscore (dict): The converted dictionary keyed by trial names.
    """
    new_zscore = {}
    for subject, subj_data in original_zscore.items():
        new_zscore[subject] = {}
        for hand, hand_data in subj_data.items():
            new_zscore[subject][hand] = {}
            # Obtain trial names from cross_metrics by using the keys 
            # from the 'durations' metric
            trial_names = list(cross_metrics[subject][hand]['durations'].keys())
            # For each metric (e.g., 'durations', 'distance', 'MotorAcuity')
            for metric, metric_data in hand_data.items():
                new_zscore[subject][hand][metric] = {}
                # Determine the number of trials from the first reach entry.
                first_reach = sorted(metric_data.keys(), key=lambda k: int(k))[0]
                num_trials = len(metric_data[first_reach])
                # If trial_names length mismatches, generate default names.
                if len(trial_names) != num_trials:
                    trial_names = [f"trial{i+1}" for i in range(num_trials)]
                # For each trial (by index), gather values across all reach indices (sorted).
                for t in range(num_trials):
                    reach_values = []
                    # Sort the reach indices as integers for correct ordering.
                    for reach in sorted(metric_data.keys(), key=lambda k: int(k)):
                        reach_values.append(metric_data[reach][t])
                    # Map the t-th trial name to its list of 16 reach values.
                    new_zscore[subject][hand][metric][trial_names[t]] = reach_values
    return new_zscore

# Convert the updated_metrics_zscore to be keyed by trial names.
updated_metrics_zscore_by_trial = convert_updated_metrics_zscore(updated_metrics_zscore,
                                                                  updated_metrics_acorss_phases)


def save_overall_median_motor_acuity(updated_metrics_zscore_by_trial, output_file=None):
    """
    Computes and optionally saves the overall median MotorAcuity per subject and hand.
    
    Parameters:
        updated_metrics_zscore_by_trial (dict): The z-scored metrics dictionary keyed by trial.
        output_file (str, optional): If provided, the path of the file to save the results using pickle.
    
    Returns:
        dict: A dictionary with the overall median MotorAcuity per subject and hand.
              Structure: { subject: { hand: overall_median, ... }, ... }
    """

    results = {}
    
    # Loop through all subjects and hands to compute the overall median MotorAcuity per subject and hand
    for subject, subject_data in updated_metrics_zscore_by_trial.items():
        results[subject] = {}
        for hand, hand_data in subject_data.items():
            motor_acuity_data = hand_data["MotorAcuity"]  # Get MotorAcuity metrics for this subject and hand
            
            # Calculate the median MotorAcuity per trial (ignoring empty lists)
            trial_medians = [
                np.nanmedian(values)
                for values in motor_acuity_data.values()
                if len(values) > 0
            ]
            
            # Calculate the overall median across all trials; if no valid trials, set to NaN
            overall_median = np.nan if len(trial_medians) == 0 else np.nanmedian(trial_medians)
            
            results[subject][hand] = overall_median
    
    return results

# Example call to save the overall median MotorAcuity per subject and hand
overall_median_motor_acuity = save_overall_median_motor_acuity(updated_metrics_zscore_by_trial, output_file="overall_median_motor_acuity.pkl")

def calculate_median_across_trials(metrics_data):
    """
    Compute the median value for each subject, each hand, for each metric across all trials.
    First, compute the median for each trial, then compute the median of these trial medians.
    
    The input dictionary should be structured as:
      { subject: { hand: { metric: { trial: [values,...], ... }, ... }, ... }, ... }
    
    Returns:
        medians: dict, with structure
            { subject: { hand: { metric: median_value, ... }, ... }, ... }
    """
    medians = {}
    for subject, subject_data in metrics_data.items():
        medians[subject] = {}
        for hand, hand_data in subject_data.items():
            medians[subject][hand] = {}
            for metric, trial_dict in hand_data.items():
                trial_medians = []
                for trial, values in trial_dict.items():
                    # Filter out values that are None or NaN
                    filtered_values = [v for v in values if v is not None and not np.isnan(v)]
                    if filtered_values:
                        trial_medians.append(np.median(filtered_values))
                if trial_medians:
                    medians[subject][hand][metric] = np.median(trial_medians)
                else:
                    medians[subject][hand][metric] = np.nan
    return medians

subject_medians = calculate_median_across_trials(updated_metrics_acorss_phases)

def plot_median_metrics(subject_medians, overall_median_motor_acuity, hand_order=["non_dominant", "dominant"], figsize=(12, 4)):
    """
    Plots boxplots of median Duration and Distance per hand using the subject_medians dictionary,
    and adds a third subplot for the subjects' median MotorAcuity using overall_median_motor_acuity.
    Performs a paired test between hands using only the Wilcoxon signed‐rank test.
    """
    import matplotlib.pyplot as plt

    # Font scaling from configuration.
    axis_label_font = plot_config_summary["general"]["axis_label_font"] * 3
    tick_label_font = plot_config_summary["general"]["tick_label_font"] * 2.5

    # Convert subject_medians into a DataFrame for Duration and Distance.
    data = []
    for subject, hands in subject_medians.items():
        for hand, metrics in hands.items():
            data.append({
                "Subject": subject,
                "Hand": hand,
                "Duration": metrics.get("durations", float("nan")),
                "Distance": metrics.get("distance", float("nan"))
            })
    df = pd.DataFrame(data)

    # Map hand keys to labels.
    hand_labels = {"dominant": "Dominant", "non_dominant": "Non-dominant"}
    df["Hand"] = df["Hand"].map(hand_labels)
    hand_order_labels = [hand_labels[h] for h in hand_order]

    # Create Motor Acuity DataFrame.
    data_z = []
    for subject, hands in overall_median_motor_acuity.items():
        for hand, overall_median in hands.items():
            data_z.append({
                "Subject": subject,
                "Hand": hand,
                "MotorAcuity": overall_median
            })
    df_z = pd.DataFrame(data_z)
    df_z["Hand"] = df_z["Hand"].map(hand_labels)

    # Create figure with three subplots.
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Each subplot configuration: (dataframe, y column, y label, axis, y limits, y ticks)
    plots = [
        (df, "Duration", "Duration (s)", axes[0], (0, 1.2), [0, 0.6, 1.2]),
        (df, "Distance", "Error (mm)", axes[1], (0, 5), [0, 2.5, 5]),
        (df_z, "MotorAcuity", "Motor acuity", axes[2], (-0.3, 0.1), [-0.3, -0.1, 0.1])
    ]

    # Define significance levels.
    sig_levels = [(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")]

    for df_i, y_col, y_label, ax, ylim, yticks in plots:
        # Draw boxplot and swarmplot.
        sns.boxplot(
            data=df_i, x="Hand", y=y_col, order=hand_order_labels, ax=ax,
            palette=["#A9A9A9", "#F0F0F0"], width=0.8
        )
        sns.swarmplot(
            data=df_i, x="Hand", y=y_col, order=hand_order_labels,
            ax=ax, color='black', size=16, alpha=0.4
        )

        # Formatting.
        ax.set_xlabel("Hand", fontsize=axis_label_font)
        ax.set_ylabel(y_label, fontsize=axis_label_font)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.tick_params(labelsize=tick_label_font)
        ax.yaxis.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate sample size in the top-right corner (only for Motor Acuity plot).
        n_participants = df_i["Subject"].nunique()
        if ax == axes[2]:
            ax.text(0.9, 1, f"n = {n_participants} participants", transform=ax.transAxes,
                    ha="right", va="top", fontsize=tick_label_font)

        # ---- Collect p-value for the paired hand-to-hand test ----
        pvals = {}
        test_method = "Wilcoxon"
        try:
            df_wide = df_i.pivot(index="Subject", columns="Hand", values=y_col).dropna()
            if all(h in df_wide.columns for h in hand_order_labels):
                # Use Wilcoxon signed-rank test only.
                stat, p_val = wilcoxon(df_wide[hand_order_labels[0]], df_wide[hand_order_labels[1]])
                pvals["paired"] = p_val
                # Compute z-value and effect size r if sample size is sufficient
                n = len(df_wide)
                expected = n * (n + 1) / 4
                std_dev = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
                z_val = (stat - expected) / std_dev if std_dev > 0 else np.nan
                r_val = abs(z_val) / np.sqrt(n) if n > 0 else np.nan
            else:
                pvals["paired"] = np.nan
                z_val = np.nan
                r_val = np.nan
        except Exception:
            pvals["paired"] = np.nan
            z_val = np.nan
            r_val = np.nan

        # Annotate paired test result without FDR correction.
        stars_paired = "ns"
        if not np.isnan(pvals.get("paired", np.nan)):
            for thr, sym in sig_levels:
                if pvals["paired"] < thr:
                    stars_paired = sym
                    break
        if stars_paired != "ns":
            y_max = df_i[y_col].max()
            ax.plot([0, 1], [y_max + 0.055 * (ylim[1] - ylim[0])]*2, color="black", linewidth=1.5)
            ax.text(0.5, y_max + 0.04 * (ylim[1] - ylim[0]), stars_paired,
                ha="center", va="bottom", fontsize=50)
            print(f"{test_method} {y_label} (paired test): z = {z_val:.2f}, p = {pvals.get('paired', np.nan):.4f}, r = {r_val:.2f}, stars = {stars_paired}")
        else:
            print(f"{test_method} {y_label} (paired test): z = {z_val}, p = {pvals.get('paired', np.nan)}, r = {r_val}")
    plt.tight_layout()
    plt.show()
    
    print(f"median duration non-dominant: {df[df['Hand']=='Non-dominant']['Duration'].median()}, IQR: {df[df['Hand']=='Non-dominant']['Duration'].quantile(0.75) - df[df['Hand']=='Non-dominant']['Duration'].quantile(0.25)}")
    print(f"median duration dominant: {df[df['Hand']=='Dominant']['Duration'].median()}, IQR: {df[df['Hand']=='Dominant']['Duration'].quantile(0.75) - df[df['Hand']=='Dominant']['Duration'].quantile(0.25)}")
    
    print(f"median distance non-dominant: {df[df['Hand']=='Non-dominant']['Distance'].median()}, IQR: {df[df['Hand']=='Non-dominant']['Distance'].quantile(0.75) - df[df['Hand']=='Non-dominant']['Distance'].quantile(0.25)}")
    print(f"median distance dominant: {df[df['Hand']=='Dominant']['Distance'].median()}, IQR: {df[df['Hand']=='Dominant']['Distance'].quantile(0.75) - df[df['Hand']=='Dominant']['Distance'].quantile(0.25)}")
    
    print(f"median motor acuity non-dominant: {df_z[df_z['Hand']=='Non-dominant']['MotorAcuity'].median()}, IQR: {df_z[df_z['Hand']=='Non-dominant']['MotorAcuity'].quantile(0.75) - df_z[df_z['Hand']=='Non-dominant']['MotorAcuity'].quantile(0.25)}")
    print(f"median motor acuity dominant: {df_z[df_z['Hand']=='Dominant']['MotorAcuity'].median()}, IQR: {df_z[df_z['Hand']=='Dominant']['MotorAcuity'].quantile(0.75) - df_z[df_z['Hand']=='Dominant']['MotorAcuity'].quantile(0.25)}")
# Example usage:
plot_median_metrics(subject_medians, overall_median_motor_acuity, hand_order=["non_dominant", "dominant"], figsize=(23, 10))


def plot_phase_metrics_custom(subject_medians, metrics_groups, phases=None, figsize=(10,5)):
    """
    Plots multiple metrics in subplots with specified colors, y-axis ticks, legend placement,
    hatching for non-dominant bars, and error bars (std dev across subjects).

    Performs Wilcoxon paired test between dominant and non-dominant hand values
    for each phase (annotated between the two bars). FDR Benjamini-Hochberg correction is applied 
    to the multiple comparisons.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    if phases is None:
        phases = ["TW", "ballistic", "correction"]

    n_subplots = len(metrics_groups)
    fig, axes = plt.subplots(1, n_subplots, figsize=figsize)
    if n_subplots == 1:
        axes = [axes]

    # Define colors per phase
    phase_colors = {"TW": "white", "ballistic": "#0047ff", "correction": "#ffb800"}

    # Define custom y-ticks per metric
    yticks_dict = {"ldlj": [0, -5, -10], "sparc": [0, -1, -2]}

    # Helper: star fontsize
    def star_fontsize(stars):
        return 14 if stars == "ns" else 18

    # Dictionary to collect paired p-values for ballistic vs correction per metric and hand
    bc_pvals = {}  # structure: {metric_prefix: {"dominant": p_val, "non_dominant": p_val}}

    for ax, (metric_prefix, title) in zip(axes, metrics_groups):
        if metric_prefix not in bc_pvals:
            bc_pvals[metric_prefix] = {}
        # Metric keys: for distance/durations, use the same key; otherwise add phase prefix.
        if metric_prefix.lower() in ["distance", "durations"]:
            metric_keys = {phase: metric_prefix for phase in phases}
        else:
            metric_keys = {phase: f"{phase}_{metric_prefix}" for phase in phases}

        # Aggregate values per phase and hand
        aggregated = {phase: {"dominant": [], "non_dominant": []} for phase in phases}
        for subj, hands in subject_medians.items():
            for hand, metrics in hands.items():
                for phase in phases:
                    key = metric_keys.get(phase)
                    if key in metrics:
                        aggregated[phase][hand].append(metrics[key])

        # Compute averages and standard deviations per phase/hand
        averages = {phase: {} for phase in phases}
        std_devs = {phase: {} for phase in phases}
        for phase in phases:
            for hand in ["dominant", "non_dominant"]:
                vals = aggregated[phase][hand]
                averages[phase][hand] = np.nanmean(vals) if vals else np.nan
                std_devs[phase][hand] = np.nanstd(vals) if vals else 0

        # Collect p-values for paired dominant vs non-dominant tests
        paired_pvals = []
        for phase in phases:
            dom_vals = aggregated[phase]["dominant"]
            nondom_vals = aggregated[phase]["non_dominant"]

            # Paired dominant vs non-dominant
            if len(dom_vals) > 0 and len(nondom_vals) > 0 and len(dom_vals) == len(nondom_vals):
                try:
                    _, p_val = wilcoxon(dom_vals, nondom_vals)
                except Exception:
                    p_val = np.nan
            else:
                p_val = np.nan
            paired_pvals.append(p_val)

        # Apply FDR Benjamini-Hochberg correction for the paired tests
        _, paired_pvals_corr, _, _ = multipletests(paired_pvals, alpha=0.05, method="fdr_bh")

        # Bar plot settings
        labels = phases
        x = np.arange(len(labels))
        width = 0.35

        # Plot bars and annotate within-phase stats
        for i, phase in enumerate(phases):
            dom_val = averages[phase]["dominant"]
            dom_err = std_devs[phase]["dominant"]
            nondom_val = averages[phase]["non_dominant"]
            nondom_err = std_devs[phase]["non_dominant"]

            # Dominant bar (left) with hatch
            ax.bar(x[i] - width/2, dom_val, width,
                   color=phase_colors[phase], edgecolor='black', hatch='//',
                   yerr=dom_err, capsize=5, label='Dominant' if i == 0 else "")
            # Non-dominant bar (right)
            ax.bar(x[i] + width/2, nondom_val, width,
                   color=phase_colors[phase], edgecolor='black',
                   yerr=nondom_err, capsize=5, label='Non-dominant' if i == 0 else "")

            # Annotate paired test between hands for this phase using corrected p-value
            p_val_pair = paired_pvals_corr[i]
            if np.isnan(p_val_pair):
                stars = "NA"
            elif p_val_pair < 0.001:
                stars = "***"
            elif p_val_pair < 0.01:
                stars = "**"
            elif p_val_pair < 0.05:
                stars = "*"
            else:
                stars = "ns"

            bottom_dom = dom_val - dom_err
            bottom_nondom = nondom_val - nondom_err
            y_annot = min(bottom_dom, bottom_nondom) - 0.1 * abs(min(bottom_dom, bottom_nondom))
            x_left = x[i] - width/2
            x_right = x[i] + width/2
            if stars not in ['ns', 'NA']:
                if y_annot < -2:
                    y_annot = y_annot - 0.2  
                    ax.plot([x_left, x_right], [y_annot, y_annot], color='black', linewidth=1.5)
                    ax.text((x_left + x_right) / 2, y_annot - 0.03, stars,
                            ha='center', va='top', fontsize=star_fontsize(stars))
                else:
                    y_annot = y_annot - 0.03
                    ax.plot([x_left, x_right], [y_annot, y_annot], color='black', linewidth=1.5)
                    ax.text((x_left + x_right) / 2, y_annot - 0.01, stars,
                            ha='center', va='top', fontsize=star_fontsize(stars))

        # Additional paired comparison: ballistic vs correction for each hand
        # Assumes phases order: index 1 = ballistic, index 2 = correction
        for hand in ["dominant", "non_dominant"]:
            ball_vals = aggregated["ballistic"][hand]
            corr_vals = aggregated["correction"][hand]
            if ball_vals and corr_vals and len(ball_vals) == len(corr_vals):
                try:
                    _, p_val = wilcoxon(ball_vals, corr_vals)
                except Exception:
                    p_val = np.nan
            else:
                p_val = np.nan
            # Store the p-value in our dictionary
            bc_pvals[metric_prefix][hand] = p_val

            if hand == "dominant":
                x_left = x[1] - width/2
                x_right = x[2] - width/2
                val_ball = averages["ballistic"]["dominant"]
                err_ball = std_devs["ballistic"]["dominant"]
                val_corr = averages["correction"]["dominant"]
                err_corr = std_devs["correction"]["dominant"]
            else:
                x_left = x[1] + width/2
                x_right = x[2] + width/2
                val_ball = averages["ballistic"]["non_dominant"]
                err_ball = std_devs["ballistic"]["non_dominant"]
                val_corr = averages["correction"]["non_dominant"]
                err_corr = std_devs["correction"]["non_dominant"]

            base_y_annot = min(val_ball - err_ball, val_corr - err_corr) - 0.1 * abs(min(val_ball - err_ball, val_corr - err_corr))
            if not np.isnan(p_val):
                if p_val < 0.001:
                    stars_bc = "***"
                elif p_val < 0.01:
                    stars_bc = "**"
                elif p_val < 0.05:
                    stars_bc = "*"
                else:
                    stars_bc = "ns"
            else:
                stars_bc = "NA"

            # Annotate for ballistic vs correction only if significant
            if stars_bc not in ['ns', 'NA']:
                if base_y_annot < -2:
                    base_y_annot = base_y_annot - 1  
                    if hand == "dominant":
                        base_y_annot = base_y_annot - 0.05
                        ax.plot([x_left, x_right], [base_y_annot, base_y_annot], color='black', linewidth=1.5)
                        ax.text((x_left + x_right) / 2, base_y_annot - 0.03, stars_bc,
                                ha='center', va='top', fontsize=star_fontsize(stars_bc), color='black')
                    if hand == "non_dominant":
                        base_y_annot = base_y_annot - 0.25
                        ax.plot([x_left, x_right], [base_y_annot, base_y_annot], color='black', linewidth=1.5)
                        ax.text((x_left + x_right) / 2, base_y_annot - 0.01, stars_bc,
                                ha='center', va='top', fontsize=star_fontsize(stars_bc), color='black')
                else:
                    base_y_annot = base_y_annot - 0.07
                    if hand == "dominant":
                        base_y_annot = base_y_annot - 0.05
                        ax.plot([x_left, x_right], [base_y_annot, base_y_annot], color='black', linewidth=1.5)
                        ax.text((x_left + x_right) / 2, base_y_annot - 0.01, stars_bc,
                                ha='center', va='top', fontsize=star_fontsize(stars_bc), color='black')
                    if hand == "non_dominant":
                        base_y_annot = base_y_annot - 0.15
                        ax.plot([x_left, x_right], [base_y_annot, base_y_annot], color='black', linewidth=1.5)
                        ax.text((x_left + x_right) / 2, base_y_annot - 0.01, stars_bc,
                                ha='center', va='top', fontsize=star_fontsize(stars_bc), color='black')
        ax.set_xticks(x)
        labels = ['TW', 'Ballistic', 'Correction']
        new_labels = [("Ballistic +\n Correction" if label == "TW" else label.capitalize()) for label in labels]
        ax.set_xticklabels(new_labels, fontsize=14)
        if metric_prefix.lower() in ["distance", "durations"]:
            ax.set_ylabel(metric_prefix.capitalize(), fontsize=14)
        else:
            ax.set_ylabel(metric_prefix.upper(), fontsize=14)
        
        if metric_prefix.lower() in yticks_dict:
            ax.set_yticks(yticks_dict[metric_prefix.lower()])

        if metric_prefix.lower() == "ldlj":
            ax.legend(loc='lower right', frameon=False, fontsize=13, bbox_to_anchor=(1.2, -0.05))
        
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()
    
    # Print out the collected p-values for paired ballistic vs correction comparisons
    print("Paired ballistic vs correction p-values (per metric, per hand):")
    for metric, hand_dict in bc_pvals.items():
        for hand, p_val in hand_dict.items():
            print(f"Metric: {metric}, Hand: {hand}, p-value: {p_val}")

plot_phase_metrics_custom(subject_medians, metrics_groups=[("LDLJ", "LDLJ"), ("sparc", "SPARC")], figsize=(8, 4))




def plot_phase_metrics_custom(subject_medians, metrics_groups, phases=None, figsize=(10,5)):
    """
    Plots multiple metrics in subplots with specified colors, y-axis ticks, legend placement,
    hatching for non-dominant bars, and error bars (std dev across subjects).

    Performs paired t-tests between dominant and non-dominant hand values
    for each phase and ballistic vs correction within each hand.
    FDR Benjamini-Hochberg correction is applied per metric across all these comparisons.
    
    Additionally, prints out each test result including:
      - The mean and standard deviation for each condition.
      - The t-statistic with its degrees of freedom in parentheses.
      - The exact p-value.
      - The effect size (Cohen's d).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import ttest_rel, shapiro
    from statsmodels.stats.multitest import multipletests

    if phases is None:
        phases = ["TW", "ballistic", "correction"]

    n_subplots = len(metrics_groups)
    fig, axes = plt.subplots(1, n_subplots, figsize=figsize)
    if n_subplots == 1:
        axes = [axes]

    # Define colors per phase
    phase_colors = {"TW": "white", "ballistic": "#0047ff", "correction": "#ffb800"}

    # Define custom y-ticks per metric
    yticks_dict = {"ldlj": [0, -5, -10], "sparc": [0, -1, -2]}

    def star_fontsize(stars):
        return 14 if stars == "ns" else 18

    # Collect p-values per metric
    bc_pvals = {}

    for ax, (metric_prefix, title) in zip(axes, metrics_groups):
        if metric_prefix not in bc_pvals:
            bc_pvals[metric_prefix] = {}

        # Define metric keys for each phase
        if metric_prefix.lower() in ["distance", "durations"]:
            metric_keys = {phase: metric_prefix for phase in phases}
        else:
            metric_keys = {phase: f"{phase}_{metric_prefix}" for phase in phases}

        # Aggregate values per phase and hand
        aggregated = {phase: {"dominant": [], "non_dominant": []} for phase in phases}
        for subj, hands in subject_medians.items():
            for hand, metrics in hands.items():
                for phase in phases:
                    key = metric_keys.get(phase)
                    if key in metrics:
                        aggregated[phase][hand].append(metrics[key])

        # Compute averages & standard deviations per phase and hand
        averages = {phase: {} for phase in phases}
        std_devs = {phase: {} for phase in phases}
        for phase in phases:
            for hand in ["dominant", "non_dominant"]:
                vals = aggregated[phase][hand]
                averages[phase][hand] = np.nanmean(vals) if vals else np.nan
                std_devs[phase][hand] = np.nanstd(vals, ddof=1) if vals else 0

        # --- Collect all p-values for this metric ---
        all_pvals, pval_labels = [], []

        # Dominant vs Non-dominant comparisons per phase
        for phase in phases:
            dom_vals = aggregated[phase]["dominant"]
            nondom_vals = aggregated[phase]["non_dominant"]

            if len(dom_vals) > 0 and len(nondom_vals) > 0 and len(dom_vals) == len(nondom_vals):
                diff = np.array(dom_vals) - np.array(nondom_vals)
                # Normality check on difference
                try:
                    stat_norm, p_norm = shapiro(diff)
                    norm_result = "normal" if p_norm > 0.05 else "not normal"
                    # print(f"Normality check for dominant vs non-dominant in phase '{phase}': W = {stat_norm:.3f}, p = {p_norm:.3f} -> {norm_result}")
                except Exception as e:
                    print(f"Normality check failed for phase '{phase}': {e}")
                try:
                    t_stat, p_val = ttest_rel(dom_vals, nondom_vals)
                except Exception:
                    t_stat, p_val = np.nan, np.nan
                n = len(dom_vals)
                df = n - 1
                # Calculate Cohen's d for paired samples
                d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else np.nan
                # Print the test result
                print(f"Phase '{phase}' comparison (Dominant vs Non-dominant):")
                print(f"  Dominant: mean = {np.mean(dom_vals):.3f}, SD = {np.std(dom_vals, ddof=1):.3f}")
                print(f"  Non-dominant: mean = {np.mean(nondom_vals):.3f}, SD = {np.std(nondom_vals, ddof=1):.3f}")
                print(f"  t({df}) = {t_stat:.3f}, p = {p_val}, Cohen's d = {d:.3f}\n")
            else:
                p_val = np.nan

            all_pvals.append(p_val)
            pval_labels.append(("dom_vs_nondom", phase))

        # Ballistic vs Correction comparisons per hand
        for hand in ["dominant", "non_dominant"]:
            ball_vals = aggregated["ballistic"][hand]
            corr_vals = aggregated["correction"][hand]
            if ball_vals and corr_vals and len(ball_vals) == len(corr_vals):
                diff = np.array(ball_vals) - np.array(corr_vals)
                try:
                    stat_norm, p_norm = shapiro(diff)
                    norm_result = "normal" if p_norm > 0.05 else "not normal"
                    # print(f"Normality check for ballistic vs correction for hand '{hand}': W = {stat_norm:.3f}, p = {p_norm:.3f} -> {norm_result}")
                except Exception as e:
                    print(f"Normality check failed for ballistic vs correction, hand '{hand}': {e}")
                try:
                    t_stat, p_val = ttest_rel(ball_vals, corr_vals)
                except Exception:
                    t_stat, p_val = np.nan, np.nan
                n = len(ball_vals)
                df = n - 1
                d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else np.nan
                print(f"Ballistic vs Correction comparison for hand '{hand}':")
                print(f"  Ballistic: mean = {np.mean(ball_vals):.3f}, SD = {np.std(ball_vals, ddof=1):.3f}")
                print(f"  Correction: mean = {np.mean(corr_vals):.3f}, SD = {np.std(corr_vals, ddof=1):.3f}")
                print(f"  t({df}) = {t_stat:.3f}, p = {p_val}, Cohen's d = {d:.3f}\n")
            else:
                p_val = np.nan

            bc_pvals[metric_prefix][hand] = p_val
            all_pvals.append(p_val)
            pval_labels.append(("ball_vs_corr", hand))

        # Apply FDR correction for the collected p-values
        _, all_pvals_corr, _, _ = multipletests(all_pvals, alpha=0.05, method="fdr_bh")
        corrected_dict = {label: p_corr for label, p_corr in zip(pval_labels, all_pvals_corr)}
        
        # Print the corrected p-values for each test with exact values
        print("Corrected p-values for tests:")
        for label, p_corr in corrected_dict.items():
            test_type, identifier = label
            print(f"  Test: {test_type} for {identifier}, corrected p-value: {p_corr}")

        # --- Plot bars and annotate ---
        labels_plot = phases
        x = np.arange(len(labels_plot))
        width = 0.35

        for i, phase in enumerate(phases):
            dom_val = averages[phase]["dominant"]
            dom_err = std_devs[phase]["dominant"]
            nondom_val = averages[phase]["non_dominant"]
            nondom_err = std_devs[phase]["non_dominant"]

            ax.bar(x[i] - width/2, dom_val, width,
                   color=phase_colors[phase], edgecolor='black', hatch='//',
                   yerr=dom_err, capsize=5, label='Dominant' if i == 0 else "")
            ax.bar(x[i] + width/2, nondom_val, width,
                   color=phase_colors[phase], edgecolor='black',
                   yerr=nondom_err, capsize=5, label='Non-dominant' if i == 0 else "")

            # Annotate corrected dominant vs non-dominant p-value
            p_val_pair = corrected_dict.get(("dom_vs_nondom", phase), np.nan)
            if np.isnan(p_val_pair):
                stars = "NA"
            elif p_val_pair < 0.001:
                stars = "***"
            elif p_val_pair < 0.01:
                stars = "**"
            elif p_val_pair < 0.05:
                stars = "*"
            else:
                stars = "ns"

            bottom_dom = dom_val - dom_err
            bottom_nondom = nondom_val - nondom_err
            y_annot = min(bottom_dom, bottom_nondom) - 0.1 * abs(min(bottom_dom, bottom_nondom))
            x_left = x[i] - width/2
            x_right = x[i] + width/2

            if stars not in ['ns', 'NA']:
                y_annot = y_annot - 0.05
                ax.plot([x_left, x_right], [y_annot, y_annot], color='black', linewidth=1.5)
                ax.text((x_left + x_right)/2, y_annot - 0.02, stars,
                        ha='center', va='top', fontsize=star_fontsize(stars))

        # Annotate ballistic vs correction for each hand
        for hand in ["dominant", "non_dominant"]:
            p_val_corr = corrected_dict.get(("ball_vs_corr", hand), np.nan)
            if np.isnan(p_val_corr):
                stars_bc = "NA"
            elif p_val_corr < 0.001:
                stars_bc = "***"
            elif p_val_corr < 0.01:
                stars_bc = "**"
            elif p_val_corr < 0.05:
                stars_bc = "*"
            else:
                stars_bc = "ns"

            if stars_bc not in ['ns', 'NA']:
                if hand == "dominant":
                    x_left = x[1] - width/2
                    x_right = x[2] - width/2
                else:
                    x_left = x[1] + width/2
                    x_right = x[2] + width/2

                y_base = min(averages["ballistic"][hand] - std_devs["ballistic"][hand],
                             averages["correction"][hand] - std_devs["correction"][hand])
                if y_base < -2:
                    y_annot = y_base - 0.15 if hand == "dominant" else y_base - 0.3
                else:
                    y_annot = y_base - 0.12 if hand == "dominant" else y_base - 0.04
                ax.plot([x_left, x_right], [y_annot, y_annot], color='black', linewidth=1.5)
                ax.text((x_left + x_right)/2, y_annot - 0.02, stars_bc,
                        ha='center', va='top', fontsize=star_fontsize(stars_bc))

        # Axes formatting
        ax.set_xticks(x)
        labels_plot = ['TW', 'Ballistic', 'Correction']
        new_labels = [("Ballistic +\n Correction" if label == "TW" else label.capitalize()) for label in labels_plot]
        ax.set_xticklabels(new_labels, fontsize=14)
        if metric_prefix.lower() in ["distance", "durations"]:
            ax.set_ylabel(metric_prefix.capitalize(), fontsize=14)
        else:
            ax.set_ylabel(metric_prefix.upper(), fontsize=14)

        if metric_prefix.lower() in yticks_dict:
            ax.set_yticks(yticks_dict[metric_prefix.lower()])

        if metric_prefix.lower() == "ldlj":
            ax.legend(loc='lower right', frameon=False, fontsize=13, bbox_to_anchor=(1.2, -0.05))

        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

plot_phase_metrics_custom(subject_medians, metrics_groups=[("LDLJ", "LDLJ"), ("sparc", "SPARC")], figsize=(8, 4))





# -------------------------------------------------------------------------------------------------------------------
def plot_sbbt_vs_median_metrics(subject_medians, All_dates, sBBTResult, overall_median_motor_acuity):
    """
    Plots 6 subplots in a 2x3 grid comparing sBBT scores against:
      1. Median Duration (s)
      2. Median Distance (mm)
      3. Overall Median Motor Acuity
    Top row shows non-dominant hand data and bottom row shows dominant hand data.
    Computes the Pearson correlations for each hand separately and adjusts p-values
    using the FDR Benjamini-Hochberg method.
    
    A linear regression line is overlaid on each scatter plot.
    
    Parameters:
        subject_medians (dict): Dictionary keyed by subject identifiers.
            Each value is a dict with hand keys (e.g., "non_dominant", "dominant")
            that further contains metrics like "durations" and "distance".
        All_dates (list): List of subject identifiers corresponding to the order in sBBTResult.
        sBBTResult (dict): Dictionary with sBBT scores.
            For example, sBBTResult["non_dominant"] and sBBTResult["dominant"] are ordered lists of scores.
        overall_median_motor_acuity (dict): Dictionary with overall median Motor Acuity per subject and hand.
    """
    import matplotlib.pyplot as plt

    # Colors for plotting (Non-dominant, Dominant)
    colors = {"Non-dominant": "#A9A9A9", "Dominant": "#F0F0F0"}

    # Initialize lists for non-dominant values.
    sBBT_scores_nd = []
    median_durations_nd = []
    median_distances_nd = []
    median_MC_nd = []

    # Initialize lists for dominant values.
    sBBT_scores_dom = []
    median_durations_dom = []
    median_distances_dom = []
    median_MC_dom = []

    subjects = list(subject_medians.keys())
    for idx, subject in enumerate(subjects):
        # Use All_dates for matching ordering.
        date_key = All_dates[idx]

        # Non-dominant metrics
        score_nd = sBBTResult["non_dominant"][idx]
        med_duration_nd = subject_medians[date_key]["non_dominant"]["durations"]
        med_distance_nd = subject_medians[date_key]["non_dominant"]["distance"]
        med_MC_nd = overall_median_motor_acuity[date_key]["non_dominant"]

        # Dominant metrics
        score_dom = sBBTResult["dominant"][idx]
        med_duration_dom = subject_medians[date_key]["dominant"]["durations"]
        med_distance_dom = subject_medians[date_key]["dominant"]["distance"]
        med_MC_dom = overall_median_motor_acuity[date_key]["dominant"]

        if (score_nd is not None and med_duration_nd is not None and 
            med_distance_nd is not None and med_MC_nd is not None):
            sBBT_scores_nd.append(score_nd)
            median_durations_nd.append(med_duration_nd)
            median_distances_nd.append(med_distance_nd)
            median_MC_nd.append(med_MC_nd)

        if (score_dom is not None and med_duration_dom is not None and 
            med_distance_dom is not None and med_MC_dom is not None):
            sBBT_scores_dom.append(score_dom)
            median_durations_dom.append(med_duration_dom)
            median_distances_dom.append(med_distance_dom)
            median_MC_dom.append(med_MC_dom)

    # Compute Pearson correlations for non-dominant hand.
    corr_dur_nd, p_dur_nd = pearsonr(sBBT_scores_nd, median_durations_nd)
    corr_dist_nd, p_dist_nd = pearsonr(sBBT_scores_nd, median_distances_nd)
    corr_MC_nd, p_MC_nd = pearsonr(sBBT_scores_nd, median_MC_nd)

    # Compute Pearson correlations for dominant hand.
    corr_dur_dom, p_dur_dom = pearsonr(sBBT_scores_dom, median_durations_dom)
    corr_dist_dom, p_dist_dom = pearsonr(sBBT_scores_dom, median_distances_dom)
    corr_MC_dom, p_MC_dom = pearsonr(sBBT_scores_dom, median_MC_dom)

    # Adjust all p-values using FDR Benjamini-Hochberg correction.
    pvals_best = [p_dur_nd, p_dist_nd, p_MC_nd, p_dur_dom, p_dist_dom, p_MC_dom]
    reject, pvals_corrected, _, _ = multipletests(pvals_best, alpha=0.05, method="fdr_bh")
    p_dur_nd_corr, p_dist_nd_corr, p_MC_nd_corr, p_dur_dom_corr, p_dist_dom_corr, p_MC_dom_corr = pvals_corrected

    # Create a figure with 2 rows and 3 columns of subplots.
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))

    # --- Top row: Non-dominant hand ---
    # Subplot 1: sBBT vs Duration (Non-dominant)
    axs[0, 0].scatter(sBBT_scores_nd, median_durations_nd, s=40, edgecolors="black", 
                      color=colors["Non-dominant"])
    # Linear regression for Duration (Non-dominant)
    x_vals = np.linspace(min(sBBT_scores_nd), max(sBBT_scores_nd), 100)
    slope, intercept = np.polyfit(sBBT_scores_nd, median_durations_nd, 1)
    axs[0, 0].plot(x_vals, slope * x_vals + intercept, color='black', linestyle='--')
    axs[0, 0].set_ylabel("Duration (s)")
    axs[0, 0].set_title(f"ND: r = {corr_dur_nd:.2f}, p = {p_dur_nd_corr:.3f}")
    axs[0, 0].grid(False)

    # Subplot 2: sBBT vs Distance (Non-dominant)
    axs[0, 1].scatter(sBBT_scores_nd, median_distances_nd, s=40, edgecolors="black", 
                      color=colors["Non-dominant"])
    # Linear regression for Distance (Non-dominant)
    slope, intercept = np.polyfit(sBBT_scores_nd, median_distances_nd, 1)
    axs[0, 1].plot(x_vals, slope * x_vals + intercept, color='black', linestyle='--')
    axs[0, 1].set_ylabel("Distance (mm)")
    axs[0, 1].set_title(f"ND: r = {corr_dist_nd:.2f}, p = {p_dist_nd_corr:.3f}")
    axs[0, 1].grid(False)

    # Subplot 3: sBBT vs Motor Acuity (Non-dominant)
    axs[0, 2].scatter(sBBT_scores_nd, median_MC_nd, s=40, edgecolors="black", 
                      color=colors["Non-dominant"])
    # Linear regression for Motor Acuity (Non-dominant)
    slope, intercept = np.polyfit(sBBT_scores_nd, median_MC_nd, 1)
    axs[0, 2].plot(x_vals, slope * x_vals + intercept, color='black', linestyle='--')
    axs[0, 2].set_ylabel("Motor acuity")
    axs[0, 2].set_title(f"ND: r = {corr_MC_nd:.2f}, p = {p_MC_nd_corr:.3f}")
    axs[0, 2].grid(False)

    # --- Bottom row: Dominant hand ---
    # Subplot 4: sBBT vs Duration (Dominant)
    axs[1, 0].scatter(sBBT_scores_dom, median_durations_dom, s=40, edgecolors="black", 
                      color=colors["Dominant"])
    # Linear regression for Duration (Dominant)
    x_vals_dom = np.linspace(min(sBBT_scores_dom), max(sBBT_scores_dom), 100)
    slope, intercept = np.polyfit(sBBT_scores_dom, median_durations_dom, 1)
    axs[1, 0].plot(x_vals_dom, slope * x_vals_dom + intercept, color='black', linestyle='--')
    axs[1, 0].set_xlabel("sBBT score")
    axs[1, 0].set_ylabel("Duration (s)")
    axs[1, 0].set_title(f"D: r = {corr_dur_dom:.2f}, p = {p_dur_dom_corr:.3f}")
    axs[1, 0].grid(False)

    # Subplot 5: sBBT vs Distance (Dominant)
    axs[1, 1].scatter(sBBT_scores_dom, median_distances_dom, s=40, edgecolors="black", 
                      color=colors["Dominant"])
    # Linear regression for Distance (Dominant)
    slope, intercept = np.polyfit(sBBT_scores_dom, median_distances_dom, 1)
    axs[1, 1].plot(x_vals_dom, slope * x_vals_dom + intercept, color='black', linestyle='--')
    axs[1, 1].set_xlabel("sBBT score")
    axs[1, 1].set_ylabel("Distance (mm)")
    axs[1, 1].set_title(f"D: r = {corr_dist_dom:.2f}, p = {p_dist_dom_corr:.3f}")
    axs[1, 1].grid(False)

    # Subplot 6: sBBT vs Motor Acuity (Dominant)
    axs[1, 2].scatter(sBBT_scores_dom, median_MC_dom, s=40, edgecolors="black", 
                      color=colors["Dominant"])
    # Linear regression for Motor Acuity (Dominant)
    slope, intercept = np.polyfit(sBBT_scores_dom, median_MC_dom, 1)
    axs[1, 2].plot(x_vals_dom, slope * x_vals_dom + intercept, color='black', linestyle='--')
    axs[1, 2].set_xlabel("sBBT score")
    axs[1, 2].set_ylabel("Motor acuity")
    axs[1, 2].set_title(f"D: r = {corr_MC_dom:.2f}, p = {p_MC_dom_corr:.3f}")
    axs[1, 2].grid(False)

    # Remove top and right spines for all subplots.
    for ax in axs.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()
    
    # Collect the computed correlation results.
    result = {
        "non_dominant": {
             "duration": {"r": corr_dur_nd, "p": p_dur_nd_corr},
             "distance": {"r": corr_dist_nd, "p": p_dist_nd_corr},
             "motor_acuity": {"r": corr_MC_nd, "p": p_MC_nd_corr}
        },
        "dominant": {
             "duration": {"r": corr_dur_dom, "p": p_dur_dom_corr},
             "distance": {"r": corr_dist_dom, "p": p_dist_dom_corr},
             "motor_acuity": {"r": corr_MC_dom, "p": p_MC_dom_corr}
        }
    }
    
    print("Computed correlation results:")
    print(result)
    return result

plot_sbbt_vs_median_metrics(subject_medians, All_dates, sBBTResult, overall_median_motor_acuity)



def plot_sbbt_vs_median_metrics(subject_medians, All_dates, sBBTResult, overall_median_motor_acuity):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pearsonr
    from statsmodels.stats.multitest import multipletests
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    colors = {"Non-dominant": "#A9A9A9", "Dominant": "#F0F0F0"}

    # Initialize lists
    sBBT_scores_nd, median_durations_nd, median_distances_nd = [], [], []
    sBBT_scores_dom, median_durations_dom, median_distances_dom = [], [], []

    subjects = list(subject_medians.keys())
    for idx, subject in enumerate(subjects):
        date_key = All_dates[idx]

        # Non-dominant
        score_nd = sBBTResult["non_dominant"][idx]
        med_duration_nd = subject_medians[date_key]["non_dominant"]["durations"]
        med_distance_nd = subject_medians[date_key]["non_dominant"]["distance"]

        # Dominant
        score_dom = sBBTResult["dominant"][idx]
        med_duration_dom = subject_medians[date_key]["dominant"]["durations"]
        med_distance_dom = subject_medians[date_key]["dominant"]["distance"]

        if (score_nd is not None and med_duration_nd is not None and med_distance_nd is not None):
            sBBT_scores_nd.append(score_nd)
            median_durations_nd.append(med_duration_nd)
            median_distances_nd.append(med_distance_nd)

        if (score_dom is not None and med_duration_dom is not None and med_distance_dom is not None):
            sBBT_scores_dom.append(score_dom)
            median_durations_dom.append(med_duration_dom)
            median_distances_dom.append(med_distance_dom)

    # Correlations
    corr_dd_nd, p_dd_nd = pearsonr(median_durations_nd, median_distances_nd)
    corr_dur_nd, p_dur_nd = pearsonr(sBBT_scores_nd, median_durations_nd)
    corr_dist_nd, p_dist_nd = pearsonr(sBBT_scores_nd, median_distances_nd)

    corr_dd_dom, p_dd_dom = pearsonr(median_durations_dom, median_distances_dom)
    corr_dur_dom, p_dur_dom = pearsonr(sBBT_scores_dom, median_durations_dom)
    corr_dist_dom, p_dist_dom = pearsonr(sBBT_scores_dom, median_distances_dom)

    # FDR correction
    pvals = [p_dd_nd, p_dur_nd, p_dist_nd, p_dd_dom, p_dur_dom, p_dist_dom]
    _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    p_dd_nd_corr, p_dur_nd_corr, p_dist_nd_corr, p_dd_dom_corr, p_dur_dom_corr, p_dist_dom_corr = pvals_corrected

    # Define synchronized ranges with padding
    all_durations = median_durations_nd + median_durations_dom
    all_distances = median_distances_nd + median_distances_dom
    all_sbbt = sBBT_scores_nd + sBBT_scores_dom

    duration_range = (min(all_durations) - 0.2, max(all_durations) + 0.2)
    distance_range = (min(all_distances) - 1, max(all_distances) + 1)
    sbbt_range     = (min(all_sbbt) - 5, max(all_sbbt) + 5)

    # Plot helper with significance levels
    def format_ax(ax, x, y, xlabel="", ylabel="", r=None, p=None, xlim=None, ylim=None, color=None, int_x=False):
        ax.scatter(x, y, s=40, edgecolors="black", color=color)
        if len(x) > 1 and len(y) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            x_vals = np.linspace(xlim[0], xlim[1], 100) if xlim else np.linspace(min(x), max(x), 100)
            ax.plot(x_vals, slope * x_vals + intercept, color="black", linestyle="--")

        if xlabel: 
            ax.set_xlabel(xlabel)
        if ylabel: 
            ax.set_ylabel(ylabel)

        # Annotate correlations with significance stars
        if r is not None and p is not None:
            sig_levels = [(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")]
            star = ""
            for threshold, symbol in sig_levels:
                if p < threshold:
                    star = symbol
                    break
            if star == "ns":  # only indicate signifcant stars
                star = ""
            ax.text(0.95, 0.95, f"r={r:.2f}{star}",
                    ha="right", va="top", transform=ax.transAxes, fontsize=20)

        if xlim: 
            ax.set_xlim(xlim)
        if ylim: 
            ax.set_ylim(ylim)

        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f' if not int_x else '%d'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Create figure with increased space between subplots
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Increase horizontal and vertical spacing

    # --- Non-dominant (top row) ---
    format_ax(axs[0, 0], median_durations_nd, median_distances_nd,
              ylabel="Error (mm)", r=corr_dd_nd, p=p_dd_nd_corr,
              xlim=duration_range, ylim=distance_range, color=colors["Non-dominant"])
    format_ax(axs[0, 1], sBBT_scores_nd, median_durations_nd,
              ylabel="Duration (s)", r=corr_dur_nd, p=p_dur_nd_corr,
              xlim=sbbt_range, ylim=duration_range, color=colors["Non-dominant"], int_x=True)
    format_ax(axs[0, 2], sBBT_scores_nd, median_distances_nd,
              ylabel="Error (mm)", r=corr_dist_nd, p=p_dist_nd_corr,
              xlim=sbbt_range, ylim=distance_range, color=colors["Non-dominant"], int_x=True)

    # --- Dominant (bottom row) ---
    format_ax(axs[1, 0], median_durations_dom, median_distances_dom,
              xlabel="Duration (s)", ylabel="Error (mm)", r=corr_dd_dom, p=p_dd_dom_corr,
              xlim=duration_range, ylim=distance_range, color=colors["Dominant"])
    format_ax(axs[1, 1], sBBT_scores_dom, median_durations_dom,
              xlabel="sBBT score", ylabel="Duration (s)", r=corr_dur_dom, p=p_dur_dom_corr,
              xlim=sbbt_range, ylim=duration_range, color=colors["Dominant"], int_x=True)
    format_ax(axs[1, 2], sBBT_scores_dom, median_distances_dom,
              xlabel="sBBT score", ylabel="Error (mm)", r=corr_dist_dom, p=p_dist_dom_corr,
              xlim=sbbt_range, ylim=distance_range, color=colors["Dominant"], int_x=True)

    plt.tight_layout()
    plt.show()

    # Print results
    print(f"Non-dominant Duration vs Error: r = {corr_dd_nd:.2f}, p = {p_dd_nd_corr:.3f}")
    print(f"Dominant Duration vs Error: r = {corr_dd_dom:.2f}, p = {p_dd_dom_corr:.3f}")

    return {
        "non_dominant": {
            "duration_distance": {"r": corr_dd_nd, "p": p_dd_nd_corr},
            "duration": {"r": corr_dur_nd, "p": p_dur_nd_corr},
            "distance": {"r": corr_dist_nd, "p": p_dist_nd_corr}
        },
        "dominant": {
            "duration_distance": {"r": corr_dd_dom, "p": p_dd_dom_corr},
            "duration": {"r": corr_dur_dom, "p": p_dur_dom_corr},
            "distance": {"r": corr_dist_dom, "p": p_dist_dom_corr}
        }
    }
    
plot_sbbt_vs_median_metrics(subject_medians, All_dates, sBBTResult, overall_median_motor_acuity)


def plot_sbbt_vs_median_metrics(subject_medians, All_dates, sBBTResult, overall_median_motor_acuity, config=plot_config_summary):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pearsonr

    # Extract general plot config
    gen = config['general']
    tick_direction = gen.get('tick_direction', 'out')
    axis_labels = config['axis_labels']
    
    # Initialize lists for non-dominant and dominant scores and durations.
    sBBT_scores_nd, durations_nd = [], []
    sBBT_scores_dom, durations_dom = [], []
    subjects = list(subject_medians.keys())
    
    for idx, subject in enumerate(subjects):
        date_key = All_dates[idx]
        # Non-dominant metrics
        score_nd = sBBTResult["non_dominant"][idx]
        med_duration_nd = subject_medians[date_key]["non_dominant"]["durations"]
        # Dominant metrics
        score_dom = sBBTResult["dominant"][idx]
        med_duration_dom = subject_medians[date_key]["dominant"]["durations"]
        if score_nd is not None and med_duration_nd is not None:
            sBBT_scores_nd.append(score_nd)
            durations_nd.append(med_duration_nd)
        if score_dom is not None and med_duration_dom is not None:
            sBBT_scores_dom.append(score_dom)
            durations_dom.append(med_duration_dom)
    
    # Determine fixed ticks as requested
    x_ticks = [0.5, 0.8, 1.1]
    y_ticks = [50, 70, 90]

    # Compute Pearson correlations
    r_nd, p_nd = pearsonr(durations_nd, sBBT_scores_nd)
    r_dom, p_dom = pearsonr(durations_dom, sBBT_scores_dom)

    # Plot separately for non-dominant and dominant
    for durations, scores, r, p, hand in zip(
        [durations_nd, durations_dom],
        [sBBT_scores_nd, sBBT_scores_dom],
        [r_nd, r_dom],
        [p_nd, p_dom],
        ["Non-dominant", "Dominant"]
    ):
        fig, ax = plt.subplots(figsize=gen['figsize'])
        
        # Scatter plot
        color = "#A9A9A9" if hand == "Non-dominant" else "#F0F0F0"
        ax.scatter(durations, scores, color=color, edgecolors="black", 
                   s=gen.get('marker_size', 50), alpha=1.0, zorder=3)
        
        # Regression line
        if len(durations) > 1:
            m, b = np.polyfit(durations, scores, 1)
            x_vals = np.linspace(min(x_ticks), max(x_ticks), 100)
            ax.plot(x_vals, m * x_vals + b, "--", color="black", linewidth=2)
        
        # Axis labels (update y-axis label)
        ax.set_xlabel(axis_labels['duration'], fontsize=gen['axis_label_font'])
        ax.set_ylabel("sBBT score (no. of blocks)", fontsize=gen['axis_label_font'])
        
        # Set fixed ticks and tick labels as requested
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='x', labelsize=gen['tick_label_font'], direction=tick_direction)
        ax.tick_params(axis='y', labelsize=gen['tick_label_font'], direction=tick_direction)
        
        # Correlation annotation (only annotate if significant)
        if p < 0.05:
            # Only display stars if significant; do not label "ns"
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            else:
                stars = "*"
            ax.text(0.1, 0.3, f"r = {r:.2f} {stars}",
                    transform=ax.transAxes, fontsize=gen['tick_label_font'], verticalalignment='top')
        # Optionally, if not significant, omit the annotation.
        else:
            ax.text(0.1, 0.3, f"r = {r:.2f}",
                    transform=ax.transAxes, fontsize=gen['tick_label_font'], verticalalignment='top')
        
        # Sample size annotation
        n = len(durations)
        ax.text(0.1, 0.1, f"n = {n} participants",
                transform=ax.transAxes, fontsize=gen['tick_label_font'], verticalalignment='bottom') if hand == "Non-dominant" else None
        
        # Grid and spines configuration
        ax.grid(gen.get('show_grid', False))
        if gen.get('hide_spines', True):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)


        # For duration: lower values imply fast performance.
        x_axis_colors = {"start": "fast", "end": "slow", "colors": ["green", "red"]}
        # For sBBT score: lower is low score, higher is high score.
        y_axis_colors = {"start": "     low     ", "end": "    high     ", "colors": ["red", "green"]}
    

        # Determine axis limits and offsets for annotations.
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        x_offset = 3
        y_offset = 0.09

        # Annotate the x-axis (duration) at both ends.
        ax.text(xmin, ymin - 1.5 * x_offset, x_axis_colors["start"],
                color=x_axis_colors["colors"][0], ha="center", va="top",
                fontsize=gen["tick_label_font"])
        ax.text(xmax, ymin - 1.5 * x_offset, x_axis_colors["end"],
                color=x_axis_colors["colors"][1], ha="center", va="top",
                fontsize=gen["tick_label_font"])

        # Annotate the y-axis (sBBT score) at both ends.
        ax.text(xmin - y_offset, ymax, y_axis_colors["start"],
                color=y_axis_colors["colors"][0], ha="right", va="center",
                fontsize=gen["tick_label_font"])
        ax.text(xmin - y_offset, ymin, y_axis_colors["end"],
                color=y_axis_colors["colors"][1], ha="right", va="bottom",
                fontsize=gen["tick_label_font"])


        plt.tight_layout()
        plt.show()

    print(f"Non-dominant: r = {r_nd:.2f}, p = {p_nd:.3f}")
    print(f"Dominant: r = {r_dom:.2f}, p = {p_dom:.3f}")

    return {
        "non_dominant": {"duration": {"r": r_nd, "p": p_nd}},
        "dominant": {"duration": {"r": r_dom, "p": p_dom}}
    }

plot_sbbt_vs_median_metrics(subject_medians, All_dates, sBBTResult, overall_median_motor_acuity, config=plot_config_summary)

# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------

def plot_phase_combined_single_figure(subject, hand, file_path, seg_indices,
                                      ballistic_phase, correction_phase, results,
                                      reach_speed_segments, placement_location_colors,
                                      marker="LFIN", frame_rate=200,
                                      figsize=(16, 9), show_icon=True, icon_path="/Users/yilinwu/Desktop/HandHoldBlock1.png",):
    """
    Plots 3D phase trajectories and kinematic signals in one combined figure using custom segment colors.
    """
    if not isinstance(seg_indices, list):
        seg_indices = [seg_indices]

    # Trajectory data
    traj_data = results[subject][hand][1][file_path]['traj_data']
    coord_prefix = "LFIN_" if hand.lower() == "left" else "RFIN_"
    coord_x = np.array(traj_data[coord_prefix + "X"])
    coord_y = np.array(traj_data[coord_prefix + "Y"])
    coord_z = np.array(traj_data[coord_prefix + "Z"])

    # Kinematic signals
    signals_space = results[subject][hand][1][file_path]['traj_space'][marker]
    position_full, velocity_full, acceleration_full, jerk_full = map(np.array, signals_space)
    signal_labels = ["Distance", "Velocity", "Acceleration", "Jerk"]

    signals = [position_full, velocity_full, acceleration_full, jerk_full]
    signal_types = ["from origin", "magnitude", "magnitude", "magnitude"]
    units = ["mm", "mm/s", "mm/s²", "mm/s³"]
    # colors_sig = {"full": "dimgray", "ballistic": "blue", "correction": "red"}
    colors_sig = {"full": "dimgray", "ballistic": "#0047ff", "correction": "#ffb800"}

    # Determine total time axis using first segment
    reach_start, reach_end = reach_speed_segments[subject][hand][file_path][seg_indices[0]]
    time_full = np.arange(reach_end - reach_start) / frame_rate

    # Create figure with GridSpec: 2 columns (3D + signals)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 5, figure=fig)  # 4 rows, 5 columns
    ax3d = fig.add_subplot(gs[:, :3], projection='3d')  # 3D plot spans all rows on left

    # --- Plot 3D trajectories ---
    for idx, seg_index in enumerate(seg_indices):
        ballistic_start, ballistic_end_idx = ballistic_phase[subject][hand][file_path][seg_index]
        corrective_start, corrective_end = correction_phase[subject][hand][file_path][seg_index]

        # Ballistic phase
        ax3d.plot(coord_x[ballistic_start:ballistic_end_idx],
                  coord_y[ballistic_start:ballistic_end_idx],
                  coord_z[ballistic_start:ballistic_end_idx],
                  color=colors_sig["ballistic"], linewidth=4,
                  label=f'Seg {seg_index+1} Ballistic' if idx == 0 else None, zorder=1)
        # Corrective phase
        ax3d.plot(coord_x[corrective_start:corrective_end],
                  coord_y[corrective_start:corrective_end],
                  coord_z[corrective_start:corrective_end],
                  color=colors_sig["correction"], linewidth=4,
                  label=f'Seg {seg_index+1} correction' if idx == 0 else None, zorder=1)
        # # Start/end points
        # ax3d.scatter(coord_x[ballistic_start], coord_y[ballistic_start], coord_z[ballistic_start],
        #              color='blue', edgecolors= "white",  s=100, marker='o', zorder=2)
        # ax3d.scatter(coord_x[corrective_end-1], coord_y[corrective_end-1], coord_z[corrective_end-1],
        #              color='red', edgecolors= "white", s=100, marker='o', zorder=2)
        # ax3d.scatter(coord_x[ballistic_end_idx], coord_y[ballistic_end_idx], coord_z[ballistic_end_idx],
        #              color='cyan', s=40, marker='o', zorder=2)

        # Optionally add icons
        if show_icon and icon_path:
            import matplotlib.image as mpimg
            img = mpimg.imread(icon_path)

            def add_image(ax, xs, ys, zs, img, zoom=0.06):
                x2, y2, _ = proj_transform(xs, ys, zs, ax.get_proj())
                imagebox = OffsetImage(img, zoom=zoom)
                ab = AnnotationBbox(imagebox, (x2, y2), frameon=False, xycoords='data')
                ax.add_artist(ab)

            add_image(ax3d, coord_x[ballistic_start]-25, coord_y[ballistic_start], coord_z[ballistic_start]-5, img)
            add_image(ax3d, coord_x[corrective_end], coord_y[corrective_end]+5, coord_z[corrective_end]-70, img)


    
    ax3d.set_xlabel("X (mm)", labelpad=5, fontsize=12)
    ax3d.set_ylabel("Y (mm)", labelpad=5, fontsize=12)
    ax3d.set_zlabel("Z (mm)", labelpad=2, fontsize=12)
    # ax3d.set_xticks([-250, 0, 250])
    # ax3d.set_yticks([100, 115, 130])
    # ax3d.set_zticks([850, 975, 1100])
    # ax3d.legend(frameon=False, loc=[0.5, 0.8])
    ax3d.set_xticks([-250, 0, 250])
    ax3d.set_xticklabels([-250, 0, 250])
    ax3d.set_yticks([-50, 50, 150])
    ax3d.set_yticklabels([-50, 50, 150])
    ax3d.set_zticks([800, 950, 1100])
    ax3d.set_zticklabels([800, 950, 1100])
    ax3d.set_xlim([-250, 250])
    ax3d.set_ylim([-50, 150])
    ax3d.set_zlim([800, 1100])
    ax3d.set_box_aspect([5, 2, 3])

    # --- Plot kinematic signals ---
    for i, (sig, label, unit) in enumerate(zip(signals, signal_labels, units)):
        ax = fig.add_subplot(gs[i, 3:])  # signals occupy right 3 columns
        ax.plot(time_full, sig[reach_start:reach_end], color=colors_sig["full"], linewidth=4, zorder=1)

        ballistic_start, ballistic_end_idx = ballistic_phase[subject][hand][file_path][seg_indices[0]]
        corrective_start, corrective_end = correction_phase[subject][hand][file_path][seg_indices[0]]
        time_ballistic = (np.arange(ballistic_start, ballistic_end_idx) - reach_start) / frame_rate
        time_corrective = (np.arange(ballistic_end_idx, corrective_end) - reach_start) / frame_rate

        ax.plot(time_ballistic, sig[ballistic_start:ballistic_end_idx],
                color=colors_sig["ballistic"], linewidth=4, zorder=1)
        ax.plot(time_corrective, sig[ballistic_end_idx:corrective_end],
                color=colors_sig["correction"], linewidth=4, zorder=1)
        # ax.scatter((ballistic_end_idx - reach_start)/frame_rate, sig[ballistic_end_idx],
        #            color="cyan", s=40, marker="o", zorder=2)

        ax.set_ylabel(f"{label}\n{signal_types[i]}\n({unit})", labelpad=5, fontsize=12, rotation=0)
        ax.yaxis.set_label_coords(-0.25, 0)
        ax.relim()       # Recalculate limits
        ax.autoscale()   # Apply new limits
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if i == 0:
            ax.legend(["Full", "Ballistic", "Correction"], frameon=False, fontsize=10)
        if i < len(signals) - 1:
            ax.set_xticklabels([])
        if i == len(signals) - 1:
            ax.set_xlabel("Time (s)", labelpad=5, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.7, hspace=0.3)
    plt.show()

plot_phase_combined_single_figure(
    subject="07/22/HW",
    hand="left",
    file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv',
    seg_indices=[2],
    ballistic_phase=ballistic_phase,
    correction_phase=correction_phase,
    results=results,
    reach_speed_segments=reach_speed_segments,
    placement_location_colors=placement_location_colors,
    marker="LFIN",
    frame_rate=200,
    figsize=(13, 5), 
    show_icon=False,
    icon_path="/Users/yilinwu/Desktop/HandHoldBlock1.png",
)






def plot_phase_combined_single_figure(subject, hand, file_path, seg_indices,
                                      ballistic_phase, correction_phase, results,
                                      reach_speed_segments, placement_location_colors,
                                      marker="LFIN", frame_rate=200,
                                      figsize=(16, 9), show_icon=True, icon_path="/Users/yilinwu/Desktop/HandHoldBlock1.png",):
    """
    Plots 3D phase trajectories and kinematic signals in one combined figure using custom segment colors.
    Instead of plotting the acceleration and jerk magnitude, this version plots the acceleration-along-velocity
    and jerk-along-velocity.
    """
    if not isinstance(seg_indices, list):
        seg_indices = [seg_indices]

    # Trajectory data
    traj_data = results[subject][hand][1][file_path]['traj_data']
    coord_prefix = "LFIN_" if hand.lower() == "left" else "RFIN_"
    coord_x = np.array(traj_data[coord_prefix + "X"])
    coord_y = np.array(traj_data[coord_prefix + "Y"])
    coord_z = np.array(traj_data[coord_prefix + "Z"])

    # Kinematic signals (unused in the modified plot for acceleration and jerk)
    signals_space = results[subject][hand][1][file_path]['traj_space'][marker]
    position_full, velocity_full, acceleration_full, jerk_full = map(np.array, signals_space)
    # Original labels and units for the kinematic signals
    signal_labels = ["Distance\nfrom\norigin", "Velocity\nmagnitude", "Acceleration", "Jerk"]
    units = ["mm", "mm/s", "mm/s²", "mm/s³"]
    # We'll update the acceleration and jerk signals to plot acc_along_vel and jerk_along_vel.
    # Use the same color scheme:
    colors_sig = {"full": "dimgray", "ballistic": "#0047ff", "correction": "#ffb800"}

    # Determine total time axis using first segment
    reach_start, reach_end = reach_speed_segments[subject][hand][file_path][seg_indices[0]]
    time_full = np.arange(reach_end - reach_start) / frame_rate

    # ---- Compute acc_along_vel and jerk_along_vel ----
    dt = 1 / frame_rate
    # Compute velocity components from the trajectory data
    vx = np.gradient(coord_x, dt)
    vy = np.gradient(coord_y, dt)
    vz = np.gradient(coord_z, dt)
    # Compute acceleration components from velocity components
    ax_data = np.gradient(vx, dt)
    ay_data = np.gradient(vy, dt)
    az_data = np.gradient(vz, dt)
    # Compute speed
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    # Compute acceleration along the direction of velocity
    acc_along_vel = (vx * ax_data + vy * ay_data + vz * az_data) / (v + 1e-8)
    # Compute jerk along velocity as the gradient of acc_along_vel
    jerk_along_vel = np.gradient(acc_along_vel, dt)
    # Replace the original acceleration and jerk signals with the computed along-velocity versions
    # and update their labels.
    acceleration_full = acc_along_vel
    jerk_full = jerk_along_vel
    signal_labels[2] = "Acceleration\nalong\nvelocity"
    signal_labels[3] = "Jerk\nalong\nvelocity"
    # Update the signals list to use the modified values.
    signals = [position_full, velocity_full, acceleration_full, jerk_full]

    # Create figure with GridSpec: 2 sections (3D + signals)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 5, figure=fig)  # 4 rows, 5 columns
    ax3d = fig.add_subplot(gs[:, :3], projection='3d')  # 3D plot spans all rows on left

    # --- Plot 3D trajectories ---
    for idx, seg_index in enumerate(seg_indices):
        ballistic_start, ballistic_end_idx = ballistic_phase[subject][hand][file_path][seg_index]
        corrective_start, corrective_end = correction_phase[subject][hand][file_path][seg_index]

        # Ballistic phase
        ax3d.plot(coord_x[ballistic_start:ballistic_end_idx],
                  coord_y[ballistic_start:ballistic_end_idx],
                  coord_z[ballistic_start:ballistic_end_idx],
                  color=colors_sig["ballistic"], linewidth=4,
                  label=f'Ballistic' if idx == 0 else None, zorder=1)
        # Corrective phase
        ax3d.plot(coord_x[corrective_start:corrective_end],
                  coord_y[corrective_start:corrective_end],
                  coord_z[corrective_start:corrective_end],
                  color=colors_sig["correction"], linewidth=4,
                  label=f'Correction' if idx == 0 else None, zorder=1)
        ax3d.plot(coord_x[reach_start:reach_end],
              coord_y[reach_start:reach_end],
              coord_z[reach_start:reach_end],
              color="gray", linewidth=4, linestyle='-',
              label=f'Full' if idx == 0 else None, zorder=0)

        # Optionally add icons
        if show_icon and icon_path:
            import matplotlib.image as mpimg
            img = mpimg.imread(icon_path)

            def add_image(ax, xs, ys, zs, img, zoom=0.06):
                x2, y2, _ = proj_transform(xs, ys, zs, ax.get_proj())
                imagebox = OffsetImage(img, zoom=zoom)
                ab = AnnotationBbox(imagebox, (x2, y2), frameon=False, xycoords='data')
                ax.add_artist(ab)

            add_image(ax3d, coord_x[ballistic_start]-25, coord_y[ballistic_start], coord_z[ballistic_start]-5, img)
            add_image(ax3d, coord_x[corrective_end], coord_y[corrective_end]+5, coord_z[corrective_end]-70, img)

    ax3d.set_xlabel("X (mm)", labelpad=5, fontsize=12)
    ax3d.set_ylabel("Y (mm)", labelpad=5, fontsize=12)
    ax3d.set_zlabel("Z (mm)", labelpad=2, fontsize=12)
    ax3d.set_xticks([-250, 0, 250])
    ax3d.set_xticklabels([-250, 0, 250])
    ax3d.set_yticks([-50, 50, 150])
    ax3d.set_yticklabels([-50, 50, 150])
    ax3d.set_zticks([800, 950, 1100])
    ax3d.set_zticklabels([800, 950, 1100])
    ax3d.set_xlim([-250, 250])
    ax3d.set_ylim([-50, 150])
    ax3d.set_zlim([800, 1100])
    ax3d.set_box_aspect([5, 2, 3])
    ax3d.legend(frameon=False, loc=[0.05, -0.15], fontsize=12)


    # --- Plot kinematic signals ---
    for i, (sig, label, unit) in enumerate(zip(signals, signal_labels, units)):
        ax = fig.add_subplot(gs[i, 3:])  # signals occupy right 3 columns
        ax.plot(time_full, sig[reach_start:reach_end], color=colors_sig["full"], linewidth=4, zorder=1)

        # Use the ballistic and correction phase indices to overlay segmented plots
        ballistic_start, ballistic_end_idx = ballistic_phase[subject][hand][file_path][seg_indices[0]]
        corrective_start, corrective_end = correction_phase[subject][hand][file_path][seg_indices[0]]
        time_ballistic = (np.arange(ballistic_start, ballistic_end_idx) - reach_start) / frame_rate
        time_corrective = (np.arange(ballistic_end_idx, corrective_end) - reach_start) / frame_rate

        ax.plot(time_ballistic, sig[ballistic_start:ballistic_end_idx],
                color=colors_sig["ballistic"], linewidth=4, zorder=1)
        ax.plot(time_corrective, sig[ballistic_end_idx:corrective_end],
                color=colors_sig["correction"], linewidth=4, zorder=1)

        if i >= 2:
            h = ax.axhline(0, color='black', linestyle='--', linewidth=1, label='y = 0')
            ax.legend(handles=[h], labels=['y = 0'], frameon=False, fontsize=14, loc='upper right') if i == 2 else None
        else:
            None

        ax.set_ylabel(f"{label}\n({unit})", labelpad=5, fontsize=12, rotation=0)
        ax.yaxis.set_label_coords(-0.285, 0)
        ax.relim()       # Recalculate limits
        ax.autoscale()   # Apply new limits
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # if i == 0:
        #     ax.legend(["Full", "Ballistic", "Correction"], frameon=False, fontsize=10)
        if i < len(signals) - 1:
            ax.set_xticklabels([])
        if i == len(signals) - 1:
            ax.set_xlabel("Time (s)", labelpad=5, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=1, hspace=0.3)
    plt.show()


plot_phase_combined_single_figure(
    subject="07/22/HW",
    hand="left",
    file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv',
    seg_indices=[7],
    ballistic_phase=ballistic_phase,
    correction_phase=correction_phase,
    results=results,
    reach_speed_segments=reach_speed_segments,
    placement_location_colors=placement_location_colors,
    marker="LFIN",
    frame_rate=200,
    figsize=(13, 5), 
    show_icon=False,
    icon_path="/Users/yilinwu/Desktop/HandHoldBlock1.png",
)





def plot_phase_combined_single_figure(subject, hand, file_path, seg_indices,
                                      ballistic_phase, correction_phase, results,
                                      reach_speed_segments, placement_location_colors,
                                      marker="LFIN", frame_rate=200,
                                      figsize=(16, 9), show_icon=True, icon_path="/Users/yilinwu/Desktop/HandHoldBlock1.png",):
    """
    Plots 3D phase trajectories and kinematic signals in one combined figure using custom segment colors.
    Instead of plotting the acceleration and jerk magnitude, this version plots the acceleration-along-velocity
    and jerk-along-velocity. The subplot for distance from origin has been removed.
    """

    import matplotlib.ticker as ticker

    if not isinstance(seg_indices, list):
        seg_indices = [seg_indices]

    # Trajectory data
    traj_data = results[subject][hand][1][file_path]['traj_data']
    coord_prefix = "LFIN_" if hand.lower() == "left" else "RFIN_"
    coord_x = np.array(traj_data[coord_prefix + "X"])
    coord_y = np.array(traj_data[coord_prefix + "Y"])
    coord_z = np.array(traj_data[coord_prefix + "Z"])

    # Kinematic signals (unused in the modified plot for acceleration and jerk)
    signals_space = results[subject][hand][1][file_path]['traj_space'][marker]
    position_full, velocity_full, acceleration_full, jerk_full = map(np.array, signals_space)
    # Original labels and units for the kinematic signals
    # Removed "Distance from origin" subplot.
    signal_labels = ["V(t)", "Acceleration\nalong\nvelocity", "Jerk\nalong\nvelocity"]
    units = ["mm/s", "mm/s²", "mm/s³"]
    # We'll update the acceleration and jerk signals to plot acc_along_vel and jerk_along_vel.
    # Use the same color scheme:
    colors_sig = {"full": "dimgray", "ballistic": "#0047ff", "correction": "#ffb800"}

    # Determine total time axis using first segment
    reach_start, reach_end = reach_speed_segments[subject][hand][file_path][seg_indices[0]]
    time_full = np.arange(reach_end - reach_start) / frame_rate

    # ---- Compute acc_along_vel and jerk_along_vel ----
    dt = 1 / frame_rate
    # Compute velocity components from the trajectory data
    vx = np.gradient(coord_x, dt)
    vy = np.gradient(coord_y, dt)
    vz = np.gradient(coord_z, dt)
    # Compute acceleration components from velocity components
    ax_data = np.gradient(vx, dt)
    ay_data = np.gradient(vy, dt)
    az_data = np.gradient(vz, dt)
    # Compute speed
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    # Compute acceleration along the direction of velocity
    acc_along_vel = (vx * ax_data + vy * ay_data + vz * az_data) / (v + 1e-8)
    # Compute jerk along velocity as the gradient of acc_along_vel
    jerk_along_vel = np.gradient(acc_along_vel, dt)
    # Replace the original acceleration and jerk signals with the computed along-velocity versions
    # and update their labels.
    acceleration_full = acc_along_vel
    jerk_full = jerk_along_vel
    # Update the signals list to use the modified values.
    signals = [velocity_full, acceleration_full, jerk_full]

    # Create figure with GridSpec: 2 sections (3D + signals)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 5, figure=fig)  # Now using 3 rows for the 3 kinematic signals.
    ax3d = fig.add_subplot(gs[:, :3], projection='3d')  # 3D plot spans all rows on left

    # --- Plot 3D trajectories ---
    for idx, seg_index in enumerate(seg_indices):
        ballistic_start, ballistic_end_idx = ballistic_phase[subject][hand][file_path][seg_index]
        corrective_start, corrective_end = correction_phase[subject][hand][file_path][seg_index]

        # Ballistic phase
        ax3d.plot(coord_x[ballistic_start:ballistic_end_idx],
                  coord_y[ballistic_start:ballistic_end_idx],
                  coord_z[ballistic_start:ballistic_end_idx],
                  color=colors_sig["ballistic"], linewidth=4,
                  label=f'Ballistic' if idx == 0 else None, zorder=1)
        # Corrective phase
        ax3d.plot(coord_x[corrective_start:corrective_end],
                  coord_y[corrective_start:corrective_end],
                  coord_z[corrective_start:corrective_end],
                  color=colors_sig["correction"], linewidth=4,
                  label=f'Correction' if idx == 0 else None, zorder=1)
        ax3d.plot(coord_x[reach_start:reach_end],
                  coord_y[reach_start:reach_end],
                  coord_z[reach_start:reach_end],
                  color="gray", linewidth=4, linestyle='-',
                  label=f'Full' if idx == 0 else None, zorder=0)

        # Optionally add icons
        if show_icon and icon_path:
            import matplotlib.image as mpimg
            img = mpimg.imread(icon_path)

            def add_image(ax, xs, ys, zs, img, zoom=0.06):
                x2, y2, _ = proj_transform(xs, ys, zs, ax.get_proj())
                imagebox = OffsetImage(img, zoom=zoom)
                ab = AnnotationBbox(imagebox, (x2, y2), frameon=False, xycoords='data')
                ax.add_artist(ab)

            add_image(ax3d, coord_x[ballistic_start]-25, coord_y[ballistic_start], coord_z[ballistic_start]-5, img)
            add_image(ax3d, coord_x[corrective_end], coord_y[corrective_end]+5, coord_z[corrective_end]-70, img)

    ax3d.set_xlabel("X (mm)", labelpad=5, fontsize=12)
    ax3d.set_ylabel("Y (mm)", labelpad=5, fontsize=12)
    ax3d.set_zlabel("Z (mm)", labelpad=2, fontsize=12)
    ax3d.set_xticks([-250, 0, 250])
    ax3d.set_xticklabels([-250, 0, 250])
    ax3d.set_yticks([-50, 50, 150])
    ax3d.set_yticklabels([-50, 50, 150])
    ax3d.set_zticks([800, 950, 1100])
    ax3d.set_zticklabels([800, 950, 1100])
    ax3d.set_xlim([-250, 250])
    ax3d.set_ylim([-50, 150])
    ax3d.set_zlim([800, 1100])
    ax3d.set_box_aspect([5, 2, 3])
    ax3d.legend(frameon=False, loc=[0.05, -0.15], fontsize=12)

    # --- Plot kinematic signals ---
    for i, (sig, label, unit) in enumerate(zip(signals, signal_labels, units)):
        ax = fig.add_subplot(gs[i, 3:])  # signals occupy right 2 columns per row
        ax.plot(time_full, sig[reach_start:reach_end], color=colors_sig["full"], linewidth=4, zorder=1)

        # Use the ballistic and correction phase indices to overlay segmented plots
        ballistic_start, ballistic_end_idx = ballistic_phase[subject][hand][file_path][seg_indices[0]]
        corrective_start, corrective_end = correction_phase[subject][hand][file_path][seg_indices[0]]
        time_ballistic = (np.arange(ballistic_start, ballistic_end_idx) - reach_start) / frame_rate
        time_corrective = (np.arange(ballistic_end_idx, corrective_end) - reach_start) / frame_rate

        ax.plot(time_ballistic, sig[ballistic_start:ballistic_end_idx],
                color=colors_sig["ballistic"], linewidth=4, zorder=1)
        ax.plot(time_corrective, sig[ballistic_end_idx:corrective_end],
                color=colors_sig["correction"], linewidth=4, zorder=1)

        if i >= 1:
            h = ax.axhline(0, color='black', linestyle='--', linewidth=1, label='y = 0')
            if i == 1:
                ax.legend(handles=[h], labels=['y = 0'], frameon=False, fontsize=14, loc='upper right')
        ax.set_ylabel(f"{label}\n({unit})", labelpad=5, fontsize=12, rotation=0)
        ax.yaxis.set_label_coords(-0.25, 0.3)
        # --- Force formatter ---
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
        ax.relim()
        ax.autoscale()
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i < len(signals) - 1:
            ax.set_xticklabels([])
        if i == len(signals) - 1:
            ax.set_xlabel("Time (s)", labelpad=5, fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=1, hspace=0.3)
    plt.show()


plot_phase_combined_single_figure(
    subject="07/22/HW",
    hand="left",
    file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv',
    seg_indices=[7],
    ballistic_phase=ballistic_phase,
    correction_phase=correction_phase,
    results=results,
    reach_speed_segments=reach_speed_segments,
    placement_location_colors=placement_location_colors,
    marker="LFIN",
    frame_rate=200,
    figsize=(12, 4), 
    show_icon=False,
    icon_path="/Users/yilinwu/Desktop/HandHoldBlock1.png",
)











# -------------------------------------------------------------------------------------------------------------------
# # -------------------------------------------------------------------------------------------------------------------
# Hypothesis 1 : smoother movements will be faster
# Hypothesis 2 : smoother movements will be accurate
# Hypothesis 3 : smoother movements will be both faster and more accurate
import numpy as np
import matplotlib.pyplot as plt

def plot_smoothness_tradeoff(hypothesis):
    """
    Plots a smoothness vs performance tradeoff line plot based on hypothesis.
    
    Parameters
    ----------
    hypothesis : str
        One of "speed", "accuracy", "motor_acuity"
    """
    # X-axis: Smoothness (0 = unsmooth, 1 = very smooth)
    x = np.linspace(0, 1, 100)

    # Define Y-axis based on hypothesis
    if hypothesis == "speed":  # smoother movements are faster
        y = -1.5 * x + 2  # duration (s)
        y_label = "Duration (s)"
        y_axis_colors = {"start": "    fast     ", "end": "    slow     ", "colors": ["green", "red"]}
    elif hypothesis == "accuracy":  # smoother movements are more accurate
        y = -10 * x + 20  # error (mm)
        y_label = "Error (mm)"
        y_axis_colors = {"start": "accurate", "end": "inaccurate", "colors": ["green", "red"]}
    elif hypothesis == "motor_acuity":  # smoother movements are both faster & accurate
        y = 15 * x - 5  # motor acuity
        y_label = "Motor Acuity"
        y_axis_colors = {"start": "Fast &\naccurate", "end": "Slow &\ninaccurate", "colors": ["green","red" ]} 
    else:
        raise ValueError("Invalid hypothesis")

    # Plot configuration
    cfg = dict(
        general=dict(
            figsize=(5, 4),
            axis_label_font=20,
            tick_label_font=20,
            title_font=16,
            label_offset=0.12,
            showGrid=True
        ),
        line=dict(
            linewidth=2,
            show_markers=True
        ),
        axis_labels=dict(
            smoothness="Smoothness",
            duration="Duration (s)",
            distance="Error (mm)",
            motor_acuity="Motor Acuity"
        ),
        axis_colors=dict(
            x={
                "Smoothness": {"start": "unsmooth", "end": "smooth", "colors": ["red", "green"]}
            },
            y={
                y_label: y_axis_colors
            }
        )
    )

    general = cfg["general"]
    line_cfg = cfg["line"]
    axis_labels = cfg["axis_labels"]
    axis_colors = cfg["axis_colors"]
    label_offset = general.get("label_offset", 0.08)
    showGrid = general.get("showGrid", False)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=general["figsize"])

    # Plot line
    ax.plot(
        x,
        y,
        color='black',
        linewidth=line_cfg.get("linewidth", 2),
        marker='o' if line_cfg.get("show_markers", False) else None
    )

    # Set axis labels
    ax.set_xlabel(axis_labels.get("smoothness", "Smoothness"), fontsize=general["axis_label_font"])
    ax.set_ylabel(y_label, fontsize=general["axis_label_font"])

    # Remove numeric ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Restore x/y spines
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Axis ranges for offsetting labels
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    # X-axis labels
    x_cfg = axis_colors.get("x", {}).get(axis_labels.get("smoothness", "Smoothness"), {})
    if x_cfg:
        ax.text(x[0], y.min() - 1.5*label_offset*y_range, x_cfg["start"],
                color=x_cfg["colors"][0], ha="center", va="top",
                fontsize=general["tick_label_font"])
        ax.text(x[-1], y.min() - 1.5*label_offset*y_range, x_cfg["end"],
                color=x_cfg["colors"][1], ha="center", va="top",
                fontsize=general["tick_label_font"])

    # Y-axis labels
    y_cfg = axis_colors.get("y", {}).get(y_label, {})
    if y_cfg:
        ax.text(x.min() - label_offset*x_range, y[-1], y_cfg["start"],
                color=y_cfg["colors"][0], ha="right", va="center",
                fontsize=general["tick_label_font"])
        ax.text(x.min() - label_offset*x_range, y[0], y_cfg["end"],
                color=y_cfg["colors"][1], ha="right", va="center",
                fontsize=general["tick_label_font"])

    # plt.title(f"Hypothesis: {hypothesis.replace('_', ' ').capitalize()}", fontsize=general["title_font"])
    ax.set_aspect('auto')

    plt.tight_layout()
    plt.grid(showGrid)
    plt.show()


# Example usage
plot_smoothness_tradeoff("speed")
plot_smoothness_tradeoff("accuracy")
plot_smoothness_tradeoff("motor_acuity")

def plot_smoothness_tradeoff(hypothesis):
    """
    Plots a speed–accuracy tradeoff where Duration is shown on the x-axis and Error on the y-axis.
    For "speed_accuracy": longer durations (x) yield lower errors (y).
    
    Parameters
    ----------
    hypothesis : str
        Use "speed_accuracy" to plot Duration vs Error.
    """
    import matplotlib.pyplot as plt

    if hypothesis == "speed_accuracy":
        # X-axis: Duration (s). For example, from 0.5 to 2.0 s.
        x = np.linspace(0.5, 2.0, 100)
        # Y-axis: Error (mm). Assume a linear tradeoff: longer durations lead to lower errors.
        # When duration is 0.5 s, error is high; when duration is 2.0 s, error is low.
        y = -2 * x + 5  # e.g., at x=0.5, y=4 mm; at x=2.0, y=1 mm.
        x_label = "Duration (s)"
        y_label = "Error (mm)"
        # Define axis color configurations for annotations.
        x_axis_colors = {"start": "    fast     ", "end": "    slow     ", "colors": ["green", "red"]}
        y_axis_colors = {"start": "accurate", "end": "inaccurate", "colors": ["green", "red"]}
    else:
        # Fallback: identity plot.
        x = np.linspace(0, 1, 100)
        y = x
        x_label = "X"
        y_label = "Y"
        x_axis_colors = {}
        y_axis_colors = {}

    # Configuration dictionary.
    cfg = {
        "general": {
            "figsize": (5, 4),
            "axis_label_font": 20,
            "tick_label_font": 20,
            "title_font": 16,
            "label_offset": 0.12,
            "showGrid": True
        },
        "line": {
            "linewidth": 2,
            "show_markers": True
        },
        "axis_labels": {
            "duration": "Duration (s)",
            "error": "Error (mm)"
        },
        "axis_colors": {
            "x": { "Duration (s)": x_axis_colors },
            "y": { "Error (mm)": y_axis_colors }
        }
    }

    general = cfg["general"]
    line_cfg = cfg["line"]
    axis_labels = cfg["axis_labels"]
    axis_colors = cfg["axis_colors"]
    label_offset = general.get("label_offset", 0.08)
    showGrid = general.get("showGrid", False)

    fig, ax = plt.subplots(figsize=general["figsize"])

    # Plot the tradeoff line.
    ax.plot(
        x,
        y,
        color='black',
        linewidth=line_cfg.get("linewidth", 2),
        marker='o' if line_cfg.get("show_markers", False) else None
    )

    # Set proper axis labels.
    ax.set_xlabel(axis_labels.get("duration", "Duration (s)"), fontsize=general["axis_label_font"])
    ax.set_ylabel(axis_labels.get("error", "Error (mm)"), fontsize=general["axis_label_font"])

    # Remove default ticks.
    ax.set_xticks([])
    ax.set_yticks([])

    # Ensure spines are visible only on left and bottom.
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Compute ranges for annotation offsets.
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    # X-axis annotations.
    x_cfg = axis_colors.get("x", {}).get(axis_labels.get("duration", "Duration (s)"), {})
    if x_cfg:
        ax.text(x[0], y.min() - label_offset * y_range, x_cfg["start"],
                color=x_cfg["colors"][0], ha="center", va="top",
                fontsize=general["tick_label_font"])
        ax.text(x[-1], y.min() - label_offset * y_range, x_cfg["end"],
                color=x_cfg["colors"][1], ha="center", va="top",
                fontsize=general["tick_label_font"])

    # Y-axis annotations.
    y_cfg = axis_colors.get("y", {}).get(axis_labels.get("error", "Error (mm)"), {})
    if y_cfg:
        ax.text(x.min() - label_offset * x_range, y[-1], y_cfg["start"],
                color=y_cfg["colors"][0], ha="right", va="center",
                fontsize=general["tick_label_font"])
        ax.text(x.min() - label_offset * x_range, y[0], y_cfg["end"],
                color=y_cfg["colors"][1], ha="right", va="bottom",
                fontsize=general["tick_label_font"])

    ax.set_aspect('auto')
    plt.tight_layout()
    plt.grid(showGrid)
    plt.show()

plot_smoothness_tradeoff("speed_accuracy")


def plot_smoothness_tradeoff(hypothesis):
    """
    Plots a trade-off line based on the hypothesis.
    
    Two supported hypotheses:
    
    1. "sBBT_duration": When movements are faster (shorter durations), the sBBT score (number of blocks)
       is higher. For example, if duration ranges from 0.5 to 2.5 s, then:
           - At 0.5 s, sBBT score is 10 (fast performance)
           - At 2.5 s, sBBT score is 0 (slow performance)
       The relationship is: sBBT_score = 12.5 - 5 * duration.
       
    2. "sBBT_error": When error increases, the sBBT score is also higher. For example, if error ranges 
       from 1 to 4 mm, then:
           - At 1 mm, sBBT score is 0 
           - At 4 mm, sBBT score is 10 
       The relationship is: sBBT_score = (error - 1) / 0.3.
    
    Parameters
    ----------
    hypothesis : str
        Either "sBBT_duration" or "sBBT_error".
    """
    import matplotlib.pyplot as plt

    if hypothesis == "sBBT_duration":
        # x-axis: Duration (s) from 0.5 to 2.5.
        x = np.linspace(0.5, 2.5, 100)
        # Inverse relationship: higher sBBT score means shorter duration.
        # Solve: duration = -0.2*score + 2.5  => score = 12.5 - 5*duration.
        y = 12.5 - 5 * x  
        x_label = "Duration (s)"
        y_label = "sBBT Score (no. of blocks)"
        # For duration: lower values imply fast performance.
        x_axis_colors = {"start": "fast", "end": "slow", "colors": ["green", "red"]}
        # For sBBT score: lower is low score, higher is high score.
        y_axis_colors = {"start": "     low     ", "end": "    high     ", "colors": ["red", "green"]}
    elif hypothesis == "sBBT_error":
        # x-axis: Error (mm) from 1 to 4.
        x = np.linspace(1, 4, 100)
        # Direct relationship: higher error gives higher sBBT score.
        # Given: error = 0.3*score + 1  => score = (error - 1) / 0.3.
        y = (x - 1) / 0.3  
        x_label = "Error (mm)"
        y_label = "sBBT Score (no. of blocks)"
        x_axis_colors = {"start": "low error", "end": "high error", "colors": ["green", "red"]}
        y_axis_colors = {"start": "low", "end": "high", "colors": ["red", "green"]}
    else:
        # Fallback: identity plot.
        x = np.linspace(0, 1, 100)
        y = x
        x_label = "X"
        y_label = "Y"
        x_axis_colors = {}
        y_axis_colors = {}

    # Configuration dictionary.
    cfg = {
        "general": {
            "figsize": (5, 4),
            "axis_label_font": 20,
            "tick_label_font": 20,
            "title_font": 16,
            "label_offset": 0.12,
            "showGrid": True
        },
        "line": {
            "linewidth": 2,
            "show_markers": True
        },
        "axis_labels": {
            "x": x_label,
            "y": y_label
        },
        "axis_colors": {
            "x": {x_label: x_axis_colors},
            "y": {y_label: y_axis_colors}
        }
    }

    general = cfg["general"]
    line_cfg = cfg["line"]
    axis_labels_cfg = cfg["axis_labels"]
    axis_colors_cfg = cfg["axis_colors"]
    label_offset = general.get("label_offset", 0.08)
    showGrid = general.get("showGrid", False)

    fig, ax = plt.subplots(figsize=general["figsize"])

    # Plot the trade-off line.
    ax.plot(
        x,
        y,
        color='black',
        linewidth=line_cfg.get("linewidth", 2),
        marker='o' if line_cfg.get("show_markers", False) else None
    )

    # Set proper axis labels.
    ax.set_xlabel(axis_labels_cfg.get("x", "X"), fontsize=general["axis_label_font"])
    ax.set_ylabel(axis_labels_cfg.get("y", "Y"), fontsize=general["axis_label_font"])

    # Remove default ticks.
    ax.set_xticks([])
    ax.set_yticks([])

    # Ensure spines are visible only on left and bottom.
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Compute ranges for annotation offsets.
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    # X-axis annotations.
    x_cfg = axis_colors_cfg.get("x", {}).get(axis_labels_cfg.get("x", ""), {})
    if x_cfg:
        ax.text(x[0], y.min() - 1.5 * label_offset * y_range, x_cfg["start"],
                color=x_cfg["colors"][0], ha="center", va="top",
                fontsize=general["tick_label_font"])
        ax.text(x[-1], y.min() - 1.5 * label_offset * y_range, x_cfg["end"],
                color=x_cfg["colors"][1], ha="center", va="top",
                fontsize=general["tick_label_font"])

    # Y-axis annotations.
    y_cfg = axis_colors_cfg.get("y", {}).get(axis_labels_cfg.get("y", ""), {})
    if y_cfg:
        ax.text(x.min() - label_offset * x_range, y[-1], y_cfg["start"],
                color=y_cfg["colors"][0], ha="right", va="center",
                fontsize=general["tick_label_font"])
        ax.text(x.min() - label_offset * x_range, y[0], y_cfg["end"],
                color=y_cfg["colors"][1], ha="right", va="bottom",
                fontsize=general["tick_label_font"])

    ax.set_aspect('auto')
    plt.tight_layout()
    plt.grid(showGrid)
    plt.show()

# Example usage for the two hypotheses:
plot_smoothness_tradeoff("sBBT_duration")



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
    Plots histograms of Fisher z-transformed median Spearman correlations for each subject,
    overlaying non_dominant and dominant hands in different colors. Reports median, IQR,
    and Wilcoxon signed-rank test result by hand, and returns these statistics as a dictionary.

    Parameters:
        heatmap_results (dict): Heatmap results containing medians with keys 'non_dominant' and 'dominant'.
        show_value_on_legend (bool): If True, numerical values are shown in the legend.

    Returns:
        dict: Dictionary containing statistics for both non dominant and dominant hands.
              Example:
              {
                "non_dominant": {"median": ..., "iqr": ..., "wilcoxon_stat": ..., "p_value": ... },
                "dominant": {"median": ..., "iqr": ..., "wilcoxon_stat": ..., "p_value": ... }
              }
    """
    import matplotlib.pyplot as plt
    # Extract original correlation medians.
    non_dominant_row_medians = list(heatmap_results['medians']["non_dominant"]["row_medians"].values())
    dominant_row_medians = list(heatmap_results['medians']["dominant"]["row_medians"].values())

    # Apply Fisher's z-transform. Clip values to avoid infinite results.
    non_dominant_fisher = np.arctanh(np.clip(non_dominant_row_medians, -0.999999, 0.999999))
    dominant_fisher = np.arctanh(np.clip(dominant_row_medians, -0.999999, 0.999999))

    # Calculate statistics for non dominant hand (Fisher z scale).
    median_non_dominant = np.median(non_dominant_fisher)
    iqr_non_dominant = np.percentile(non_dominant_fisher, 75) - np.percentile(non_dominant_fisher, 25)
    q1_non_dominant = np.percentile(non_dominant_fisher, 25)
    q3_non_dominant = np.percentile(non_dominant_fisher, 75)

    # Calculate statistics for dominant hand (Fisher z scale).
    median_dominant = np.median(dominant_fisher)
    iqr_dominant = np.percentile(dominant_fisher, 75) - np.percentile(dominant_fisher, 25)
    q1_dominant = np.percentile(dominant_fisher, 25)
    q3_dominant = np.percentile(dominant_fisher, 75)

    # Perform Wilcoxon signed-rank test on the Fisher transformed correlations.
    stat_non_dominant, p_value_non_dominant = wilcoxon(non_dominant_fisher)
    stat_dominant, p_value_dominant = wilcoxon(dominant_fisher)

    # Define labels based on option.
    label_median_non = f"Median non dominant: {median_non_dominant:.2f}" if show_value_on_legend else "Median non dominant"
    label_median_dom = f"Median dominant: {median_dominant:.2f}" if show_value_on_legend else "Median dominant"
    label_iqr_non = f"IQR non dominant: {iqr_non_dominant:.2f}" if show_value_on_legend else "IQR non dominant"
    label_iqr_dom = f"IQR dominant: {iqr_dominant:.2f}" if show_value_on_legend else "IQR dominant"

    # Plot histogram for Fisher z-transformed median Spearman correlations.
    plt.figure(figsize=(8, 6))
    plt.hist(non_dominant_fisher, bins=15, color='orange', alpha=0.7, edgecolor='black', label='Non dominant Hand')
    plt.hist(dominant_fisher, bins=15, color='blue', alpha=0.7, edgecolor='black', label='Dominant Hand')
    plt.axvline(median_non_dominant, color='orange', linestyle='--', label=label_median_non)
    plt.axvline(median_dominant, color='blue', linestyle='--', label=label_median_dom)
    plt.axvspan(q1_non_dominant, q3_non_dominant, color='orange', alpha=0.2, label=label_iqr_non)
    plt.axvspan(q1_dominant, q3_dominant, color='blue', alpha=0.2, label=label_iqr_dom)
    plt.title("Histogram of Fisher z-transformed Median Spearman Correlations by Hand")
    plt.xlabel("Fisher z-transformed Correlation", fontsize=24)
    plt.ylabel("Frequency", fontsize=24)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()

    print(f"Non-dominant Hand (Fisher z): Median = {median_non_dominant:.2f}, IQR = {iqr_non_dominant:.2f}, Wilcoxon stat = {stat_non_dominant:.2f}, p-value = {p_value_non_dominant:.4f}")
    print(f"Dominant Hand (Fisher z): Median = {median_dominant:.2f}, IQR = {iqr_dominant:.2f}, Wilcoxon stat = {stat_dominant:.2f}, p-value = {p_value_dominant:.4f}")

    # Return statistics as a dictionary.
    return {
        "non_dominant": {
            "median": median_non_dominant,
            "iqr": iqr_non_dominant,
            "wilcoxon_stat": stat_non_dominant,
            "p_value": p_value_non_dominant
        },
        "dominant": {
            "median": median_dominant,
            "iqr": iqr_dominant,
            "wilcoxon_stat": stat_dominant,
            "p_value": p_value_dominant
        }
    }

## -------------------------------------------------------------------------------------------------------------------
saved_heatmaps = {}
saved_medians = {}

for var in ["TW_LDLJ", "TW_sparc", "ballistic_LDLJ", "ballistic_sparc", "correction_LDLJ", "correction_sparc"]:
    indep_var = var
    for dep_var in ['durations', 'distance', 'MotorAcuity']:
        print(f"Processing {indep_var} vs {dep_var}")
        
        # Example call: using independent variable and dependent variable for a specific subject/hand
        correlation_results = plot_indep_dep_scatter(
            updated_metrics_acorss_phases, updated_metrics_zscore,
            subject='07/22/HW', hand='non_dominant',
            indep_var=indep_var, dep_var=dep_var, cols=4
        )
        
        # Example call: using independent variable and dependent variable for both hands
        heatmap_results = heatmap_spearman_correlation_all_subjects(
            updated_metrics_acorss_phases, updated_metrics_zscore, hand="both",
            indep_var=indep_var, dep_var=dep_var,
            simplified=True, return_medians=True, overlay_median=True
        )
        
        # Example call: plot histogram with statistics and capture the median result
        median_of_median = plot_histogram_spearman_corr_with_stats_reach_indices_by_subject(
            heatmap_results, show_value_on_legend=True
        )
        
        # Save the outputs for later use
        saved_heatmaps[(indep_var, dep_var)] = heatmap_results
        saved_medians[(indep_var, dep_var)] = median_of_median

# Convert saved_medians dictionary to a list of records
records = []
for (indep_var, dep_var), medians in saved_medians.items():
    record = {
        "Independent Variable": indep_var,
        "Dependent Variable": dep_var
    }
    # Flatten the medians dictionary (which holds summary statistics for non_dominant and dominant)
    for hand in medians:
        for stat, value in medians[hand].items():
            record[f"{hand}_{stat}"] = value
    records.append(record)

# Create DataFrame and display it
df_saved_medians = pd.DataFrame(records)
print(df_saved_medians)

# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# Pairwise comparisons between phases (TW vs Ballistic, TW vs Correction, Ballistic vs Correction)
# Each phase vs zero
# Fisher Z-transform for statistical tests
# FDR correction for multiple comparisons

def plot_grouped_median_correlations(saved_heatmaps, hand='non_dominant', overlay_points=True, figuresize=(10, 5), metric='ldlj'):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    axis_labels = dict(
        duration="Duration (s)",
        distance="Error (mm)",
        correlation="Correlation"
    )

    sig_levels=[(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")]

    prefix = metric.lower()
    phases = ['TW', 'Ballistic', 'Correction']
    metrics = ['durations', 'distance', 'MotorAcuity']

    data_dicts = {}
    for m in metrics:
        data_dicts[m] = {}
        for p in phases:
            key = (f"{p.lower() if p!='TW' else p}_{prefix.upper()}", m)
            data_dicts[m][p] = saved_heatmaps[key]['medians'][hand]['row_medians']

    def compute_median_and_error(data):
        vals = np.array(list(data.values()))
        median = np.median(vals)
        sem = np.std(vals, ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else 0.0
        return median, sem

    # Use the raw values (no Fisher z-transform)
    phase_vals = {p: np.array(list(data_dicts[m][p].values())) for p in phases}  # dummy initiation

    # Prepare plotting values
    all_vals, all_errs = [], []
    for m in metrics:
        vals, errs = [], []
        for p in phases:
            v, e = compute_median_and_error(data_dicts[m][p])
            vals.append(v)
            errs.append(e)
        all_vals.append(vals)
        all_errs.append(errs)
    all_vals = np.array(all_vals)
    all_errs = np.array(all_errs)

    categories = ["Duration (s)", "Error (mm)", "Motor Acuity"]
    x = np.arange(len(categories))
    width = 0.25
    colors = ['white', 'blue', 'red']

    fig, ax = plt.subplots(figsize=figuresize)

    bar_positions = {}  # store actual x positions of each bar for line plotting

    for i, p in enumerate(phases):
        bar = ax.bar(x + i*width - width, all_vals[:, i], width, yerr=all_errs[:, i], capsize=5,
               color=colors[i], edgecolor='black', label=p)
        if overlay_points:
            for j, m in enumerate(metrics):
                pts = np.array(list(data_dicts[m][p].values()))
                jitter = np.random.uniform(-0.05, 0.05, size=pts.shape)
                ax.scatter(np.full(pts.shape, x[j]+i*width-width)+jitter, pts, color='black', alpha=0.5, zorder=5)
        # Store bar positions for each metric & phase
        for j, m in enumerate(metrics):
            bar_positions[(m,p)] = x[j]+i*width-width

    # Statistical comparisons using the original correlations (no transform)
    pvals = []
    comparisons = []
    for m in metrics:
        phase_vals = {p: np.array(list(data_dicts[m][p].values())) for p in phases}
        # Use phase_vals directly (no transformation)
        for pair in [('TW','Ballistic'), ('TW','Correction'), ('Ballistic','Correction')]:
            p1, p2 = pair
            stat_val, p_val = wilcoxon(phase_vals[p1], phase_vals[p2])
            pvals.append(p_val)
            comparisons.append((m, p1, p2, p_val))
        for phase_name in phases:
            stat_val, p_val = wilcoxon(phase_vals[phase_name], np.zeros_like(phase_vals[phase_name]))
            pvals.append(p_val)
            comparisons.append((m, phase_name, 'Zero', p_val))

    # FDR correction
    _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

    # Annotate significance lines (horizontal)
    y_max = 0.5
    y_step = 0.2
    line_positions = {m: y_max for m in metrics}

    for idx, (m, p1, p2, _) in enumerate(comparisons):
        y = line_positions[m]
        x1 = bar_positions[(m, p1)]
        if p2 == 'Zero':
            bar_height = all_vals[metrics.index(m), phases.index(p1)]
            for thresh, star in sig_levels:
                if pvals_corrected[idx] <= thresh:
                    ax.text(x1, y_max - 0.2, star, ha='center', fontsize=20)
                    break
        else:
            x2 = bar_positions[(m, p2)]
            ax.plot([x1, x2], [y, y], color='black', linewidth=1.2)
            for thresh, star in sig_levels:
                if pvals_corrected[idx] <= thresh:
                    ax.text((x1 + x2) / 2, y + 0.015, star, ha='center', fontsize=20)
                    break
            line_positions[m] += y_step

    # Final touches
    ax.axhline(0, color='lightgrey', linestyle='-', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=14)
    ax.set_ylabel(axis_labels["correlation"], fontsize=14)
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([-1, 0, 1], fontsize=12)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ['Ballistic + Correction' if label == 'TW' else label for label in labels]
    ax.legend(handles, new_labels, fontsize=12, frameon=False, loc='best')
    
    # Add sample size annotation: count distinct subjects
    all_subjects = set()
    for m in metrics:
        for p in phases:
            all_subjects.update(data_dicts[m][p].keys())
    n = len(all_subjects)
    ax.text(0.95, 0.25, f"n = {n} participants", transform=ax.transAxes,
            ha='right', va='top', fontsize=14)
    
    plt.tight_layout()
    plt.show()

    print(f"median of median correlations ({hand}, {metric}):")
    for m in metrics:
        tw_vals = np.array(list(data_dicts[m]['TW'].values()))
        ballistic_vals = np.array(list(data_dicts[m]['Ballistic'].values()))
        correction_vals = np.array(list(data_dicts[m]['Correction'].values()))
        overall_median = np.median(tw_vals)
        print(f"  {m}: {overall_median:.3f}")
        ballistic_median = np.median(ballistic_vals)
        print(f"  {m} ballistic: {ballistic_median:.3f}")
        correction_median = np.median(correction_vals)
        print(f"  {m} correction: {correction_median:.3f}")
    # Print detailed comparison results
    for idx, (m, p1, p2, orig_p) in enumerate(comparisons):
        print(f"Comparison for {m} ({p1} vs {p2}): original p = {orig_p:.3e}, corrected p = {pvals_corrected[idx]:.3f}")
        
plot_grouped_median_correlations(saved_heatmaps, hand='non_dominant', overlay_points=True, figuresize=(10, 5), metric='ldlj')

def plot_grouped_median_correlations(saved_heatmaps, hand='non_dominant', overlay_points=True, figuresize=(10, 5), metric='ldlj'):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    axis_labels = dict(
        duration="Duration (s)",
        distance="Error (mm)",
        correlation="Correlation"
    )

    sig_levels=[(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")]

    prefix = metric
    phases = ['TW', 'Ballistic', 'Correction']
    metrics = ['durations', 'distance', 'MotorAcuity']

    data_dicts = {}
    for m in metrics:
        data_dicts[m] = {}
        for p in phases:
            key = (f"{p.lower() if p!='TW' else p}_{prefix}", m)
            data_dicts[m][p] = saved_heatmaps[key]['medians'][hand]['row_medians']

    def compute_median_and_error(data):
        vals = np.array(list(data.values()))
        median = np.median(vals)
        sem = np.std(vals, ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else 0.0
        return median, sem

    # Prepare plotting values
    all_vals, all_errs = [], []
    for m in metrics:
        vals, errs = [], []
        for p in phases:
            v, e = compute_median_and_error(data_dicts[m][p])
            vals.append(v)
            errs.append(e)
        all_vals.append(vals)
        all_errs.append(errs)
    all_vals = np.array(all_vals)
    all_errs = np.array(all_errs)

    categories = ["Duration (s)", "Error (mm)", "Motor Acuity"]
    x = np.arange(len(categories))
    width = 0.25
    colors = ['white', 'blue', 'red']

    fig, ax = plt.subplots(figsize=figuresize)

    bar_positions = {}  # store actual x positions of each bar for line plotting

    for i, p in enumerate(phases):
        bar = ax.bar(x + i*width - width, all_vals[:, i], width, yerr=all_errs[:, i], capsize=5,
               color=colors[i], edgecolor='black', label=p)
        if overlay_points:
            for j, m in enumerate(metrics):
                pts = np.array(list(data_dicts[m][p].values()))
                jitter = np.random.uniform(-0.05, 0.05, size=pts.shape)
                ax.scatter(np.full(pts.shape, x[j]+i*width-width)+jitter, pts, color='black', alpha=0.5, zorder=5)
        # Store bar positions for each metric & phase
        for j, m in enumerate(metrics):
            bar_positions[(m,p)] = x[j]+i*width-width

    # Statistical comparisons using raw correlations
    pvals = []
    comparisons = []
    for m in metrics:
        phase_vals = {p: np.array(list(data_dicts[m][p].values())) for p in phases}
        # Pairwise
        pairs = [('TW','Ballistic'), ('TW','Correction'), ('Ballistic','Correction')]
        for p1, p2 in pairs:
            stat_val, p_val = wilcoxon(phase_vals[p1], phase_vals[p2])
            pvals.append(p_val)
            comparisons.append((m, p1, p2, p_val))
        # vs zero
        for phase_name in phases:
            stat_val, p_val = wilcoxon(phase_vals[phase_name], np.zeros_like(phase_vals[phase_name]))
            pvals.append(p_val)
            comparisons.append((m, phase_name, 'Zero', p_val))

    # FDR correction
    _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

    # Annotate significance lines (horizontal)
    y_max = 0.6
    y_step = 0.1
    line_positions = {m: y_max for m in metrics}

    for idx, (m, p1, p2, _) in enumerate(comparisons):
        y = line_positions[m]
        x1 = bar_positions[(m,p1)]
        if p2=='Zero':
            bar_height = all_vals[metrics.index(m), phases.index(p1)]
            for thresh, star in sig_levels:
                if pvals_corrected[idx] <= thresh:
                    ax.text(x1, y_max - 0.1, star, ha='center', fontsize=20)
                    break
        else:
            x2 = bar_positions[(m,p2)]
            ax.plot([x1,x2],[y,y], color='black', linewidth=1.2)
            for thresh, star in sig_levels:
                if pvals_corrected[idx] <= thresh:
                    ax.text((x1+x2)/2, y+0.015, star, ha='center', fontsize=20)
                    break
            line_positions[m] += y_step

    # Annotate sample size
    n = len(list(data_dicts[metrics[0]][phases[0]].keys()))
    ax.text(0.95, 0.25, f"n = {n} participants", transform=ax.transAxes, ha='right', va='top', fontsize=14)

    # Final touches
    ax.axhline(0, color='lightgrey', linestyle='-', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=14)
    ax.set_ylabel(axis_labels["correlation"], fontsize=14)
    ax.set_ylim(-1,1)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([-1, 0, 1], fontsize=12)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ['Ballistic + Correction' if label == 'TW' else label for label in labels]
    ax.legend(handles, new_labels, fontsize=12, frameon=False, loc='best')
    plt.tight_layout()
    plt.show()

plot_grouped_median_correlations(saved_heatmaps, hand='non_dominant', overlay_points=True, figuresize=(10, 5), metric='sparc')
# -------------------------------------------------------------------------------------------------------------------
def plot_grouped_median_correlations(saved_heatmaps, hand='non_dominant', overlay_points=True, figuresize=(10, 5), metric='ldlj'):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_rel, shapiro, ttest_1samp
    from statsmodels.stats.multitest import multipletests

    axis_labels = dict(
        duration="Duration (s)",
        distance="Error (mm)",
        correlation="Correlation"
    )

    # Define significance levels; note that we will annotate only if p < 0.05 (i.e., ignore 'ns')
    sig_levels = [(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")]

    prefix = metric
    phases = ['TW', 'Ballistic', 'Correction']
    metrics = ['durations', 'distance', 'MotorAcuity']

    data_dicts = {}
    for m in metrics:
        data_dicts[m] = {}
        for p in phases:
            key = (f"{p.lower() if p != 'TW' else p}_{prefix}", m)
            data_dicts[m][p] = saved_heatmaps[key]['medians'][hand]['row_medians']

    def compute_median_and_error(data):
        vals = np.array(list(data.values()))
        median = np.median(vals)
        sem = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        return median, sem

    # Compute aggregated values for plotting boxplots (no transformation)
    all_vals, all_errs = [], []
    for m in metrics:
        vals, errs = [], []
        for p in phases:
            v, e = compute_median_and_error(data_dicts[m][p])
            vals.append(v)
            errs.append(e)
        all_vals.append(vals)
        all_errs.append(errs)
    all_vals = np.array(all_vals)
    all_errs = np.array(all_errs)

    categories = ["Duration (s)", "Error (mm)", "Motor Acuity"]
    x = np.arange(len(categories))
    width = 0.3
    colors = ['white', '#0047ff', '#ffb800']

    fig, ax = plt.subplots(figsize=figuresize)

    bar_positions = {}  # store actual x positions of each box for line plotting

    # Replace bar plots with box plots.
    for i, p in enumerate(phases):
        for j, m in enumerate(metrics):
            pos = x[j] + i * width - width
            values = np.array(list(data_dicts[m][p].values()))
            ax.boxplot(values, positions=[pos], widths=width * 0.8, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=colors[i], color='black'),
                       medianprops=dict(color='black'))
            if overlay_points:
                jitter = np.random.uniform(-0.05, 0.05, size=values.shape)
                ax.scatter(np.full(values.shape, pos) + jitter, values, color='black', alpha=0.4, zorder=5)
            bar_positions[(m, p)] = pos

    # Statistical comparisons: each phase vs Zero, and Ballistic vs Correction.
    pvals = []
    comparisons = []
    for m in metrics:
        phase_vals = {p: np.array(list(data_dicts[m][p].values())) for p in phases}
        # Each phase vs Zero: check normality first then perform paired t-test.
        for phase_name in phases:
            # Normality check using Shapiro-Wilk test.
            stat_norm, p_norm = shapiro(phase_vals[phase_name])
            # print(f"Normality test for metric '{m}', phase '{phase_name}' vs Zero: Shapiro-Wilk p-value = {p_norm:.3f}")
            t_stat, p_val = ttest_1samp(phase_vals[phase_name], 0)
            pvals.append(p_val)
            comparisons.append((m, phase_name, 'Zero', p_val))
        # Comparison between Ballistic and Correction: check normality for both.
        stat_norm_b, p_norm_b = shapiro(phase_vals['Ballistic'])
        stat_norm_c, p_norm_c = shapiro(phase_vals['Correction'])
        # print(f"Normality test for metric '{m}', phase 'Ballistic': Shapiro-Wilk p-value = {p_norm_b:.3f}")
        # print(f"Normality test for metric '{m}', phase 'Correction': Shapiro-Wilk p-value = {p_norm_c:.3f}")
        t_stat, p_val = ttest_rel(phase_vals['Ballistic'], phase_vals['Correction'])
        pvals.append(p_val)
        comparisons.append((m, 'Ballistic', 'Correction', p_val))

    # FDR correction
    _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

    # Annotate significance lines for each metric only if p < 0.05
    y_max = 0.5
    y_step = 0.2
    line_positions = {m: y_max for m in metrics}

    for idx, (m, p1, p2, _) in enumerate(comparisons):
        # Only annotate if corrected p-value is below 0.05
        if pvals_corrected[idx] < 0.05:
            y = line_positions[m]
            x1 = bar_positions[(m, p1)]
            if p2 == 'Zero':
                # Annotate significance versus zero at the bar's x-position
                for thresh, star in sig_levels:
                    if pvals_corrected[idx] <= thresh and thresh < 1.0:
                        ax.text(x1, y_max - 0.2, star, ha='center', va='bottom', fontsize=20)
                        break
            elif p1 == 'Ballistic' and p2 == 'Correction':
                # Between Ballistic and Correction: draw a line and add the significance text above it
                x2 = bar_positions[(m, p2)]
                ax.plot([x1, x2], [y, y], color='black', linewidth=1.2)
                for thresh, star in sig_levels:
                    if pvals_corrected[idx] <= thresh and thresh < 1.0:
                        ax.text((x1 + x2) / 2, y-0.08, star, ha='center', va='bottom', fontsize=20)
                        break
                # Increase y position for further annotations of the same metric
                line_positions[m] += y_step

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=phases[i]) for i in range(len(phases))]
    new_labels = ['Ballistic + Correction' if label == 'TW' else label for label in [patch.get_label() for patch in legend_elements]]
    ax.legend(handles=legend_elements, labels=new_labels, fontsize=14, frameon=False, loc='upper right', bbox_to_anchor=(1, 1.1))

    # Final touches
    ax.axhline(0, color='lightgrey', linestyle='-', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=14)
    ax.set_ylabel(axis_labels["correlation"], fontsize=14)
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([-1, 0, 1], fontsize=14)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation for sample size
    all_subjects = set()
    for m in metrics:
        for p in phases:
            all_subjects.update(data_dicts[m][p].keys())
    n = len(all_subjects)
    ax.text(0.95, 0.25, f"n = {n} participants", transform=ax.transAxes,
            ha='right', va='top', fontsize=14)

    plt.tight_layout()
    plt.show()

    print(f"median of median correlations ({hand}, {metric}):")
    for m in metrics:
        tw_vals = np.array(list(data_dicts[m]['TW'].values()))
        ballistic_vals = np.array(list(data_dicts[m]['Ballistic'].values()))
        correction_vals = np.array(list(data_dicts[m]['Correction'].values()))
        overall_median = np.median(tw_vals)
        print(f"  {m}: {overall_median:.3f}, IQR = {np.percentile(tw_vals, 75) - np.percentile(tw_vals, 25):.3f}")
        ballistic_median = np.median(ballistic_vals)
        print(f"  {m} ballistic: {ballistic_median:.3f}, IQR = {np.percentile(ballistic_vals, 75) - np.percentile(ballistic_vals, 25):.3f}")
        correction_median = np.median(correction_vals)
        print(f"  {m} correction: {correction_median:.3f}, IQR = {np.percentile(correction_vals, 75) - np.percentile(correction_vals, 25):.3f}")
    for idx, (m, p1, p2, orig_p) in enumerate(comparisons):
        # Retrieve the values used in the comparison
        values1 = np.array(list(data_dicts[m][p1].values()))
        if p2 == 'Zero':
            values2 = np.zeros_like(values1)
        else:
            values2 = np.array(list(data_dicts[m][p2].values()))
        n = len(values1)
        # Recompute the paired t-test statistic.
        t_stat, _ = ttest_rel(values1, values2)
        # Cohen's d defined as t_stat divided by the square root of n (for paired samples)
        d = t_stat / np.sqrt(n)
        print(f"Comparison for {m} ({p1} vs {p2}): original p = {orig_p:.3e}, corrected p = {pvals_corrected[idx]}, t = {t_stat:.3f}, d = {d:.3f}")

plot_grouped_median_correlations(saved_heatmaps, hand='non_dominant', overlay_points=True, figuresize=(8, 4), metric='LDLJ')
plot_grouped_median_correlations(saved_heatmaps, hand='dominant', overlay_points=True, figuresize=(8, 4), metric='LDLJ')
plot_grouped_median_correlations(saved_heatmaps, hand='non_dominant', overlay_points=True, figuresize=(8, 4), metric='sparc')
plot_grouped_median_correlations(saved_heatmaps, hand='dominant', overlay_points=True, figuresize=(8, 4), metric='sparc')


# -------------------------------------------------------------------------------------------------------------------
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

def compare_hands(saved_heatmaps, metric='ldlj'):
    phases = ['TW', 'Ballistic', 'Correction']
    metrics = ['durations', 'distance', 'MotorAcuity']

    # Collect data for both hands
    data_dicts = {'non_dominant': {}, 'dominant': {}}
    for hand in ['non_dominant', 'dominant']:
        data_dicts[hand] = {}
        for m in metrics:
            data_dicts[hand][m] = {}
            for p in phases:
                key = (f"{p.lower() if p != 'TW' else p}_{metric}", m)
                data_dicts[hand][m][p] = saved_heatmaps[key]['medians'][hand]['row_medians']

    # Compare each phase-metric pair between hands
    comparisons = []
    pvals = []
    for m in metrics:
        for p in phases:
            vals_nd = np.array(list(data_dicts['non_dominant'][m][p].values()))
            vals_d = np.array(list(data_dicts['dominant'][m][p].values()))
            # Ensure the arrays are paired correctly (matching subjects)
            common_subjects = sorted(set(data_dicts['non_dominant'][m][p].keys()) & 
                                     set(data_dicts['dominant'][m][p].keys()))
            vals_nd_paired = np.array([data_dicts['non_dominant'][m][p][s] for s in common_subjects])
            vals_d_paired = np.array([data_dicts['dominant'][m][p][s] for s in common_subjects])
            
            # Paired test
            stat, p_val = wilcoxon(vals_nd_paired, vals_d_paired)
            comparisons.append((m, p, 'non_dominant vs dominant', p_val))
            pvals.append(p_val)

    # FDR correction for multiple comparisons
    _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

    # Print results
    for idx, (m, phase, comp, orig_p) in enumerate(comparisons):
        print(f"{m} {phase} ({comp}): original p = {orig_p:.3e}, corrected p = {pvals_corrected[idx]:.3f}")

# Example usage:
compare_hands(saved_heatmaps, metric='LDLJ')
compare_hands(saved_heatmaps, metric='sparc')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

def plot_hand_comparisons(saved_heatmaps, metric='ldlj'):
    phases = ['TW', 'Ballistic', 'Correction']
    metrics = ['durations', 'distance', 'MotorAcuity']
    
    data_dicts = {'non_dominant': {}, 'dominant': {}}
    for hand in ['non_dominant', 'dominant']:
        data_dicts[hand] = {}
        for m in metrics:
            data_dicts[hand][m] = {}
            for p in phases:
                key = (f"{p.lower() if p != 'TW' else p}_{metric}", m)
                data_dicts[hand][m][p] = saved_heatmaps[key]['medians'][hand]['row_medians']

    comparisons = []
    pvals = []
    paired_data = []

    for m in metrics:
        for p in phases:
            common_subjects = sorted(set(data_dicts['non_dominant'][m][p].keys()) & 
                                     set(data_dicts['dominant'][m][p].keys()))
            vals_nd = np.array([data_dicts['non_dominant'][m][p][s] for s in common_subjects])
            vals_d = np.array([data_dicts['dominant'][m][p][s] for s in common_subjects])
            
            stat, p_val = wilcoxon(vals_nd, vals_d)
            comparisons.append((m, p, 'non_dominant vs dominant', p_val))
            pvals.append(p_val)
            
            paired_data.append((m, p, vals_nd, vals_d))

    # FDR correction
    _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

    # Plotting
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(len(metrics), len(phases), figsize=(15, 10), sharey='row')
    
    for i, m in enumerate(metrics):
        for j, p in enumerate(phases):
            ax = axes[i, j] if len(metrics) > 1 else axes[j]
            vals_nd, vals_d = None, None
            for idx, (m_tmp, p_tmp, nd, d) in enumerate(paired_data):
                if m_tmp == m and p_tmp == p:
                    vals_nd, vals_d = nd, d
                    p_val_corr = pvals_corrected[idx]
                    break
            
            sns.boxplot(data=[vals_nd, vals_d], ax=ax)
            sns.swarmplot(data=[vals_nd, vals_d], color=".25", ax=ax)
            ax.set_xticklabels(['Non-dominant', 'Dominant'])
            ax.set_title(f'{m} - {p}\nFDR p = {p_val_corr:.3f}')
            ax.set_ylabel('Value')
    
    plt.tight_layout()
    plt.show()

# Usage
plot_hand_comparisons(saved_heatmaps, metric='LDLJ')







# -------------------------------------------------------------------------------------------------------------------

def calculate_combined_median_of_reach_indices(updated_metrics, updated_metrics_zscore):
    """
    Calculates the median values for each reach_index for two sets of metrics:
      • Raw metrics from updated_metrics for keys:
        'durations', 'distance', 'TW_LDLJ', 'TW_sparc',
        'ballistic_LDLJ', 'ballistic_sparc', 'correction_LDLJ', 'correction_sparc'
      • Z-scored metrics from updated_metrics_zscore for keys:
        'durations', 'distance', 'zscore_durations', 'zscore_distance', 'MotorAcuity'
    
    Returns:
        dict: A dictionary with structure:
          {
            "raw": { subject: { hand: { metric_name: [ {"reach_index": i, "median_value": v}, ... ] } } },
            "zscore": { subject: { hand: { metric_name: [ {"reach_index": i, "median_value": v}, ... ] } } }
          }
    """

    def compute_raw_medians(data):
        desired_metrics = [
            'durations', 
            'distance', 
            'TW_LDLJ', 
            'TW_sparc', 
            'ballistic_LDLJ', 
            'ballistic_sparc', 
            'correction_LDLJ', 
            'correction_sparc'
        ]
        median_raw = {}
        for subject, hands_data in data.items():
            median_raw[subject] = {}
            for hand, metrics in hands_data.items():
                median_raw[subject][hand] = {}
                for key in desired_metrics:
                    if key in metrics:
                        median_list = []
                        trials = metrics[key].keys()
                        # Assume each trial contains a list of 16 reach indices.
                        for reach_index in range(16):
                            values = []
                            for trial in trials:
                                trial_vals = np.array(metrics[key][trial])
                                if reach_index < len(trial_vals):
                                    values.append(trial_vals[reach_index])
                            values = np.array(values)
                            valid = values[~np.isnan(values)]
                            median_val = np.median(valid) if valid.size > 0 else np.nan
                            median_list.append({"reach_index": reach_index, "median_value": median_val})
                        median_raw[subject][hand][key] = median_list
        return median_raw

    def compute_zscore_medians(data):
        desired_metrics = [
            'durations', 
            'distance', 
            'zscore_durations', 
            'zscore_distance',
            'MotorAcuity'
        ]
        median_zscore = {}
        for subject, hands_data in data.items():
            median_zscore[subject] = {}
            for hand, metrics in hands_data.items():
                median_zscore[subject][hand] = {}
                for key in desired_metrics:
                    if key in metrics:
                        median_list = []
                        # In updated_metrics_zscore, keys are reach indices.
                        for reach_index, values in metrics[key].items():
                            values = np.array(values)
                            valid = values[~np.isnan(values)]
                            median_val = np.median(valid) if valid.size > 0 else np.nan
                            # Ensure reach_index is an integer.
                            actual_index = int(reach_index) if not isinstance(reach_index, int) else reach_index
                            median_list.append({"reach_index": actual_index, "median_value": median_val})
                        median_list = sorted(median_list, key=lambda d: d["reach_index"])
                        median_zscore[subject][hand][key] = median_list
        return median_zscore

    return {
        "raw": compute_raw_medians(updated_metrics),
        "zscore": compute_zscore_medians(updated_metrics_zscore)
    }

all_median_stats = calculate_combined_median_of_reach_indices(updated_metrics_acorss_phases, updated_metrics_zscore)

def plot_median_scatter_full(all_median_stats, subject, hand, indep_var, dep_var, config=plot_config_summary, marker_style='s'):
    """
    Creates a scatter plot between median values of an independent variable and a dependent variable
    for a given subject and hand, in the same visual style as plot_trials_mean_median_of_reach_indices.
    Each reach index gets a specific color from placement_location_colors. Includes Spearman correlation,
    axis annotations, and optional placement icon.

    Parameters:
        all_median_stats (dict): Nested dictionary with median statistics per subject, hand, and metric.
        subject (str): Subject identifier (e.g., "07/22/HW").
        hand (str): Hand identifier (e.g., "non_dominant" or "dominant").
        indep_var (str): Key for the independent variable (e.g., "TW_LDLJ" or "MotorAcuity").
        dep_var (str): Key for the dependent variable (e.g., "durations", "MotorAcuity", etc).
        config (dict): Plot configuration dictionary.
        marker_style (str): Marker style for points (default: 's' for square).

    Returns:
        tuple: Spearman correlation and p-value.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import spearmanr
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import matplotlib.image as mpimg

    # Select median data source:
    # If the variable is MotorAcuity, use the zscore version; otherwise use raw.
    if indep_var == "MotorAcuity":
        print("Using zscore data for MotorAcuity as independent variable.")
    else:
        median_data_x = all_median_stats["raw"][subject][hand][indep_var]

    if dep_var == "MotorAcuity":
        median_data_y = all_median_stats["zscore"][subject][hand][dep_var]
    else:
        median_data_y = all_median_stats["raw"][subject][hand][dep_var]

    # Configuration values
    general_cfg = config.get("general", {})
    axis_labels = config.get("axis_labels", {})
    figsize = (6, 4)
    axis_label_font = general_cfg.get("axis_label_font", 14)
    tick_label_font = general_cfg.get("tick_label_font", 14)
    showGrid = general_cfg.get("showGrid", False)
    tick_direction = general_cfg.get("tick_direction", "out")
    label_offset = general_cfg.get("label_offset", 4)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    x_values = []
    y_values = []

    # Plot each reach index in a loop
    for reach_index in range(16):
        # Get x from appropriate data source
        x_data = next((d["median_value"] for d in median_data_x if d["reach_index"] == reach_index), np.nan)
        y_data = next((d["median_value"] for d in median_data_y if d["reach_index"] == reach_index), np.nan)
        if not np.isnan(x_data) and not np.isnan(y_data):
            x_values.append(x_data)
            y_values.append(y_data)
            color = placement_location_colors[reach_index]
            ax.scatter(x_data, y_data, facecolors=color, edgecolors=color, s=100,
                       alpha=1.0, marker=marker_style, zorder=5)

    # Calculate Spearman correlation
    if len(x_values) > 1 and len(y_values) > 1:
        corr, p_val = spearmanr(x_values, y_values)
    else:
        corr, p_val = np.nan, np.nan

    # Axis labels: Use config value if available, otherwise default to key names.
    if indep_var.upper().endswith("LDLJ"):
        xlabel = f"{indep_var[:-5]} (LDLJ)"
        x_cfg = {"start": "unsmoothness", "end": "smoothness", "colors": ["red", "green"]}
    else:
        xlabel = axis_labels.get(indep_var, f"{indep_var}")
        x_cfg = config.get("axis_colors", {}).get("x", {}).get(xlabel, None)

    if dep_var.lower() == "durations":
        ylabel = "Duration (s)"
        y_cfg = {"start": "fast", "end": "slow", "colors": ["green", "red"]}
    else:
        ylabel = axis_labels.get(dep_var, f"{dep_var.capitalize()}")
        y_cfg = config.get("axis_colors", {}).get("y", {}).get(ylabel, None)

    ax.set_xlabel(xlabel, fontsize=axis_label_font)
    ax.set_ylabel(ylabel, fontsize=axis_label_font)

    # Axis annotations
    if x_cfg:
        ax.annotate(x_cfg["start"], xy=(0, -label_offset-0.1), xycoords='axes fraction',
                    fontsize=tick_label_font, ha="left", va="top", color=x_cfg["colors"][0])
        ax.annotate(x_cfg["end"], xy=(1, -label_offset-0.1), xycoords='axes fraction',
                    fontsize=tick_label_font, ha="right", va="top", color=x_cfg["colors"][-1])
    if y_cfg:
        ax.annotate(y_cfg["end"], xy=(-label_offset-0.1, 1-0.07), xycoords='axes fraction',
                    fontsize=tick_label_font, ha="right", va="top", color=y_cfg["colors"][-1])
        ax.annotate(y_cfg["start"], xy=(-label_offset-0.1, 0+0.07), xycoords='axes fraction',
                    fontsize=tick_label_font, ha="right", va="bottom", color=y_cfg["colors"][0])

    # Set tick parameters
    ax.tick_params(axis='both', which='both', labelsize=tick_label_font, direction=tick_direction)

    # Grid and spines
    ax.grid(showGrid)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add placement location icon
    try:
        icon_img = mpimg.imread('/Users/yilinwu/Desktop/Thesis/PlacementLocationIcon_RBOX.png')
        imagebox = OffsetImage(icon_img, zoom=0.2)
        ab = AnnotationBbox(imagebox, (1.2, 0.15), xycoords='axes fraction', frameon=False)
        ax.add_artist(ab)
    except Exception as e:
        print("Error loading icon image:", e)

    # Annotate Spearman correlation and sample size on the plot
    ax.text(1.1, 0.4, f"ρ = {corr:.2f}", transform=ax.transAxes, fontsize=tick_label_font)
    ax.text(1.1, 0.3, f"n = {len(x_values)} Locations", transform=ax.transAxes, fontsize=tick_label_font)

    plt.tight_layout()
    plt.show()

    print(f"Subject: {subject}, Hand: {hand}, Indep: {indep_var}, Dep: {dep_var}\n"
          f"Spearman Corr: {corr:.2f}, P-value: {p_val:.3f}")

    return corr, p_val
plot_median_scatter_full(all_median_stats, subject="07/22/HW", hand="non_dominant", indep_var="TW_sparc", dep_var="MotorAcuity", marker_style='o')

for subj in all_median_stats["raw"].keys():
    plot_median_scatter_full(all_median_stats, subject=subj, hand="non_dominant", indep_var="TW_sparc", dep_var="durations", marker_style='o')

for subj in all_median_stats["raw"].keys():
    plot_median_scatter_full(all_median_stats, subject=subj, hand="non_dominant", indep_var="ballistic_sparc", dep_var="durations", marker_style='o')

for subj in all_median_stats["raw"].keys():
    plot_median_scatter_full(all_median_stats, subject=subj, hand="non_dominant", indep_var="correction_sparc", dep_var="durations", marker_style='o')






def plot_median_scatter_overlay(all_median_stats, subjects, hand, indep_var, dep_var, config=plot_config_summary, marker_style='s'):
    """
    Creates an overlay scatter plot of median values for the specified independent and dependent variables
    from multiple subjects. Each reach location is plotted with a pre-specified color from placement_location_colors.
    Computes the overall Spearman correlation across all subjects.

    Parameters:
        all_median_stats (dict): Nested dictionary with median statistics per subject, hand, and metric.
        subjects (list): List of subject identifiers to include.
        hand (str): Hand identifier (e.g., "non_dominant" or "dominant").
        indep_var (str): Key for the independent variable (e.g., "TW_LDLJ" or "MotorAcuity").
        dep_var (str): Key for the dependent variable (e.g., "durations", "MotorAcuity", etc).
        config (dict): Plot configuration dictionary.
        marker_style (str): Marker style for points (default: 's' for square).

    Returns:
        tuple: Overall Spearman correlation and p-value across all subjects.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import spearmanr
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import matplotlib.image as mpimg

    # Configuration values
    general_cfg = config.get("general", {})
    axis_labels = config.get("axis_labels", {})
    figsize = (10, 8)
    axis_label_font = general_cfg.get("axis_label_font", 14)
    tick_label_font = general_cfg.get("tick_label_font", 14)
    showGrid = general_cfg.get("showGrid", False)
    tick_direction = general_cfg.get("tick_direction", "out")
    label_offset = general_cfg.get("label_offset", 4)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    all_x_values = []
    all_y_values = []

    # Loop over all subjects.
    for subject in subjects:
        # Determine median data sources based on variable type.
        if indep_var == "MotorAcuity":
            print(f"Subject {subject}: Using zscore data for MotorAcuity as independent variable.")
        else:
            median_data_x = all_median_stats["raw"][subject][hand][indep_var]
        if dep_var == "MotorAcuity":
            median_data_y = all_median_stats["zscore"][subject][hand][dep_var]
        else:
            median_data_y = all_median_stats["raw"][subject][hand][dep_var]

        # Plot each reach index with its location-specific color.
        subject_plotted = False  # label a subject only once in the legend.
        for reach_index in range(16):
            x_data = next((d["median_value"] for d in median_data_x if d["reach_index"] == reach_index), np.nan)
            y_data = next((d["median_value"] for d in median_data_y if d["reach_index"] == reach_index), np.nan)
            if not np.isnan(x_data) and not np.isnan(y_data):
                all_x_values.append(x_data)
                all_y_values.append(y_data)
                # Use the location-specific color.
                color = placement_location_colors[reach_index]
                ax.scatter(x_data, y_data,
                           facecolors=color, edgecolors=color,
                           s=100, alpha=0.8, marker=marker_style, zorder=5,
                           label=subject if not subject_plotted else None)
                subject_plotted = True

    # Compute overall Spearman correlation across subjects
    if len(all_x_values) > 1 and len(all_y_values) > 1:
        corr, p_val = spearmanr(all_x_values, all_y_values)
    else:
        corr, p_val = np.nan, np.nan

    # Set axis labels using config if available
    if indep_var.upper().endswith("LDLJ"):
        xlabel = f"{indep_var[:-5]} (LDLJ)"
        x_cfg = {"start": "unsmoothness", "end": "smoothness", "colors": ["red", "green"]}
    else:
        xlabel = axis_labels.get(indep_var, indep_var)
        x_cfg = config.get("axis_colors", {}).get("x", {}).get(xlabel, None)

    if dep_var.lower() == "durations":
        ylabel = "Duration (s)"
        y_cfg = {"start": "fast", "end": "slow", "colors": ["green", "red"]}
    else:
        ylabel = axis_labels.get(dep_var, dep_var.capitalize())
        y_cfg = config.get("axis_colors", {}).get("y", {}).get(ylabel, None)

    ax.set_xlabel(xlabel, fontsize=axis_label_font)
    ax.set_ylabel(ylabel, fontsize=axis_label_font)

    # Add axis annotations if config available
    if x_cfg:
        ax.annotate(x_cfg["start"], xy=(0, -label_offset-0.1), xycoords='axes fraction',
                    fontsize=tick_label_font, ha="left", va="top", color=x_cfg["colors"][0])
        ax.annotate(x_cfg["end"], xy=(1, -label_offset-0.1), xycoords='axes fraction',
                    fontsize=tick_label_font, ha="right", va="top", color=x_cfg["colors"][-1])
    if y_cfg:
        ax.annotate(y_cfg["end"], xy=(-label_offset-0.1, 1-0.07), xycoords='axes fraction',
                    fontsize=tick_label_font, ha="right", va="top", color=y_cfg["colors"][-1])
        ax.annotate(y_cfg["start"], xy=(-label_offset-0.1, 0+0.07), xycoords='axes fraction',
                    fontsize=tick_label_font, ha="right", va="bottom", color=y_cfg["colors"][0])

    ax.tick_params(axis='both', which='both', labelsize=tick_label_font, direction=tick_direction)
    ax.grid(showGrid)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add placement location icon only once (if available)
    try:
        icon_img = mpimg.imread('/Users/yilinwu/Desktop/Thesis/PlacementLocationIcon_RBOX.png')
        imagebox = OffsetImage(icon_img, zoom=0.2)
        ab = AnnotationBbox(imagebox, (1.2, 0.15), xycoords='axes fraction', frameon=False)
        ax.add_artist(ab)
    except Exception as e:
        print("Error loading icon image:", e)

    # # Annotate overall Spearman correlation and sample size on the plot
    # ax.text(1.1, 0.4, f"ρ = {corr:.2f}", transform=ax.transAxes, fontsize=tick_label_font)
    # ax.text(1.1, 0.3, f"n = {len(all_x_values)} Locations", transform=ax.transAxes, fontsize=tick_label_font)

    # plt.legend(title="Subjects")
    plt.tight_layout()
    plt.show()

    print(f"Overlay Plot for subjects {subjects}, Hand: {hand}, Indep: {indep_var}, Dep: {dep_var}\n"
          f"Overall Spearman Corr: {corr:.2f}, P-value: {p_val:.3f}")
    return corr, p_val

# Example usage:
plot_median_scatter_overlay(all_median_stats, subjects=All_dates, hand="non_dominant",
                            indep_var="TW_sparc", dep_var="durations", marker_style='o')

plot_median_scatter_overlay(all_median_stats, subjects=All_dates, hand="non_dominant",
                            indep_var="ballistic_sparc", dep_var="durations", marker_style='o')
plot_median_scatter_overlay(all_median_stats, subjects=All_dates, hand="non_dominant",
                            indep_var="correction_sparc", dep_var="durations", marker_style='o')


# def compute_spearman_all_median_stats(all_median_stats):
#     """
#     For each independent variable (among TW_LDLJ, TW_sparc, ballistic_LDLJ, ballistic_sparc,
#     correction_LDLJ, correction_sparc) and for each dependent variable (durations, distance, MotorAcuity),
#     compute the Spearman correlation (and p-value) between their median values across reach indices.
    
#     Note:
#       • Independent variables and dependent variables (except MotorAcuity) are obtained from the 'raw' data.
#       • For MotorAcuity, the dependent value is obtained from the 'zscore' data.
      
#     Returns:
#       results: dict keyed by (indep_var, dep_var) with a list of results for each subject and hand.
#                Each result is a dict with keys "subject", "hand", "correlation", and "p_value".
#     """

#     independent_vars = [
#         "TW_LDLJ", "TW_sparc",
#         "ballistic_LDLJ", "ballistic_sparc",
#         "correction_LDLJ", "correction_sparc"
#     ]
#     dependent_vars = ['durations', 'distance', 'MotorAcuity']

#     results = {}

#     # Loop over independent and dependent variables
#     for indep_var in independent_vars:
#         for dep_var in dependent_vars:
#             key_pair = (indep_var, dep_var)
#             results[key_pair] = []

#             # Loop over subjects in the raw data (assumed to be common between raw and zscore)
#             for subject in all_median_stats["raw"]:
#                 # Loop over hands for the subject
#                 for hand in all_median_stats["raw"][subject]:
#                     # Get independent variable from raw data
#                     if indep_var in all_median_stats["raw"][subject][hand]:
#                         indep_list = all_median_stats["raw"][subject][hand][indep_var]
#                     else:
#                         continue

#                     # For dependent variable, use zscore if it's MotorAcuity, else raw.
#                     if dep_var == "MotorAcuity":
#                         if dep_var in all_median_stats["zscore"][subject][hand]:
#                             dep_list = all_median_stats["zscore"][subject][hand][dep_var]
#                         else:
#                             continue
#                     else:
#                         if dep_var in all_median_stats["raw"][subject][hand]:
#                             dep_list = all_median_stats["raw"][subject][hand][dep_var]
#                         else:
#                             continue

#                     # # Sort by reach_index to align data
#                     # indep_sorted = sorted(indep_list, key=lambda d: d["reach_index"])
#                     # dep_sorted   = sorted(dep_list, key=lambda d: d["reach_index"])

#                     # x_vals = []
#                     # y_vals = []
#                     # for d_indep, d_dep in zip(indep_sorted, dep_sorted):
#                     #     v1 = d_indep.get("median_value")
#                     #     v2 = d_dep.get("median_value")
#                     #     if v1 is not None and v2 is not None:
#                     #         if not (np.isnan(v1) or np.isnan(v2)):
#                     #             x_vals.append(v1)
#                     #             y_vals.append(v2)

#                     # Build dicts keyed by reach_index
#                     indep_dict = {d["reach_index"]: d.get("median_value") for d in indep_list}
#                     dep_dict   = {d["reach_index"]: d.get("median_value") for d in dep_list}

#                     x_vals, y_vals = [], []
#                     for idx in set(indep_dict.keys()) & set(dep_dict.keys()):
#                         v1, v2 = indep_dict[idx], dep_dict[idx]
#                         if v1 is not None and v2 is not None:
#                             if not (np.isnan(v1) or np.isnan(v2)):
#                                 x_vals.append(v1)
#                                 y_vals.append(v2)                                

#                     if len(x_vals) > 1 and len(y_vals) > 1:
#                         corr, p_val = spearmanr(x_vals, y_vals)
#                         results[key_pair].append({
#                             "subject": subject,
#                             "hand": hand,
#                             "correlation": corr,
#                             "p_value": p_val
#                         })
#     return results
# spearman_results = compute_spearman_all_median_stats(all_median_stats)

def compute_pearson_all_median_stats(all_median_stats):
    """
    For each independent variable (among TW_LDLJ, TW_sparc, ballistic_LDLJ, ballistic_sparc,
    correction_LDLJ, correction_sparc) and for each dependent variable (durations, distance, MotorAcuity),
    compute the Pearson correlation (and p-value) between their median values across reach indices.
    
    Note:
      • Independent variables and dependent variables (except MotorAcuity) are obtained from the 'raw' data.
      • For MotorAcuity, the dependent value is obtained from the 'zscore' data.
      
    Returns:
      results: dict keyed by (indep_var, dep_var) with a list of results for each subject and hand.
               Each result is a dict with keys "subject", "hand", "correlation", and "p_value".
    """

    independent_vars = [
        "TW_LDLJ", "TW_sparc",
        "ballistic_LDLJ", "ballistic_sparc",
        "correction_LDLJ", "correction_sparc"
    ]
    dependent_vars = ['durations', 'distance', 'MotorAcuity']

    results = {}

    # Loop over independent and dependent variables
    for indep_var in independent_vars:
        for dep_var in dependent_vars:
            key_pair = (indep_var, dep_var)
            results[key_pair] = []

            # Loop over subjects in the raw data (assumed to be common between raw and zscore)
            for subject in all_median_stats["raw"]:
                # Loop over hands for the subject
                for hand in all_median_stats["raw"][subject]:
                    # Get independent variable from raw data
                    if indep_var in all_median_stats["raw"][subject][hand]:
                        indep_list = all_median_stats["raw"][subject][hand][indep_var]
                    else:
                        continue

                    # For dependent variable, use zscore if it's MotorAcuity, else raw.
                    if dep_var == "MotorAcuity":
                        if dep_var in all_median_stats["zscore"][subject][hand]:
                            dep_list = all_median_stats["zscore"][subject][hand][dep_var]
                        else:
                            continue
                    else:
                        if dep_var in all_median_stats["raw"][subject][hand]:
                            dep_list = all_median_stats["raw"][subject][hand][dep_var]
                        else:
                            continue

                    # Build dicts keyed by reach_index
                    indep_dict = {d["reach_index"]: d.get("median_value") for d in indep_list}
                    dep_dict   = {d["reach_index"]: d.get("median_value") for d in dep_list}

                    x_vals, y_vals = [], []
                    for idx in set(indep_dict.keys()) & set(dep_dict.keys()):
                        v1, v2 = indep_dict[idx], dep_dict[idx]
                        if v1 is not None and v2 is not None:
                            if not (np.isnan(v1) or np.isnan(v2)):
                                x_vals.append(v1)
                                y_vals.append(v2)                                

                    if len(x_vals) > 1 and len(y_vals) > 1:
                        corr, p_val = pearsonr(x_vals, y_vals)
                        results[key_pair].append({
                            "subject": subject,
                            "hand": hand,
                            "correlation": corr,
                            "p_value": p_val
                        })
    return results

pearson_results = compute_pearson_all_median_stats(all_median_stats)

def plot_grouped_median_correlations(pearson_results, hand='non_dominant', overlay_points=True, figuresize=(10, 5), metric='ldlj'):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_rel, ttest_1samp, shapiro
    from statsmodels.stats.multitest import multipletests

    axis_labels = dict(
        duration="Duration (s)",
        distance="Error (mm)",
        correlation="Correlation"
    )

    sig_levels = [(0.001, "***"), (0.01, "**"), (0.05, "*")]  # only annotate if p < 0.05

    prefix = metric
    phases = ['TW', 'Ballistic', 'Correction']
    metrics = ['durations', 'distance', 'MotorAcuity']

    # Build dictionary of median correlations per subject per key for each dependent metric
    data_dicts = {}
    for m in metrics:
        data_dicts[m] = {}
        for p in phases:
            # Key format: TW_ldlj or ballistic_ldlj, etc.
            key = (f"{p if p=='TW' else p.lower()}_{prefix}", m)
            entries = pearson_results.get(key, [])
            data_dicts[m][p] = {entry["subject"]: entry["correlation"]
                                  for entry in entries
                                  if entry["hand"].lower() == hand.lower()}

    # Prepare box plot positions and collect raw data for each metric-phase.
    categories = ["Duration (s)", "Error (mm)", "Motor Acuity"]
    x = np.arange(len(categories))
    width = 0.25
    colors = ['white', '#0047ff', '#ffb800']

    fig, ax = plt.subplots(figsize=figuresize)

    plot_positions = {}  # to store x positions used for each (metric, phase)
    # Plot a box plot for each metric and each phase.
    for j, phase in enumerate(phases):
        for i, m in enumerate(metrics):
            pos = x[i] + (j - 1) * width  # positions: TW left, Ballistic center, Correction right
            values = list(data_dicts[m][phase].values())
            bp = ax.boxplot(values, positions=[pos], widths=width * 0.8, patch_artist=True, showfliers=False,
                            medianprops=dict(color='black'))
            for box in bp['boxes']:
                box.set_facecolor(colors[j])
                box.set_edgecolor('black')
            # Overlay individual points if desired.
            if overlay_points and values:
                jitter = np.random.uniform(-0.05, 0.05, size=len(values))
                ax.scatter(np.full(len(values), pos) + jitter, values, color='black', alpha=0.5, zorder=5)
            plot_positions[(m, phase)] = pos

    # Statistical tests: compute paired tests and one-sample tests (each phase vs zero)
    test_results = []
    def fisher_z(r):
        r = np.clip(r, -0.9999, 0.9999)
        return 0.5 * np.log((1 + r) / (1 - r))
    
    for m in metrics:
        # Retrieve values and apply Fisher z-transform.
        phase_vals = {p: np.array(list(data_dicts[m][p].values())) for p in phases}
        phase_z = {p: fisher_z(phase_vals[p]) for p in phases}
        
        # Paired t-test: Ballistic vs Correction
        diff = phase_z['Ballistic'] - phase_z['Correction']
        n = len(diff)
        if n > 0:
            median_diff = np.median(diff)
            iqr_diff = np.percentile(diff, 75) - np.percentile(diff, 25)
            t_val, p_val = ttest_rel(phase_z['Ballistic'], phase_z['Correction'])
            df = n - 1
            # Cohen's d using the paired difference (mean/std)
            cohen_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else np.nan
            test_results.append({
                'metric': m,
                'test': 'Ballistic vs Correction',
                'median': median_diff,
                'IQR': iqr_diff,
                't': t_val,
                'df': df,
                'raw_p': p_val,
                'cohen_d': cohen_d
            })
        
        # One-sample t-tests: each phase vs zero.
        for phase in phases:
            x_vals = phase_z[phase]
            n_phase = len(x_vals)
            if n_phase > 0:
                median_x = np.median(x_vals)
                iqr_x = np.percentile(x_vals, 75) - np.percentile(x_vals, 25)
                t_val_phase, p_val_phase = ttest_1samp(x_vals, 0)
                df_phase = n_phase - 1
                # Cohen's d approximated as t/sqrt(n)
                cohen_d_phase = t_val_phase / np.sqrt(n_phase) if n_phase > 0 else np.nan
                test_results.append({
                    'metric': m,
                    'test': f'{phase} vs Zero',
                    'median': median_x,
                    'IQR': iqr_x,
                    't': t_val_phase,
                    'df': df_phase,
                    'raw_p': p_val_phase,
                    'cohen_d': cohen_d_phase
                })
                
    # Adjust all raw p-values via FDR.
    raw_p_values = [res['raw_p'] for res in test_results]
    _, pvals_corrected, _, _ = multipletests(raw_p_values, alpha=0.05, method='fdr_bh')
    for i, res in enumerate(test_results):
        res['adjusted_p'] = pvals_corrected[i]
        # Print details for each test if significant.
        if res['adjusted_p']:
            print(f"Metric: {res['metric']}, Test: {res['test']}")
            print(f"  Median: {res['median']:.2f}, IQR: {res['IQR']:.2f}")
            print(f"  t: {res['t']:.4f}, Adjusted p: {res['adjusted_p']:.3f}, Cohen's d: {res['cohen_d']:.2f}")

    # Annotate significance on the plot.
    y_max = 0.9
    y_step = 0.1
    line_positions = {m: y_max for m in metrics}
    for res in test_results:
        if res['adjusted_p'] < 0.05:
            m = res['metric']
            test_name = res['test']
            # Determine x-position(s) for annotation.
            if test_name == 'Ballistic vs Correction':
                x1 = plot_positions[(m, 'Ballistic')]
                x2 = plot_positions[(m, 'Correction')]
            else:
                phase = test_name.split()[0]  # "TW", "Ballistic", or "Correction"
                x1 = plot_positions[(m, phase)]
                x2 = x1
            y = line_positions[m]
            star = ""
            for thresh, s in sig_levels:
                if res['adjusted_p'] <= thresh:
                    star = s
                    break
            if test_name == 'Ballistic vs Correction':
                ax.plot([x1, x2], [y + 0.09, y + 0.09], color='black', linewidth=1.2)
                ax.text((x1 + x2) / 2, y + 0.1, star, ha='center', fontsize=20)
                line_positions[m] += y_step
            else:
                ax.text(x1, y - 0.15, star, ha='center', fontsize=20)

    # Annotate number of participants.
    all_subjects = set()
    for m in metrics:
        for p in phases:
            all_subjects.update(data_dicts[m][p].keys())
    n_participants = len(all_subjects)
    ax.text(0.91, 0.21, f"n = {n_participants} participants", transform=ax.transAxes,
            ha='right', va='top', fontsize=14)

    # Final plot touches.
    ax.axhline(0, color='lightgrey', linestyle='-', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=14)
    ax.set_ylabel(axis_labels["correlation"], fontsize=14)
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([-1, 0, 1], fontsize=12)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=phases[i]) for i in range(len(phases))]
    new_labels = ['Ballistic + Correction' if label == 'TW' else label for label in [patch.get_label() for patch in legend_elements]]
    ax.legend(legend_elements, new_labels, fontsize=14, frameon=False, loc='upper right', bbox_to_anchor=(1, 1.15))
    plt.tight_layout()
    plt.show()

plot_grouped_median_correlations(pearson_results, hand='non_dominant', overlay_points=True, figuresize=(8, 4), metric='LDLJ')
plot_grouped_median_correlations(pearson_results, hand='dominant', overlay_points=True, figuresize=(8, 4), metric='LDLJ')
plot_grouped_median_correlations(pearson_results, hand='non_dominant', overlay_points=True, figuresize=(8, 4), metric='sparc')
plot_grouped_median_correlations(pearson_results, hand='dominant', overlay_points=True, figuresize=(8, 4), metric='sparc')







def plot_grouped_median_correlations(pearson_results, hand='non_dominant', overlay_points=True, figuresize=(10, 5), metric='ldlj'):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_rel, ttest_1samp, shapiro
    from statsmodels.stats.multitest import multipletests

    axis_labels = dict(
        duration="Duration (s)",
        distance="Error (mm)",
        correlation="Correlation"
    )

    sig_levels = [(0.001, "***"), (0.01, "**"), (0.05, "*")]  # only annotate if p < 0.05

    prefix = metric
    phases = ['TW', 'Ballistic', 'Correction']
    metrics = ['durations', 'distance', 'MotorAcuity']

    # Build dictionary of median correlations per subject per key for each dependent metric
    data_dicts = {}
    for m in metrics:
        data_dicts[m] = {}
        for p in phases:
            # Key format: TW_ldlj or ballistic_ldlj, etc.
            key = (f"{p if p=='TW' else p.lower()}_{prefix}", m)
            entries = pearson_results.get(key, [])
            data_dicts[m][p] = {entry["subject"]: entry["correlation"]
                                  for entry in entries
                                  if entry["hand"].lower() == hand.lower()}

    # Prepare box plot positions and collect raw data for each metric-phase.
    categories = ["Duration (s)", "Error (mm)", "Motor Acuity"]
    x = np.arange(len(categories))
    width = 0.25
    colors = ['white', '#0047ff', '#ffb800']

    fig, ax = plt.subplots(figsize=figuresize)

    plot_positions = {}  # to store x positions used for each (metric, phase)
    # Plot a box plot for each metric and each phase.
    for j, phase in enumerate(phases):
        for i, m in enumerate(metrics):
            pos = x[i] + (j - 1) * width  # positions: TW left, Ballistic center, Correction right
            values = list(data_dicts[m][phase].values())
            bp = ax.boxplot(values, positions=[pos], widths=width * 0.8, patch_artist=True, showfliers=False,
                            medianprops=dict(color='black'))
            for box in bp['boxes']:
                box.set_facecolor(colors[j])
                box.set_edgecolor('black')
            # Overlay individual points if desired.
            if overlay_points and values:
                jitter = np.random.uniform(-0.05, 0.05, size=len(values))
                ax.scatter(np.full(len(values), pos) + jitter, values, color='black', alpha=0.5, zorder=5)
            plot_positions[(m, phase)] = pos

    # Connect same subject data points across the three phases.
    for m in metrics:
        # Get the union of subjects that have values for this metric.
        subjects = set().union(*(data_dicts[m][p].keys() for p in phases))
        for subj in subjects:
            xs = []
            ys = []
            for p in phases:
                if subj in data_dicts[m][p]:
                    xs.append(plot_positions[(m, p)])
                    ys.append(data_dicts[m][p][subj])
            if len(xs) >= 2:
                ax.plot(xs, ys, linestyle='-', color='black', alpha=0.7, linewidth=1)

    # Statistical tests: compute paired tests and one-sample tests (each phase vs zero)
    test_results = []
    def fisher_z(r):
        r = np.clip(r, -0.9999, 0.9999)
        return 0.5 * np.log((1 + r) / (1 - r))
    
    for m in metrics:
        # Retrieve values and apply Fisher z-transform.
        phase_vals = {p: np.array(list(data_dicts[m][p].values())) for p in phases}
        phase_z = {p: fisher_z(phase_vals[p]) for p in phases}
        
        # Paired t-test: Ballistic vs Correction
        diff = phase_z['Ballistic'] - phase_z['Correction']
        n = len(diff)
        if n > 0:
            median_diff = np.median(diff)
            iqr_diff = np.percentile(diff, 75) - np.percentile(diff, 25)
            t_val, p_val = ttest_rel(phase_z['Ballistic'], phase_z['Correction'])
            df = n - 1
            # Cohen's d using the paired difference (mean/std)
            cohen_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else np.nan
            test_results.append({
                'metric': m,
                'test': 'Ballistic vs Correction',
                'median': median_diff,
                'IQR': iqr_diff,
                't': t_val,
                'df': df,
                'raw_p': p_val,
                'cohen_d': cohen_d
            })
        
        # One-sample t-tests: each phase vs zero.
        for phase in phases:
            x_vals = phase_z[phase]
            n_phase = len(x_vals)
            if n_phase > 0:
                median_x = np.median(x_vals)
                iqr_x = np.percentile(x_vals, 75) - np.percentile(x_vals, 25)
                t_val_phase, p_val_phase = ttest_1samp(x_vals, 0)
                df_phase = n_phase - 1
                # Cohen's d approximated as t/sqrt(n)
                cohen_d_phase = t_val_phase / np.sqrt(n_phase) if n_phase > 0 else np.nan
                test_results.append({
                    'metric': m,
                    'test': f'{phase} vs Zero',
                    'median': median_x,
                    'IQR': iqr_x,
                    't': t_val_phase,
                    'df': df_phase,
                    'raw_p': p_val_phase,
                    'cohen_d': cohen_d_phase
                })
                
    # Adjust all raw p-values via FDR.
    raw_p_values = [res['raw_p'] for res in test_results]
    _, pvals_corrected, _, _ = multipletests(raw_p_values, alpha=0.05, method='fdr_bh')
    for i, res in enumerate(test_results):
        res['adjusted_p'] = pvals_corrected[i]
        # Print details for each test if significant.
        if res['adjusted_p'] < 0.05:
            print(f"Metric: {res['metric']}, Test: {res['test']}")
            print(f"  Median: {res['median']:.2f}, IQR: {res['IQR']:.2f}")
            print(f"  t: {res['t']:.4f}, Adjusted p: {res['adjusted_p']:.3f}, Cohen's d: {res['cohen_d']:.2f}")

    # Annotate significance on the plot.
    y_max = 0.9
    y_step = 0.1
    line_positions = {m: y_max for m in metrics}
    for res in test_results:
        if res['adjusted_p'] < 0.05:
            m = res['metric']
            test_name = res['test']
            # Determine x-position(s) for annotation.
            if test_name == 'Ballistic vs Correction':
                x1 = plot_positions[(m, 'Ballistic')]
                x2 = plot_positions[(m, 'Correction')]
            else:
                phase = test_name.split()[0]  # "TW", "Ballistic", or "Correction"
                x1 = plot_positions[(m, phase)]
                x2 = x1
            y = line_positions[m]
            star = ""
            for thresh, s in sig_levels:
                if res['adjusted_p'] <= thresh:
                    star = s
                    break
            if test_name == 'Ballistic vs Correction':
                ax.plot([x1, x2], [y + 0.09, y + 0.09], color='black', linewidth=1.2)
                ax.text((x1 + x2) / 2, y + 0.1, star, ha='center', fontsize=20)
                line_positions[m] += y_step
            else:
                ax.text(x1, y - 0.15, star, ha='center', fontsize=20)

    # Annotate number of participants.
    all_subjects = set()
    for m in metrics:
        for p in phases:
            all_subjects.update(data_dicts[m][p].keys())
    n_participants = len(all_subjects)
    ax.text(0.91, 0.21, f"n = {n_participants} participants", transform=ax.transAxes,
            ha='right', va='top', fontsize=14)

    # Final plot touches.
    ax.axhline(0, color='lightgrey', linestyle='-', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=14)
    ax.set_ylabel(axis_labels["correlation"], fontsize=14)
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([-1, 0, 1], fontsize=12)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=phases[i]) for i in range(len(phases))]
    new_labels = ['Ballistic + Correction' if label == 'TW' else label for label in [patch.get_label() for patch in legend_elements]]
    ax.legend(legend_elements, new_labels, fontsize=14, frameon=False, loc='upper right', bbox_to_anchor=(1, 1.15))
    plt.tight_layout()
    plt.show()
plot_grouped_median_correlations(pearson_results, hand='non_dominant', overlay_points=True, figuresize=(20, 20), metric='sparc')
plot_grouped_median_correlations(pearson_results, hand='dominant', overlay_points=True, figuresize=(8, 4), metric='sparc')










from scipy.stats import wilcoxon
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, ttest_rel, shapiro
from scipy.stats import shapiro, wilcoxon, ttest_rel
import math
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from statsmodels.stats.multitest import multipletests
from tabulate import tabulate
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def compare_hands(pearson_results, metrics=['durations', 'distance', 'MotorAcuity'], 
                  phases=['TW', 'Ballistic', 'Correction'], metric_prefix='ldlj'):
    """
    Compare correlation values between dominant and non-dominant hands for each subject.
    Performs paired Wilcoxon signed-rank test.
    """
    # Build dictionary of correlations per subject per hand
    hand_data = {'dominant': {}, 'non_dominant': {}}
    
    for hand in ['dominant', 'non_dominant']:
        hand_data[hand] = {m: {} for m in metrics}
        for m in metrics:
            for p in phases:
                key = f"{p if p=='TW' else p.lower()}_{metric_prefix}"
                entries = pearson_results.get((key, m), [])
                hand_data[hand][m][p] = {entry["subject"]: entry["correlation"]
                                         for entry in entries if entry["hand"].lower() == hand}

    # Paired comparisons between hands
    results = []
    for m in metrics:
        for p in phases:
            dom_values = hand_data['dominant'][m][p]
            nondom_values = hand_data['non_dominant'][m][p]
            # Only keep subjects present in both hands
            common_subjects = set(dom_values.keys()) & set(nondom_values.keys())
            if len(common_subjects) > 0:
                dom = [dom_values[s] for s in common_subjects]
                nondom = [nondom_values[s] for s in common_subjects]
                stat, pval = wilcoxon(dom, nondom)
                results.append({
                    "metric": m,
                    "phase": p,
                    "n_subjects": len(common_subjects),
                    "stat": stat,
                    "p_value": pval
                })
    return results

# Example usage:
hand_comparison_results = compare_hands(pearson_results, metric_prefix='LDLJ')
for r in hand_comparison_results:
    print(r)
hand_comparison_results = compare_hands(pearson_results, metric_prefix='sparc')
for r in hand_comparison_results:
    print(r)





# --- 
# def plot_spearman_correlations_heatmap(spearman_results):
#     """
#     Separates spearman_results into four groups based on the independent variable (LDLJ or SPARC)
#     and hand (dominant or non_dominant). For each key (indep_var, dep_var), the function averages the 
#     correlation values (per subject) separately for dominant and non-dominant hands and then plots 
#     four heatmaps: one for LDLJ-dominant, one for LDLJ-non_dominant, one for SPARC-dominant, and one for SPARC-non_dominant.
    
#     Parameters:
#         spearman_results (dict): Dictionary keyed by (indep_var, dep_var) with a list of results,
#                                  where each result is a dict with keys "subject", "hand", and "correlation".
#     """
#     ldldj_data_dom = {}
#     ldldj_data_ndom = {}
#     sparc_data_dom = {}
#     sparc_data_ndom = {}

#     for key, res_list in spearman_results.items():
#         indep_var, dep_var = key
#         col_name = f"{indep_var} vs {dep_var}"
#         if "LDLJ" in indep_var.upper():
#             for entry in res_list:
#                 subj = entry["subject"]
#                 corr = entry["correlation"]
#                 if entry["hand"].lower() == "dominant":
#                     if subj not in ldldj_data_dom:
#                         ldldj_data_dom[subj] = {}
#                     ldldj_data_dom[subj].setdefault(col_name, []).append(corr)
#                 elif entry["hand"].lower() == "non_dominant":
#                     if subj not in ldldj_data_ndom:
#                         ldldj_data_ndom[subj] = {}
#                     ldldj_data_ndom[subj].setdefault(col_name, []).append(corr)
#         elif "SPARC" in indep_var.upper():
#             for entry in res_list:
#                 subj = entry["subject"]
#                 corr = entry["correlation"]
#                 if entry["hand"].lower() == "dominant":
#                     if subj not in sparc_data_dom:
#                         sparc_data_dom[subj] = {}
#                     sparc_data_dom[subj].setdefault(col_name, []).append(corr)
#                 elif entry["hand"].lower() == "non_dominant":
#                     if subj not in sparc_data_ndom:
#                         sparc_data_ndom[subj] = {}
#                     sparc_data_ndom[subj].setdefault(col_name, []).append(corr)

#     # Function to average correlations over the list of values per subject per column.
#     def average_dict(data_dict):
#         for subj in data_dict:
#             for col in data_dict[subj]:
#                 data_dict[subj][col] = np.mean(data_dict[subj][col])
#         return pd.DataFrame.from_dict(data_dict, orient="index").sort_index()
    
#     df_ldldj_dom = average_dict(ldldj_data_dom) if ldldj_data_dom else pd.DataFrame()
#     df_ldldj_ndom = average_dict(ldldj_data_ndom) if ldldj_data_ndom else pd.DataFrame()
#     df_sparc_dom = average_dict(sparc_data_dom) if sparc_data_dom else pd.DataFrame()
#     df_sparc_ndom = average_dict(sparc_data_ndom) if sparc_data_ndom else pd.DataFrame()

#     # Plotting function for one heatmap
#     def plot_heatmap(df, title, ax):
#         sns.heatmap(df, annot=True, cmap="coolwarm", center=0,
#                     cbar_kws={'label': 'Spearman r'}, ax=ax)
#         ax.set_xlabel("Correlation Type", fontsize=14)
#         ax.set_ylabel("Subject", fontsize=14)
#         ax.set_title(title, fontsize=16)

#     # Create a figure with 2 rows and 2 columns
#     fig, axes = plt.subplots(2, 2, figsize=(18, 12))
#     plot_heatmap(df_ldldj_dom, "LDLJ Comparisons - Dominant Hand", axes[0, 0])
#     plot_heatmap(df_ldldj_ndom, "LDLJ Comparisons - Non-Dominant Hand", axes[0, 1])
#     plot_heatmap(df_sparc_dom, "SPARC Comparisons - Dominant Hand", axes[1, 0])
#     plot_heatmap(df_sparc_ndom, "SPARC Comparisons - Non-Dominant Hand", axes[1, 1])

#     plt.tight_layout()
#     plt.show()
# plot_spearman_correlations_heatmap(spearman_results)

# def plot_spearman_correlations_bars(spearman_results):
#     """
#     Separates spearman_results into four groups based on the independent variable (LDLJ or SPARC)
#     and hand (dominant or non_dominant). For each key (indep_var, dep_var), the function aggregates the 
#     correlation values (per subject) separately for dominant and non_dominant hands and then computes
#     the median correlation (and standard error using the standard deviation across subjects) for each correlation type.
#     The plots are reordered so that all comparisons with durations come first, then distance, then MotorAcuity.
#     Creates 4 bar plots: one for LDLJ-dominant, one for LDLJ-non_dominant, one for SPARC-dominant, 
#     and one for SPARC-non_dominant.
    
#     Parameters:
#         spearman_results (dict): Dictionary keyed by (indep_var, dep_var) with a list of results,
#                                  where each result is a dict with keys "subject", "hand", and "correlation".
#     """
#     import matplotlib.pyplot as plt
#     import pandas as pd
#     import numpy as np

#     # Separate entries into four groups
#     ldldj_data_dom = {}
#     ldldj_data_ndom = {}
#     sparc_data_dom = {}
#     sparc_data_ndom = {}

#     for key, res_list in spearman_results.items():
#         indep_var, dep_var = key
#         # Form column name as: "<indep_var> vs <dep_var>" 
#         col_name = f"{indep_var} vs {dep_var}"
#         if "LDLJ" in indep_var.upper():
#             for entry in res_list:
#                 subj = entry["subject"]
#                 corr = entry["correlation"]
#                 if entry["hand"].lower() == "dominant":
#                     if subj not in ldldj_data_dom:
#                         ldldj_data_dom[subj] = {}
#                     ldldj_data_dom[subj].setdefault(col_name, []).append(corr)
#                 elif entry["hand"].lower() == "non_dominant":
#                     if subj not in ldldj_data_ndom:
#                         ldldj_data_ndom[subj] = {}
#                     ldldj_data_ndom[subj].setdefault(col_name, []).append(corr)
#         elif "SPARC" in indep_var.upper():
#             for entry in res_list:
#                 subj = entry["subject"]
#                 corr = entry["correlation"]
#                 if entry["hand"].lower() == "dominant":
#                     if subj not in sparc_data_dom:
#                         sparc_data_dom[subj] = {}
#                     sparc_data_dom[subj].setdefault(col_name, []).append(corr)
#                 elif entry["hand"].lower() == "non_dominant":
#                     if subj not in sparc_data_ndom:
#                         sparc_data_ndom[subj] = {}
#                     sparc_data_ndom[subj].setdefault(col_name, []).append(corr)

#     # Function to aggregate correlations by taking the median over the list of values per subject per column.
#     def aggregate_dict(data_dict):
#         for subj in data_dict:
#             for col in data_dict[subj]:
#                 data_dict[subj][col] = np.median(data_dict[subj][col])
#         df = pd.DataFrame.from_dict(data_dict, orient="index")
#         # Reorder columns by dependent variable: durations first, then distance, then motoracuity.
#         def dep_order(col):
#             try:
#                 dep = col.split(" vs ")[1].replace(" ", "").lower()
#             except IndexError:
#                 dep = ""
#             order = {"durations": 0, "distance": 1, "motoracuity": 2}
#             return order.get(dep, 99)
#         sorted_cols = sorted(df.columns, key=dep_order)
#         return df.reindex(columns=sorted_cols)

#     df_ldldj_dom = aggregate_dict(ldldj_data_dom) if ldldj_data_dom else pd.DataFrame()
#     df_ldldj_ndom = aggregate_dict(ldldj_data_ndom) if ldldj_data_ndom else pd.DataFrame()
#     df_sparc_dom = aggregate_dict(sparc_data_dom) if sparc_data_dom else pd.DataFrame()
#     df_sparc_ndom = aggregate_dict(sparc_data_ndom) if sparc_data_ndom else pd.DataFrame()

#     # Function to plot a bar plot for one group given a DataFrame.
#     def plot_bar(df, title, ax):
#         # For each column (correlation type), compute median and standard error across subjects.
#         medians = df.median(axis=0)
#         sem = df.std(axis=0, ddof=1) / np.sqrt(df.shape[0])
#         x = np.arange(len(medians))
#         ax.bar(x, medians, yerr=sem, capsize=5, color='skyblue', edgecolor='black')
#         ax.set_xticks(x)
#         ax.set_xticklabels(medians.index, rotation=45, ha='right', fontsize=10)
#         ax.set_ylabel('Spearman r', fontsize=12)
#         ax.set_title(title, fontsize=14)
#         ax.axhline(0, color='gray', linewidth=0.8)

#     # Create a figure with 2 rows and 2 columns for the four groups
#     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#     plot_bar(df_ldldj_dom, "LDLJ Comparisons - Dominant Hand", axes[0, 0])
#     plot_bar(df_ldldj_ndom, "LDLJ Comparisons - Non-Dominant Hand", axes[0, 1])
#     plot_bar(df_sparc_dom, "SPARC Comparisons - Dominant Hand", axes[1, 0])
#     plot_bar(df_sparc_ndom, "SPARC Comparisons - Non-Dominant Hand", axes[1, 1])
#     plt.tight_layout()
#     plt.show()
# plot_spearman_correlations_bars(spearman_results)

# def plot_grouped_median_correlations(spearman_results, hand='non_dominant', overlay_points=True, figuresize=(10, 5), metric='ldlj'):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from scipy.stats import wilcoxon
#     from statsmodels.stats.multitest import multipletests



#     axis_labels = dict(
#         duration="Duration (s)",
#         distance="Error (mm)",
#         correlation="Correlation"
#     )

#     sig_levels = [(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")]

#     prefix = metric
#     phases = ['TW', 'Ballistic', 'Correction']
#     metrics = ['durations', 'distance', 'MotorAcuity']

#     # Build dictionary of median correlations per subject per key for each dependent metric
#     data_dicts = {}
#     for m in metrics:
#         data_dicts[m] = {}
#         for p in phases:
#             # Key format: TW_LDLJ or ballistic_LDLJ, etc.
#             key = (f"{p if p=='TW' else p.lower()}_{prefix}", m)
#             # Extract entries from spearman_results for this key matching the specified hand.
#             # Expect spearman_results[key] to be a list of dicts with keys "subject", "hand", "correlation"
#             entries = spearman_results.get(key, [])
#             # Build a dictionary mapping subject -> correlation value
#             data_dicts[m][p] = {entry["subject"]: entry["correlation"] 
#                                   for entry in entries 
#                                   if entry["hand"].lower() == hand.lower()}
    
#     def compute_median_and_error(data):
#         vals = np.array(list(data.values()))
#         median = np.median(vals)
#         sem = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
#         return median, sem

#     def fisher_z(r):
#         r = np.clip(r, -0.9999, 0.9999)
#         return 0.5 * np.log((1 + r) / (1 - r))

#     # Prepare plotting values
#     all_vals, all_errs = [], []
#     for m in metrics:
#         vals, errs = [], []
#         for p in phases:
#             v, e = compute_median_and_error(data_dicts[m][p])
#             vals.append(v)
#             errs.append(e)
#         all_vals.append(vals)
#         all_errs.append(errs)
#     all_vals = np.array(all_vals)
#     all_errs = np.array(all_errs)

#     categories = ["Duration (s)", "Error (mm)", "Motor Acuity"]
#     x = np.arange(len(categories))
#     width = 0.25
#     colors = ['white', 'blue', 'red']

#     fig, ax = plt.subplots(figsize=figuresize)

#     bar_positions = {}  # store actual x positions of each bar for line plotting

#     for i, p in enumerate(phases):
#         bar = ax.bar(x + i * width - width, all_vals[:, i], width, yerr=all_errs[:, i],
#                      capsize=5, color=colors[i], edgecolor='black', label=p)
#         if overlay_points:
#             for j, m in enumerate(metrics):
#                 pts = np.array(list(data_dicts[m][p].values()))
#                 jitter = np.random.uniform(-0.05, 0.05, size=pts.shape)
#                 ax.scatter(np.full(pts.shape, x[j] + i * width - width) + jitter, pts,
#                            color='black', alpha=0.5, zorder=5)
#         # Store bar positions for each metric & phase
#         for j, m in enumerate(metrics):
#             bar_positions[(m, p)] = x[j] + i * width - width

#     # Statistical comparisons
#     pvals = []
#     comparisons = []
#     for m in metrics:
#         phase_vals = {p: np.array(list(data_dicts[m][p].values())) for p in phases}
#         phase_z = {p: fisher_z(phase_vals[p]) for p in phases}

#         # Pairwise comparisons between phases
#         pairs = [('TW', 'Ballistic'), ('TW', 'Correction'), ('Ballistic', 'Correction')]
#         for p1, p2 in pairs:
#             stat_val, p_val = wilcoxon(phase_z[p1], phase_z[p2])
#             pvals.append(p_val)
#             comparisons.append((m, p1, p2, p_val))

#         # Comparisons vs zero
#         for phase_name in phases:
#             stat_val, p_val = wilcoxon(phase_z[phase_name], np.zeros_like(phase_z[phase_name]))
#             pvals.append(p_val)
#             comparisons.append((m, phase_name, 'Zero', p_val))

#     # FDR correction for multiple comparisons
#     _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

#     # Annotate significance lines (horizontal)
#     y_max = 0.8
#     y_step = 0.1
#     line_positions = {m: y_max for m in metrics}

#     for idx, (m, p1, p2, _) in enumerate(comparisons):
#         y = line_positions[m]
#         x1 = bar_positions[(m, p1)]
#         if p2 == 'Zero':
#             for thresh, star in sig_levels:
#                 if pvals_corrected[idx] <= thresh:
#                     ax.text(x1, y_max - 0.1, star, ha='center', fontsize=20)
#                     break
#         else:
#             x2 = bar_positions[(m, p2)]
#             ax.plot([x1, x2], [y, y], color='black', linewidth=1.2)
#             for thresh, star in sig_levels:
#                 if pvals_corrected[idx] <= thresh:
#                     ax.text((x1 + x2) / 2, y + 0.01, star, ha='center', fontsize=20)
#                     break
#             line_positions[m] += y_step

#     # Annotate sample size (n) where unit is participants. Compute overall subjects across all metrics and phases.
#     all_subjects = set()
#     for m in metrics:
#         for p in phases:
#             all_subjects.update(data_dicts[m][p].keys())
#     n_participants = len(all_subjects)
#     ax.text(0.95, 0.25, f"n = {n_participants} participants", transform=ax.transAxes,
#             ha='right', va='top', fontsize=14)

#     # Final figure touches
#     ax.axhline(0, color='lightgrey', linestyle='-', linewidth=1)
#     ax.set_xticks(x)
#     ax.set_xticklabels(categories, fontsize=14)
#     ax.set_ylabel(axis_labels["correlation"], fontsize=14)
#     ax.set_ylim(-1, 1)
#     ax.set_yticks([-1, 0, 1])
#     ax.set_yticklabels([-1, 0, 1], fontsize=12)
#     ax.grid(False)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     handles, labels = ax.get_legend_handles_labels()
#     new_labels = ['Ballistic + Correction' if label == 'TW' else label for label in labels]
#     ax.legend(handles, new_labels, fontsize=12, frameon=False, loc='best')
#     plt.tight_layout()
#     plt.show()
# plot_grouped_median_correlations(spearman_results, hand='non_dominant', overlay_points=True, figuresize=(10, 5), metric='LDLJ')
# plot_grouped_median_correlations(spearman_results, hand='non_dominant', overlay_points=True, figuresize=(10, 5), metric='sparc')
# #------
# def plot_grouped_median_correlations(spearman_results, hand='non_dominant', overlay_points=True, figuresize=(10, 5), metric='ldlj'):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from scipy.stats import wilcoxon
#     from statsmodels.stats.multitest import multipletests

#     axis_labels = dict(
#         duration="Duration (s)",
#         distance="Error (mm)",
#         correlation="Correlation"
#     )

#     sig_levels = [(0.001, "***"), (0.01, "**"), (0.05, "*")]  # only annotate if p < 0.05

#     prefix = metric
#     phases = ['TW', 'Ballistic', 'Correction']
#     metrics = ['durations', 'distance', 'MotorAcuity']

#     # Build dictionary of median correlations per subject per key for each dependent metric
#     data_dicts = {}
#     for m in metrics:
#         data_dicts[m] = {}
#         for p in phases:
#             # Key format: TW_ldlj or ballistic_ldlj, etc.
#             key = (f"{p if p=='TW' else p.lower()}_{prefix}", m)
#             entries = spearman_results.get(key, [])
#             data_dicts[m][p] = {entry["subject"]: entry["correlation"]
#                                   for entry in entries
#                                   if entry["hand"].lower() == hand.lower()}

#     # Prepare box plot positions and collect raw data for each metric-phase.
#     categories = ["Duration (s)", "Error (mm)", "Motor Acuity"]
#     x = np.arange(len(categories))
#     width = 0.25
#     colors = ['white', '#0047ff', '#ffb800']

#     fig, ax = plt.subplots(figsize=figuresize)

#     plot_positions = {}  # to store x positions used for each (metric, phase)
#     # Plot a box plot for each metric and each phase.
#     for j, phase in enumerate(phases):
#         for i, m in enumerate(metrics):
#             pos = x[i] + (j - 1) * width  # center phase TW at offset -width, Ballistic at center, Correction at +width
#             # Get data as a list.
#             values = list(data_dicts[m][phase].values())
#             # Plot boxplot. Specify medianprops to have the median bar black.
#             bp = ax.boxplot(values, positions=[pos], widths=width*0.8, patch_artist=True, showfliers=False,
#                             medianprops=dict(color='black'))
#             for box in bp['boxes']:
#                 box.set_facecolor(colors[j])
#                 box.set_edgecolor('black')
#             # Optionally overlay individual points.
#             if overlay_points and values:
#                 jitter = np.random.uniform(-0.05, 0.05, size=len(values))
#                 ax.scatter(np.full(len(values), pos) + jitter, values, color='black', alpha=0.5, zorder=5)
#             plot_positions[(m, phase)] = pos

#     # Statistical comparisons: only test Ballistic vs Correction and each phase vs zero.
#     def fisher_z(r):
#         r = np.clip(r, -0.9999, 0.9999)
#         return 0.5 * np.log((1 + r) / (1 - r))

#     pvals = []
#     comparisons = []
#     for m in metrics:
#         phase_vals = {p: np.array(list(data_dicts[m][p].values())) for p in phases}
#         phase_z = {p: fisher_z(phase_vals[p]) for p in phases}
#         # Comparison between Ballistic and Correction.
#         stat_val, p_val = wilcoxon(phase_z['Ballistic'], phase_z['Correction'])
#         pvals.append(p_val)
#         comparisons.append((m, 'Ballistic', 'Correction', p_val))
#         # Comparisons for each phase vs zero.
#         for phase in phases:
#             stat_val, p_val = wilcoxon(phase_z[phase], np.zeros_like(phase_z[phase]))
#             pvals.append(p_val)
#             comparisons.append((m, phase, 'Zero', p_val))

#     _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

#     # Annotate significance lines using the boxplot positions only if p < 0.05.
#     y_max = 0.8
#     y_step = 0.1
#     line_positions = {m: y_max for m in metrics}

#     for idx, (m, p1, p2, _) in enumerate(comparisons):
#         if pvals_corrected[idx] < 0.05:
#             y = line_positions[m]
#             x1 = plot_positions[(m, p1)]
#             if p2 == 'Zero':
#                 for thresh, star in sig_levels:
#                     if pvals_corrected[idx] <= thresh:
#                         ax.text(x1, y - 0.15, star, ha='center', fontsize=20)
#                         break
#             else:
#                 x2 = plot_positions[(m, p2)]
#                 ax.plot([x1, x2], [y + 0.09, y + 0.09], color='black', linewidth=1.2)
#                 for thresh, star in sig_levels:
#                     if pvals_corrected[idx] <= thresh:
#                         ax.text((x1 + x2) / 2, y + 0.1, star, ha='center', fontsize=20)
#                         break
#                 line_positions[m] += y_step

#     # Annotate number of participants.
#     all_subjects = set()
#     for m in metrics:
#         for p in phases:
#             all_subjects.update(data_dicts[m][p].keys())
#     n_participants = len(all_subjects)
#     ax.text(0.91, 0.23, f"n = {n_participants} participants", transform=ax.transAxes,
#             ha='right', va='top', fontsize=14)

#     # Final touches.
#     ax.axhline(0, color='lightgrey', linestyle='-', linewidth=1)
#     ax.set_xticks(x)
#     ax.set_xticklabels(categories, fontsize=14)
#     ax.set_ylabel(axis_labels["correlation"], fontsize=14)
#     ax.set_ylim(-1, 1)
#     ax.set_yticks([-1, 0, 1])
#     ax.set_yticklabels([-1, 0, 1], fontsize=12)
#     ax.grid(False)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     # Modify legend: show phase names with colors.
#     from matplotlib.patches import Patch

#     legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=phases[i]) for i in range(len(phases))]
#     new_labels = ['Ballistic + Correction' if label == 'TW' else label for label in [patch.get_label() for patch in legend_elements]]
#     ax.legend(legend_elements, new_labels, fontsize=14, frameon=False, loc='upper right', bbox_to_anchor=(1, 1.1))
#     plt.tight_layout()
#     plt.show()

# plot_grouped_median_correlations(spearman_results, hand='non_dominant', overlay_points=True, figuresize=(8, 4), metric='LDLJ')
# plot_grouped_median_correlations(spearman_results, hand='non_dominant', overlay_points=True, figuresize=(8, 4), metric='sparc')
# plot_grouped_median_correlations(spearman_results, hand='dominant', overlay_points=True, figuresize=(10, 5), metric='LDLJ')
# plot_grouped_median_correlations(spearman_results, hand='dominant', overlay_points=True, figuresize=(10, 5), metric='sparc')

# #------------------------------------------------------------------------------
# #------
# # Compute the minimal and maximal LDLJ values across segments (1 to 16) and print 
# # the corresponding trial name and segment index

# def compute_ldlj_stats(LDLJ_phase, subject="07/22/HW", hand="left"):
#     min_ldlj = float('inf')
#     max_ldlj = float('-inf')
#     min_trial = None
#     max_trial = None
#     min_segment = None
#     max_segment = None

#     # Create a list to hold all entries for further computations
#     ldlj_entries = []

#     for trial_name, trial_values in LDLJ_phase['reach_LDLJ'][subject][hand].items():
#         for seg in range(1, 17):  # iteration over segments 1 to 16
#             value = trial_values[seg - 1]
#             ldlj_entries.append((value, trial_name, seg))
#             if value < min_ldlj:
#                 min_ldlj = value
#                 min_trial = trial_name
#                 min_segment = seg
#             if value > max_ldlj:
#                 max_ldlj = value
#                 max_trial = trial_name
#                 max_segment = seg

#     print("Minimum LDLJ value across trials:", min_ldlj)
#     print("Found in trial:", min_trial, "at segment index:", min_segment)
#     print("Maximum LDLJ value across trials:", max_ldlj)
#     print("Found in trial:", max_trial, "at segment index:", max_segment)

#     # Compute the median LDLJ value across segments from all trials
#     all_values = [entry[0] for entry in ldlj_entries]
#     median_ldlj = np.median(all_values)

#     closest_diff = float('inf')
#     median_trial = None
#     median_segment = None

#     for value, trial_name, seg in ldlj_entries:
#         diff = abs(value - median_ldlj)
#         if diff < closest_diff:
#             closest_diff = diff
#             median_trial = trial_name
#             median_segment = seg

#     print("Median LDLJ value across trials:", median_ldlj)
#     print("Closest value found in trial:", median_trial, "at segment index:", median_segment)

#     # Find the bottom 5 (smallest) and top 5 (largest) LDLJ values
#     sorted_entries = sorted(ldlj_entries, key=lambda x: x[0])
#     bottom_5 = sorted_entries[:5]
#     top_5 = sorted_entries[-5:]  # highest 5 values

#     print("\nBottom 5 LDLJ values:")
#     for value, trial_name, seg in bottom_5:
#         print(f"LDLJ: {value}, Trial: {trial_name}, Segment: {seg}")

#     print("\nTop 5 LDLJ values:")
#     for value, trial_name, seg in top_5:
#         print(f"LDLJ: {value}, Trial: {trial_name}, Segment: {seg}")
#     return bottom_5, top_5
    
# bottom_5, top_5 = compute_ldlj_stats(reach_TW_metrics_TW, subject="07/22/HW", hand="left")

# def compute_sparc_stats(reach_sparc_ballistic_phase, subject="07/22/HW", hand="left"):
#     min_sparc = float('inf')
#     max_sparc = float('-inf')
#     min_trial = None
#     max_trial = None
#     min_segment = None
#     max_segment = None

#     # Create a list to hold all entries for further computations
#     sparc_entries = []

#     for trial_name, trial_values in reach_sparc_ballistic_phase[subject][hand].items():
#         for seg in range(1, 17):  # iteration over segments 1 to 16
#             value = trial_values[seg - 1]
#             sparc_entries.append((value, trial_name, seg))
#             if value < min_sparc:
#                 min_sparc = value
#                 min_trial = trial_name
#                 min_segment = seg
#             if value > max_sparc:
#                 max_sparc = value
#                 max_trial = trial_name
#                 max_segment = seg

#     print("Minimum SPARC value across trials:", min_sparc)
#     print("Found in trial:", min_trial, "at segment index:", min_segment)
#     print("Maximum SPARC value across trials:", max_sparc)
#     print("Found in trial:", max_trial, "at segment index:", max_segment)

#     # Compute the median SPARC value across segments from all trials
#     all_values = [entry[0] for entry in sparc_entries]
#     median_sparc = np.median(all_values)

#     closest_diff = float('inf')
#     median_trial = None
#     median_segment = None

#     for value, trial_name, seg in sparc_entries:
#         diff = abs(value - median_sparc)
#         if diff < closest_diff:
#             closest_diff = diff
#             median_trial = trial_name
#             median_segment = seg

#     print("Median SPARC value across trials:", median_sparc)
#     print("Closest value found in trial:", median_trial, "at segment index:", median_segment)

#     # Find the bottom 5 (smallest) and top 5 (largest) SPARC values
#     sorted_entries = sorted(sparc_entries, key=lambda x: x[0])
#     bottom_5 = sorted_entries[:5]
#     top_5 = sorted_entries[-5:]  # highest 5 values

#     print("\nBottom 5 SPARC values:")
#     for value, trial_name, seg in bottom_5:
#         print(f"SPARC: {value}, Trial: {trial_name}, Segment: {seg}")

#     print("\nTop 5 SPARC values:")
#     for value, trial_name, seg in top_5:
#         print(f"SPARC: {value}, Trial: {trial_name}, Segment: {seg}")
#     return bottom_5, top_5
# bottom_5, top_5 = compute_sparc_stats(reach_sparc_TW, subject="07/22/HW", hand="left")

# # Iterate through the time-window metrics (LDLJ) and SPARC metrics,
# # printing out the subject, hand, trial (file_path), and segment number
# # when LDLJ < 9.47 and the corresponding SPARC value is < 1.83.

# for subj, hands in reach_TW_metrics_TW['reach_LDLJ'].items():
#     if subj != "07/22/HW":
#         continue
#     for hand, files in hands.items():
#         for file_path, segments in files.items():
#             for seg_idx, metrics in enumerate(segments):
#                 # Check if metrics is a dictionary; if not, use it directly.
#                 if isinstance(metrics, dict):
#                     LDLJ_value = metrics.get("LDLJ", None)
#                 else:
#                     try:
#                         LDLJ_value = float(metrics)
#                     except Exception:
#                         LDLJ_value = None
#                 # Retrieve the corresponding SPARC value from reach_sparc_TW.
#                 SPARC_value = reach_sparc_TW[subj][hand][file_path][seg_idx]
#                 if LDLJ_value is not None and LDLJ_value > - 6.9 and SPARC_value < - 1.7:
#                     print(f"Subject: {subj}, Hand: {hand}, Trial: {file_path}, "
#                           f"Segment: {seg_idx + 1}, LDLJ: {LDLJ_value}, SPARC: {SPARC_value}")
                    
# #------
# LDLJ_value < -9.47 and SPARC_value < -1.83:
# Subject: 07/22/HW, Hand: left, Trial: /Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT10.csv, Segment: 9, LDLJ: -10.21185458332706, SPARC: -1.8387383284339676
# Subject: 07/22/HW, Hand: left, Trial: /Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT12.csv, Segment: 1, LDLJ: -10.178535340481318, SPARC: -1.9096222479686555
# Subject: 07/22/HW, Hand: left, Trial: /Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT22.csv, Segment: 1, LDLJ: -9.645157299012682, SPARC: -1.8839955467539733
# Subject: 07/22/HW, Hand: left, Trial: /Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT28.csv, Segment: 6, LDLJ: -9.530197460486503, SPARC: -2.0366285426437103

# LDLJ_value > -6.7 and SPARC_value > - 1.6
# Subject: 07/22/HW, Hand: left, Trial: /Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT48.csv, Segment: 8, LDLJ: -6.611903888306084, SPARC: -1.5918342671544263
# Subject: 07/22/HW, Hand: left, Trial: /Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT54.csv, Segment: 4, LDLJ: -6.674661640803834, SPARC: -1.5949645782246982
# Subject: 07/22/HW, Hand: left, Trial: /Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT58.csv, Segment: 8, LDLJ: -6.653237115495734, SPARC: -1.5943414266235232


# LDLJ_value < -9 and SPARC_value > - 1.7
# Subject: 07/22/HW, Hand: left, Trial: /Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT14.csv, Segment: 12, LDLJ: -9.419054055232264, SPARC: -1.6311243091836567
# Subject: 07/22/HW, Hand: left, Trial: /Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT30.csv, Segment: 15, LDLJ: -9.477670844156284, SPARC: -1.652000056460096

# LDLJ_value > - 6.9 and SPARC_value < - 1.75
# Subject: 07/22/HW, Hand: left, Trial: /Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv, Segment: 11, LDLJ: -6.848995873509286, SPARC: -1.7085276166416765
# Subject: 07/22/HW, Hand: left, Trial: /Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT36.csv, Segment: 11, LDLJ: -6.8736794658287765, SPARC: -1.7198262119826055
# Subject: 07/22/HW, Hand: left, Trial: /Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT54.csv, Segment: 3, LDLJ: -6.845864685424475, SPARC: -1.7056184678886497
# Subject: 07/22/HW, Hand: left, Trial: /Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT56.csv, Segment: 14, LDLJ: -6.823606683677508, SPARC: -1.7860389526213138

# Both_unsmooth = [
#  (-10.21185458332706, 
#   '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT10.csv', 
#   9),
#  (-10.178535340481318,
#   '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT12.csv',
#   1),
#  (-9.645157299012682,
#   '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT22.csv',
#   1),
#  (-9.530197460486503,
#   '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT28.csv',
#   6)]

# Both_smooth = [
#  (-6.611903888306084,
#   '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT48.csv',
#   8),
#  (-6.674661640803834,
#   '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT54.csv',
#   4),
#  (-6.653237115495734,
#   '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT58.csv',
#   8)]

# SPARC_smooth = [
#  (-9.419054055232264,
#   '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT14.csv',
#   12),
#  (-9.477670844156284,
#   '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT30.csv',
#   15)]

# LDLJ_smooth = [
#  (-6.8736794658287765,
#   '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT36.csv',
#   11),
#  (-6.848995873509286,
#   '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv',
#   11),
#  (-6.845864685424475,
#   '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT54.csv',
#   3),
#  (-6.823606683677508,
#   '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT56.csv',
#   14)]


Both  = [
     (-6.611903888306084,
  '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT48.csv',
  8),
 (-6.674661640803834,
  '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT54.csv',
  4),
 (-10.178535340481318,
  '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT12.csv',
  1),
 (-9.530197460486503,
  '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT28.csv',
  6)]

one  = [
 (-6.845864685424475,
  '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT54.csv',
  3),
 (-6.823606683677508,
  '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT56.csv',
  14),
 (-9.419054055232264,
  '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT14.csv',
  12),
 (-9.477670844156284,
  '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT30.csv',
  15)]


def plot_phase_combined_multi(subject, hand, trial_seg_list,
                              ballistic_phase, correction_phase, results,
                              reach_speed_segments, placement_location_colors,
                              marker="LFIN", frame_rate=200,
                              figsize=(15, 5), show_icon=True,
                              icon_path="/Users/yilinwu/Desktop/HandHoldBlock1.png", LDLJ_phase=reach_TW_metrics_ballistic_phase, SPARC_phase=reach_sparc_TW, smoothness_metric='LDLJ'):
    """
    Plots multiple trial+segment combinations as subplots.
    Each subplot has 3D trajectory (left) and kinematic signals (right).
    
    Parameters
    ----------
    trial_seg_list : list of tuples
        [(file_path, seg_index), (file_path, seg_index), ...]
    """
    n_plots = len(trial_seg_list)
    fig = plt.figure(figsize=(figsize[0], figsize[1] * n_plots))

    for plot_idx, (_, file_path, seg_index) in enumerate(trial_seg_list):
        seg_index = seg_index - 1  # convert to 0-based index
        # Data for this trial
        traj_data = results[subject][hand][1][file_path]['traj_data']
        coord_prefix = "LFIN_" if hand.lower() == "left" else "RFIN_"
        coord_x = np.array(traj_data[coord_prefix + "X"])
        coord_y = np.array(traj_data[coord_prefix + "Y"])
        coord_z = np.array(traj_data[coord_prefix + "Z"])

        # Signals
        signals_space = results[subject][hand][1][file_path]['traj_space'][marker]
        position_full, velocity_full, acceleration_full, jerk_full = map(np.array, signals_space)
        signals = [position_full, velocity_full, acceleration_full, jerk_full]
        signal_labels = ["Distance", "Velocity", "Acceleration", "Jerk"]
        signal_types = ["from origin", "magnitude", "magnitude", "magnitude"]
        units = ["mm", "mm/s", "mm/s²", "mm/s³"]
        colors_sig = {"full": "dimgray", "ballistic": "#0047ff", "correction": "#ffb800"}


        # Time axis
        reach_start, reach_end = reach_speed_segments[subject][hand][file_path][seg_index]
        time_full = np.arange(reach_end - reach_start) / frame_rate

        signal_hspace = 0.3  # keep signals close
        # define bigger vertical spacing between trials
        trial_gap = 0.05 # gap between trial blocks

        # Calculate top/bottom for this trial
        top = 1 - plot_idx * (1/n_plots)
        bottom = 1 - (plot_idx + 1) * (1/n_plots) + trial_gap

        gs = GridSpec(4, 5, figure=fig,
                    top=top,
                    bottom=bottom,
                    hspace=signal_hspace,  # small space between 4 signals
                    wspace=0)

        # --- 3D trajectory ---
        ax3d = fig.add_subplot(gs[:, :3], projection='3d')
        ballistic_start, ballistic_end_idx = ballistic_phase[subject][hand][file_path][seg_index]
        corrective_start, corrective_end = correction_phase[subject][hand][file_path][seg_index]

        ax3d.plot(coord_x[ballistic_start:ballistic_end_idx],
                  coord_y[ballistic_start:ballistic_end_idx],
                  coord_z[ballistic_start:ballistic_end_idx],
                  color=colors_sig["ballistic"], linewidth=3, label="Ballistic")
        ax3d.plot(coord_x[corrective_start:corrective_end],
                  coord_y[corrective_start:corrective_end],
                  coord_z[corrective_start:corrective_end],
                  color=colors_sig["correction"], linewidth=3, label="Correction")
        ax3d.set_title(f"SPARC: {SPARC_phase[subject][hand][file_path][seg_index]:.2f}\nLDLJ: {LDLJ_phase['reach_LDLJ'][subject][hand][file_path][seg_index]:.2f}",
        fontsize=16, pad=20, loc='right', y=0.85)


        if plot_idx == n_plots - 1:
            ax3d.set_xlabel("X (mm)", fontsize=16)
            ax3d.set_ylabel("Y (mm)", fontsize=16)
            ax3d.set_zlabel("Z (mm)", fontsize=16)

        # # both
        # if plot_idx == 0:
        #     ax3d.set_yticks([60, 75, 90])
        # if plot_idx == 1:
        #     ax3d.set_yticks([100, 110, 120])
        # if plot_idx == 2:
        #     ax3d.set_yticks([100, 110, 120])
        # if plot_idx == 3:
        #     ax3d.set_yticks([50, 60, 70])
        # ax3d.set_xticks([-250, 0, 250])
        # ax3d.set_zticks([800, 950, 1100])

        ax3d.set_xticks([-250, 0, 250])
        ax3d.set_xticklabels([-250, 0, 250])
        ax3d.set_yticks([-50, 50, 150])
        ax3d.set_yticklabels([-50, 50, 150])
        ax3d.set_zticks([800, 950, 1100])
        ax3d.set_zticklabels([800, 950, 1100])
        ax3d.set_xlim([-250, 250])
        ax3d.set_ylim([-50, 150])
        ax3d.set_zlim([800, 1100])
        ax3d.set_box_aspect([5, 2, 3])
        # # one
        # if plot_idx == 0:
        #     ax3d.set_yticks([100, 110, 120])
        # if plot_idx == 1:
        #     ax3d.set_yticks([-40, -15, 10])
        # if plot_idx == 2:
        #     ax3d.set_yticks([0, 20, 40])
        # if plot_idx == 3:
        #     ax3d.set_yticks([-40, -15, 10])
        # ax3d.set_xticks([-250, 0, 250])
        # ax3d.set_zticks([800, 950, 1100])



        # --- Signals ---
        for i, (sig, label, unit) in enumerate(zip(signals, signal_labels, units)):
            ax = fig.add_subplot(gs[i, 3:])
            ax.plot(time_full, sig[reach_start:reach_end], color=colors_sig["full"], linewidth=2)

            # highlight ballistic & corrective
            time_ballistic = (np.arange(ballistic_start, ballistic_end_idx) - reach_start) / frame_rate
            time_corrective = (np.arange(ballistic_end_idx, corrective_end) - reach_start) / frame_rate
            ax.plot(time_ballistic, sig[ballistic_start:ballistic_end_idx],
                color=colors_sig["ballistic"], linewidth=3)
            ax.plot(time_corrective, sig[ballistic_end_idx:corrective_end],
                color=colors_sig["correction"], linewidth=3)

            ax.set_ylabel(f"{label}\n{signal_types[i]}\n({unit})", fontsize=14, rotation=0)
            ax.yaxis.set_label_coords(-0.2, 0.1)
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xticks([0, 0.8, 1.6])
            if i == 3:
                ax.set_xticklabels([0, 0.8, 1.6], fontsize=14)
            else:
                ax.set_xticklabels([])            
            if i == 3 and plot_idx == n_plots - 1:
                ax.set_xlabel("Time (s)")
                ax.legend(["Full", "Ballistic", "Correction"], frameon=False, fontsize=18, ncol=3, loc=[-0.7, -1.1])

            # # both
            # if plot_idx == 0 and i == 0:
            #     # Place title between 3D plot (cols 0-2) and signal plots (cols 3-4)
            #     fig.text(0.57, top + 0.02, "---------------------------------Smooth in both LDLJ and SPARC---------------------------------", ha='center', fontsize=20)
            # if plot_idx == 2 and i == 0:
            #     fig.text(0.57, top + 0.02, "---------------------------------Unsmooth in both LDLJ and SPARC---------------------------------", ha='center', fontsize=20)

            # one
            if plot_idx == 0 and i == 0:
                # Place title between 3D plot (cols 0-2) and signal plots (cols 3-4)
                fig.text(0.57, top + 0.02, "---------------------------------Smooth in LDLJ not SPARC---------------------------------", ha='center', fontsize=20)
            if plot_idx == 2 and i == 0:
                fig.text(0.57, top + 0.02, "---------------------------------Smooth in SPARC not LDLJ---------------------------------", ha='center', fontsize=20)


    plt.tight_layout(pad=0.5)  # reduced padding
    plt.subplots_adjust(wspace=0.75, hspace=0.3)

    plt.show()

plot_phase_combined_multi(
    subject="07/22/HW",
    hand="left",
    trial_seg_list=one,
    ballistic_phase=ballistic_phase,
    correction_phase=correction_phase,
    results=results,
    reach_speed_segments=reach_speed_segments,
    placement_location_colors=placement_location_colors,
    marker="LFIN",
    frame_rate=200,
    figsize=(17, 4.5),  # width, height-per-row
    show_icon=False,
    icon_path="/Users/yilinwu/Desktop/HandHoldBlock1.png", LDLJ_phase=reach_TW_metrics_TW, SPARC_phase=reach_sparc_TW, smoothness_metric='LDLJ')






Both  = [
 (-6.674661640803834,
  '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT54.csv',
  4),
 (-9.530197460486503,
  '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT28.csv',
  6),
 (-6.823606683677508,
  '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT56.csv',
  14),  
 (-9.419054055232264,
  '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT14.csv',
  12)  
  ]

def plot_phase_combined_multi(subject, hand, trial_seg_list,
                              ballistic_phase, correction_phase, results,
                              reach_speed_segments, placement_location_colors,
                              marker="LFIN", frame_rate=200,
                              figsize=(15, 5), show_icon=True,
                              icon_path="/Users/yilinwu/Desktop/HandHoldBlock1.png", LDLJ_phase=reach_TW_metrics_ballistic_phase, SPARC_phase=reach_sparc_TW, smoothness_metric='LDLJ'):
    """
    Plots multiple trial+segment combinations as subplots.
    Each subplot has a 3D trajectory (left) and two kinematic signal plots (right): Velocity and Jerk.
    
    Parameters
    ----------
    trial_seg_list : list of tuples
        [(file_path, seg_index), (file_path, seg_index), ...]
    """
    n_plots = len(trial_seg_list)
    # Adjust figure height for 2 signal subplots per trial.
    fig = plt.figure(figsize=(figsize[0], figsize[1] * n_plots))
    
    for plot_idx, (_, file_path, seg_index) in enumerate(trial_seg_list):
        seg_index = seg_index - 1  # convert to 0-based index
        # Data for this trial
        traj_data = results[subject][hand][1][file_path]['traj_data']
        coord_prefix = "LFIN_" if hand.lower() == "left" else "RFIN_"
        coord_x = np.array(traj_data[coord_prefix + "X"])
        coord_y = np.array(traj_data[coord_prefix + "Y"])
        coord_z = np.array(traj_data[coord_prefix + "Z"])

        # Signals: remove Distance and Acceleration
        signals_space = results[subject][hand][1][file_path]['traj_space'][marker]
        # Only keep Velocity and Jerk
        _, velocity_full, _, jerk_full = map(np.array, signals_space)
        signals = [velocity_full, jerk_full]
        signal_labels = ["V(t)", "J(t)"]
        units = ["mm/s", "mm/s³"]
        colors_sig = {"full": "dimgray", "ballistic": "#0047ff", "correction": "#ffb800"}

        # Time axis
        reach_start, reach_end = reach_speed_segments[subject][hand][file_path][seg_index]
        time_full = np.arange(reach_end - reach_start) / frame_rate

        signal_hspace = 0.4  # vertical gap between signal subplots
        trial_gap = 0.06     # gap between trial blocks

        # Calculate top/bottom for this trial
        top = 1 - plot_idx * (1/n_plots)
        bottom = 1 - (plot_idx + 1) * (1/n_plots) + trial_gap

        # Use a grid with 2 rows (instead of 4); increase horizontal gap (wspace) slightly.
        gs = GridSpec(2, 5, figure=fig,
                    top=top,
                    bottom=bottom,
                    hspace=signal_hspace,
                    wspace=0.15)

        # --- 3D trajectory ---
        ax3d = fig.add_subplot(gs[:, :3], projection='3d')
        ballistic_start, ballistic_end_idx = ballistic_phase[subject][hand][file_path][seg_index]
        corrective_start, corrective_end = correction_phase[subject][hand][file_path][seg_index]

        ax3d.plot(coord_x[ballistic_start:ballistic_end_idx],
                    coord_y[ballistic_start:ballistic_end_idx],
                    coord_z[ballistic_start:ballistic_end_idx],
                    color=colors_sig["ballistic"], linewidth=3, label="Ballistic")
        ax3d.plot(coord_x[corrective_start:corrective_end],
                    coord_y[corrective_start:corrective_end],
                    coord_z[corrective_start:corrective_end],
                    color=colors_sig["correction"], linewidth=3, label="Correction")
        
        # if plot_idx == 0:
        #     sparc_color = "green"
        #     ldlj_color = "green"
        # elif plot_idx == 1:
        #     sparc_color = "red"
        #     ldlj_color = "red"
        # elif plot_idx == 2:
        #     sparc_color = "red"
        #     ldlj_color = "green"
        # elif plot_idx == 3:
        #     sparc_color = "green"
        #     ldlj_color = "red"
        # else:
        #     sparc_color = "black"
        #     ldlj_color = "black"

        # # Removed mathtext color formatting to avoid parsing errors.
        # sparc_str = f"SPARC: {SPARC_phase[subject][hand][file_path][seg_index]:.2f}"
        # ldlj_str = f"LDLJ: {LDLJ_phase['reach_LDLJ'][subject][hand][file_path][seg_index]:.2f}"

        # # Add colored text manually using 2D text on 3D axes
        # ax3d.text2D(
        #     1.05, 0.8, sparc_str, transform=ax3d.transAxes,
        #     fontsize=14, color=sparc_color, ha='right'
        # )
        # ax3d.text2D(
        #     1.05, 0.73, ldlj_str, transform=ax3d.transAxes,
        #     fontsize=14, color=ldlj_color, ha='right'
        # )
        ax3d.set_title(f"SPARC: {SPARC_phase[subject][hand][file_path][seg_index]:.2f}\nLDLJ: {LDLJ_phase['reach_LDLJ'][subject][hand][file_path][seg_index]:.2f}",
                        fontsize=14, loc='right', y=0.8)

        if plot_idx == n_plots - 1:
            ax3d.set_xlabel("X (mm)", fontsize=14)
            ax3d.set_ylabel("Y (mm)", fontsize=14)
            ax3d.set_zlabel("Z (mm)", fontsize=14)

        ax3d.set_xticks([-250, 0, 250])
        ax3d.set_xticklabels([-250, 0, 250])
        ax3d.set_yticks([-50, 50, 150])
        ax3d.set_yticklabels([-50, 50, 150])
        ax3d.set_zticks([800, 950, 1100])
        ax3d.set_zticklabels([800, 950, 1100])
        ax3d.set_xlim([-250, 250])
        ax3d.set_ylim([-50, 150])
        ax3d.set_zlim([800, 1100])
        ax3d.set_box_aspect([5, 2, 3])


        import matplotlib.ticker as ticker

        for i, (sig, label, unit) in enumerate(zip(signals, signal_labels, units)):
            ax = fig.add_subplot(gs[i, 3:])
            ax.plot(time_full, sig[reach_start:reach_end], color=colors_sig["full"], linewidth=2)

            # Highlight ballistic & corrective segments
            time_ballistic = (np.arange(ballistic_start, ballistic_end_idx) - reach_start) / frame_rate
            time_corrective = (np.arange(ballistic_end_idx, corrective_end) - reach_start) / frame_rate
            ax.plot(time_ballistic, sig[ballistic_start:ballistic_end_idx],
                    color=colors_sig["ballistic"], linewidth=3)
            ax.plot(time_corrective, sig[ballistic_end_idx:corrective_end],
                    color=colors_sig["correction"], linewidth=3)

            ax.set_ylabel(f"{label}\n({unit})", fontsize=14)
            ax.yaxis.set_label_coords(-0.1, 0.5)
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xticks([0, 0.8, 1.6])
            if i == 1:
                ax.set_xticklabels([0, 0.8, 1.6], fontsize=12)
            else:
                ax.set_xticklabels([])

            # --- Force formatter ---
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(formatter)

            # --- Fixed ranges ---
            if i == 0:   # velocity
                ax.set_ylim(0, 1500)     # always 0-100
            elif i == 1: # jerk
                ax.set_ylim(0, 400000)  # always 0-400000

            if i == 1 and plot_idx == n_plots - 1:
                ax.set_xlabel("Time (s)")
                ax.legend(["Full", "Ballistic", "Correction"], frameon=False,
                        fontsize=14, ncol=3, loc=[-0.48, -0.8])

            #  # both:
            # if plot_idx == 0 and i == 0:
            #     fig.text(0.57, top + 0.02, "----------------------Smooth in both LDLJ and SPARC----------------------", ha='center', fontsize=14)
            # if plot_idx == 2 and i == 0:
            #     fig.text(0.57, top + 0.02, "---------------------Unsmooth in both LDLJ and SPARC---------------------", ha='center', fontsize=14)

            # one: add annotations if needed
            if plot_idx == 0 and i == 0:
                fig.text(0.68, top + 0.025, "------------------Smooth in both LDLJ and SPARC------------------", ha='center', fontsize=14)
                fig.text(0.25, top + 0.025, "(A)", ha='center', fontsize=20)

            if plot_idx == 1 and i == 0:
                fig.text(0.68, top + 0.025, "------------------Unsmooth in both LDLJ and SPARC------------------", ha='center', fontsize=14)
                fig.text(0.25, top + 0.025, "(B)", ha='center', fontsize=20)
            if plot_idx == 2 and i == 0:
                fig.text(0.68, top + 0.025, "------------------Smooth in LDLJ not SPARC------------------", ha='center', fontsize=14)
                fig.text(0.25, top + 0.025, "(C)", ha='center', fontsize=20)
            if plot_idx == 3 and i == 0:
                fig.text(0.68, top+ 0.025, "------------------Smooth in SPARC not LDLJ------------------", ha='center', fontsize=14)
                fig.text(0.25, top + 0.025, "(D)", ha='center', fontsize=20)

    plt.tight_layout()
    plt.subplots_adjust()

    plt.show()

plot_phase_combined_multi(
    subject="07/22/HW",
    hand="left",
    trial_seg_list=Both,
    ballistic_phase=ballistic_phase,
    correction_phase=correction_phase,
    results=results,
    reach_speed_segments=reach_speed_segments,
    placement_location_colors=placement_location_colors,
    marker="LFIN",
    frame_rate=200,
    figsize=(10.5, 3.2),
    show_icon=False,
    icon_path="/Users/yilinwu/Desktop/HandHoldBlock1.png", LDLJ_phase=reach_TW_metrics_TW, SPARC_phase=reach_sparc_TW, smoothness_metric='LDLJ')






from matplotlib.gridspec import GridSpec
import numpy as np
from statsmodels.stats.multitest import multipletests
def plot_phase_combined_multi(subject, hand, trial_seg_list,
                              ballistic_phase, correction_phase, results,
                              reach_speed_segments, placement_location_colors,
                              marker="LFIN", frame_rate=200,
                              figsize=(15, 5), show_icon=True,
                              icon_path="/Users/yilinwu/Desktop/HandHoldBlock1.png", LDLJ_phase=reach_TW_metrics_TW, SPARC_phase=reach_sparc_TW, smoothness_metric='LDLJ'):
    n_plots = len(trial_seg_list)
    fig = plt.figure(figsize=(figsize[0], figsize[1] * n_plots))
    
    for plot_idx, (_, file_path, seg_index) in enumerate(trial_seg_list):
        seg_index = seg_index - 1  
        traj_data = results[subject][hand][1][file_path]['traj_data']
        coord_prefix = "LFIN_" if hand.lower() == "left" else "RFIN_"
        coord_x = np.array(traj_data[coord_prefix + "X"])
        coord_y = np.array(traj_data[coord_prefix + "Y"])
        coord_z = np.array(traj_data[coord_prefix + "Z"])

        signals_space = results[subject][hand][1][file_path]['traj_space'][marker]
        _, velocity_full, _, jerk_full = map(np.array, signals_space)
        signals = [velocity_full, jerk_full]
        signal_labels = ["V(t)", "J(t)"]
        units = ["mm/s", "mm/s³"]
        colors_sig = {"full": "dimgray", "ballistic": "#0047ff", "correction": "#ffb800"}

        reach_start, reach_end = reach_speed_segments[subject][hand][file_path][seg_index]
        time_full = np.arange(reach_end - reach_start) / frame_rate

        signal_hspace = 0.4
        trial_gap = 0.07  # Increased gap between each section

        top = 1 - plot_idx * (1/n_plots)
        bottom = 1 - (plot_idx + 1) * (1/n_plots) + trial_gap

        # Expand grid to 7 columns with customized width ratios:
        # Columns 0-2 for 3D trajectory, 3-4 (vt/jt) get 1.25x width relative to 5-6 (frequency)
        gs = GridSpec(2, 7, figure=fig,
                      top=top,
                      bottom=bottom,
                      hspace=signal_hspace,
                      wspace=0.3,
                      width_ratios=[1, 1, 1, 1.25, 1.25, 1, 1])

        # --- 3D trajectory ---
        ax3d = fig.add_subplot(gs[:, :3], projection='3d')
        ballistic_start, ballistic_end_idx = ballistic_phase[subject][hand][file_path][seg_index]
        corrective_start, corrective_end = correction_phase[subject][hand][file_path][seg_index]

        ax3d.plot(coord_x[ballistic_start:ballistic_end_idx],
                  coord_y[ballistic_start:ballistic_end_idx],
                  coord_z[ballistic_start:ballistic_end_idx],
                  color=colors_sig["ballistic"], linewidth=3, label="Ballistic")
        ax3d.plot(coord_x[corrective_start:corrective_end],
                  coord_y[corrective_start:corrective_end],
                  coord_z[corrective_start:corrective_end],
                  color=colors_sig["correction"], linewidth=3, label="Correction")

        if plot_idx == n_plots - 1:
            ax3d.set_xlabel("X (mm)", fontsize=12, labelpad=1)
            ax3d.set_ylabel("Y (mm)", fontsize=12, labelpad=1)
            ax3d.set_zlabel("Z (mm)", fontsize=12, labelpad=1)

        ax3d.tick_params(axis='x', pad=2)
        ax3d.tick_params(axis='y', pad=2)
        ax3d.tick_params(axis='z', pad=2)

        ax3d.set_xticks([-250, 0, 250])
        ax3d.set_xticklabels([-250, 0, 250])
        ax3d.set_yticks([-50, 50, 150])
        ax3d.set_yticklabels([-50, 50, 150])
        ax3d.set_zticks([800, 950, 1100])
        ax3d.set_zticklabels([800, 950, 1100])
        ax3d.set_xlim([-250, 250])
        ax3d.set_ylim([-50, 150])
        ax3d.set_zlim([800, 1100])
        ax3d.set_box_aspect([5, 2, 3])

        import matplotlib.ticker as ticker

        # --- Velocity and Jerk ---
        for i, (sig, label, unit) in enumerate(zip(signals, signal_labels, units)):
            ax = fig.add_subplot(gs[i, 3:5])   # velocity & jerk in columns 3-4 (wider)
            ax.plot(time_full, sig[reach_start:reach_end], color=colors_sig["full"], linewidth=2)

            time_ballistic = (np.arange(ballistic_start, ballistic_end_idx) - reach_start) / frame_rate
            time_corrective = (np.arange(ballistic_end_idx, corrective_end) - reach_start) / frame_rate
            ax.plot(time_ballistic, sig[ballistic_start:ballistic_end_idx],
                    color=colors_sig["ballistic"], linewidth=3)
            ax.plot(time_corrective, sig[ballistic_end_idx:corrective_end],
                    color=colors_sig["correction"], linewidth=3)

            ax.set_ylabel(f"{label}({unit})", fontsize=12)
            ax.yaxis.set_label_coords(-0.13, 0.5)

            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xticks([0, 0.8, 1.6])
            if i == 1:
                ax.set_xticklabels([0, 0.8, 1.6], fontsize=12)
            else:
                ax.set_xticklabels([])

            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(formatter)

            if i == 0:   # velocity axis limits
                ax.set_ylim(0, 1500)
            elif i == 1: # jerk axis limits
                ax.set_ylim(0, 400000)

            if i == 1 and plot_idx == n_plots - 1:
                ax.set_xlabel("Time (s)")
                ax.legend(["Full", "Ballistic", "Correction"], frameon=False,
                          fontsize=14, ncol=3, loc=[-1, -0.8])

            # Increase gap between sections by shifting annotation text upward
            if plot_idx == 0 and i == 0:
                fig.text(0.68, top +0.02, "------------------Smooth in both LDLJ and SPARC------------------", ha='center', fontsize=14)
                fig.text(0.2, top + 0.02, "(A)", ha='center', fontsize=20)
            if plot_idx == 1 and i == 0:
                fig.text(0.68, top + 0.02, "------------------Unsmooth in both LDLJ and SPARC------------------", ha='center', fontsize=14)
                fig.text(0.2, top + 0.02, "(B)", ha='center', fontsize=20)
            if plot_idx == 2 and i == 0:
                fig.text(0.68, top + 0.02, "------------------Smooth in LDLJ not SPARC------------------", ha='center', fontsize=14)
                fig.text(0.2, top + 0.02, "(C)", ha='center', fontsize=20)
            if plot_idx == 3 and i == 0:
                fig.text(0.68, top + 0.02, "------------------Smooth in SPARC not LDLJ------------------", ha='center', fontsize=14)
                fig.text(0.2, top + 0.02, "(D)", ha='center', fontsize=20)

        # --- Frequency spectrum (aligned with Velocity row only) ---
        speed_segment = signals[0][reach_start:reach_end]  # velocity segment
        sparc_value, (f, Mf), (f_sel, Mf_sel) = sparc_Normalizing(speed_segment, fs=frame_rate)

        # Reduced frequency plot width using column span 5 only (instead of 2 full columns)
        ax_freq = fig.add_subplot(gs[0, 5:7])
        ax_freq.plot(f_sel, Mf_sel, color="black", linewidth=3, label=f"SPARC={sparc_value:.2f}")
        ax_freq.set_xlabel("Frequency (Hz)", fontsize=12)
        ax_freq.set_ylabel("Magnitude", fontsize=12)
        ax_freq.set_xlim(0, 6)
        ax_freq.set_ylim(0, 1.1)
        ax_freq.spines["top"].set_visible(False)
        ax_freq.spines["right"].set_visible(False)
        ax_freq.grid(False)
        if plot_idx == n_plots - 1:
            ax_freq.legend(["Fourier transform spectrum of\nthe velocity signal (ballistic + correction)"],
                          frameon=False, fontsize=14, loc=[-0.6, -2.3])
        ax_freq.annotate(
            f"SPARC: {SPARC_phase[subject][hand][file_path][seg_index]:.2f}\n"
            f"LDLJ: {LDLJ_phase['reach_LDLJ'][subject][hand][file_path][seg_index]:.2f}",
            xy=(0.5, -1.25), xycoords='axes fraction',
            ha='center', fontsize=14
        )

    plt.tight_layout()
    plt.show()

plot_phase_combined_multi(
    subject="07/22/HW",
    hand="left",
    trial_seg_list=Both,
    ballistic_phase=ballistic_phase,
    correction_phase=correction_phase,
    results=results,
    reach_speed_segments=reach_speed_segments,
    placement_location_colors=placement_location_colors,
    marker="LFIN",
    frame_rate=200,
    figsize=(13, 3.2),
    show_icon=False,
    icon_path="/Users/yilinwu/Desktop/HandHoldBlock1.png", LDLJ_phase=reach_TW_metrics_TW, SPARC_phase=reach_sparc_TW, smoothness_metric='LDLJ')



def sparc_Normalizing(movement, fs, padlevel=4, fc=20.0, amp_th=0.05):
    """
    Calculates the smoothness of the given movement profile using the modified
    spectral arc length metric. Before computing the metric, the trajectory is
    normalized in amplitude (space) so that the start-to-target displacement has unit length.

    Parameters
    ----------
    movement : np.array
               The array containing the movement trajectory. It can be multidimensional.
    fs       : float
               The sampling frequency of the data.
    padlevel : integer, optional
               Amount of zero padding to be done to the movement data for estimating
               the spectral arc length. [default = 4]
    fc       : float, optional
               The cutoff frequency for calculating the spectral arc length metric. [default = 20.]
    amp_th   : float, optional
               The amplitude threshold used for determining the cutoff frequency up to 
               which the spectral arc length is estimated. [default = 0.05]

    Returns
    -------
    sal      : float
               The spectral arc length estimate of the movement's smoothness.
    (f, Mf)  : tuple of two np.arrays
               The frequency (f) and the normalized magnitude spectrum (Mf) computed from the fft.
    (f_sel, Mf_sel) : tuple of two np.arrays
                      The portion of the spectrum that is selected for spectral arc length calculation.

    Notes
    -----
    This modified spectral arc length metric has been tested only for discrete movements.
    
    Examples
    --------
    >>> t = np.linspace(0, 1, 200)
    >>> move = np.exp(-5 * (t - 0.5)**2)
    >>> sal, _, _ = sparc(move, fs=200.)
    >>> '%.5f' % sal
    '-1.41403'
    """
    # Normalize movement: scale trajectory to have unit start-to-end displacement.
    # Ensure movement is a numpy array.
    movement = np.array(movement)
    amp = np.linalg.norm(movement[-1] - movement[0])
    if amp != 0:
        movement = movement / amp

    # Number of zeros to be padded.
    nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

    # Frequency vector.
    f = np.arange(0, fs, fs / nfft)
    # Normalized magnitude spectrum.
    Mf = abs(np.fft.fft(movement, nfft))
    Mf = Mf / max(Mf)

    # Select spectrum within the cutoff frequency fc.
    fc_inx = ((f <= fc) * 1).nonzero()
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]

    # Apply amplitude threshold for further selection.
    inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
    fc_inx = range(inx[0], inx[-1] + 1)
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]

    # Calculate spectral arc length.
    new_sal = -sum(np.sqrt(pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) +
                           pow(np.diff(Mf_sel), 2)))
    
    # 1. Time domain plot (original movement)
    t = np.arange(len(movement)) / fs
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 2, 1)
    plt.plot(t, movement)
    plt.title("1. Speed Profile (Time Domain)")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed")
    plt.grid(False)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 2. FFT and Normalization
    plt.subplot(3, 2, 2)
    plt.plot(f[:nfft // 2], Mf[:nfft // 2])
    plt.title("2. Full Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Magnitude")
    plt.grid(False)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    # 3. Cutoff Frequency
    plt.subplot(3, 2, 3)
    plt.plot(f_sel, Mf_sel)
    plt.title("3. Spectrum Below Cutoff (fc = 20 Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(False)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    # 4. Amplitude Threshold Filtering
    amp_mask = Mf_sel >= amp_th
    f_cut = f_sel[amp_mask]
    Mf_cut = Mf_sel[amp_mask]

    plt.subplot(3, 2, 4)
    plt.plot(f_sel, Mf_sel, color='lightgray', label='All under 20Hz')
    plt.plot(f_cut, Mf_cut, color='blue', label='Above threshold')
    plt.axhline(y=amp_th, color='red', linestyle='--', label='Threshold')
    plt.title("4. After Amplitude Thresholding")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(False)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 5. Spectral Arc Length Calculation
    df = np.diff(f_cut) / (f_cut[-1] - f_cut[0])  # Normalize frequency
    # dM = np.diff(Mf_cut)
    # arc_length = -np.sum(np.sqrt(df ** 2 + dM ** 2))

    plt.subplot(3, 2, 5)
    plt.plot(f_cut, Mf_cut, marker='o')
    for i in range(len(df)):
        plt.plot([f_cut[i], f_cut[i+1]], [Mf_cut[i], Mf_cut[i+1]], 'k--')
    plt.title("5. Arc Segments Used in SPARC")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(False)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.subplot(3, 2, 6)
    plt.text(0.1, 0.5, f"SPARC Value:\n{new_sal:.4f}", fontsize=20)
    plt.axis('off')
    plt.grid(False)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

    


    return new_sal, (f, Mf), (f_sel, Mf_sel)

# unsmooth
speed = results['07/22/HW']['left'][1]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT10.csv']['traj_space']['LFIN'][1]
segments = test_windows_7['07/22/HW']['left']['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT10.csv'][8]
speed_segment = speed[segments[0]:segments[1]]
sparc_value, _,_ = sparc_Normalizing(speed_segment, fs=200)


speed = results['07/22/HW']['left'][1]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT10.csv']['traj_space']['LFIN'][1]
segments = ballistic_phase['07/22/HW']['left']['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT10.csv'][8]
speed_segment = speed[segments[0]:segments[1]]
sparc_value, _,_ = sparc_Normalizing(speed_segment, fs=200)


speed = results['07/22/HW']['left'][1]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT10.csv']['traj_space']['LFIN'][1]
segments = correction_phase['07/22/HW']['left']['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT10.csv'][8]
speed_segment = speed[segments[0]:segments[1]]
sparc_value, _,_ = sparc_Normalizing(speed_segment, fs=200)




# smooth
speed = results['07/22/HW']['left'][1]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT48.csv']['traj_space']['LFIN'][1]
segments = test_windows_7['07/22/HW']['left']['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT48.csv'][7]
speed_segment = speed[segments[0]:segments[1]]
sparc_value, _,_ = sparc_Normalizing(speed_segment, fs=200)

speed = results['07/22/HW']['left'][1]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT48.csv']['traj_space']['LFIN'][1]
segments = ballistic_phase['07/22/HW']['left']['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT48.csv'][7]
speed_segment = speed[segments[0]:segments[1]]
sparc_value, _,_ = sparc_Normalizing(speed_segment, fs=200)

speed = results['07/22/HW']['left'][1]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT48.csv']['traj_space']['LFIN'][1]
segments = correction_phase['07/22/HW']['left']['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT48.csv'][7]
speed_segment = speed[segments[0]:segments[1]]
sparc_value, _,_ = sparc_Normalizing(speed_segment, fs=200)






def sparc_Normalizing(movement, fs, padlevel=4, fc=20.0, amp_th=0.05):
    """
    Calculates the smoothness of the given movement profile using the modified
    spectral arc length metric. Before computing the metric, the trajectory is
    normalized in amplitude (space) so that the start-to-target displacement has unit length.

    Parameters
    ----------
    movement : np.array
               The array containing the movement trajectory. It can be multidimensional.
    fs       : float
               The sampling frequency of the data.
    padlevel : integer, optional
               Amount of zero padding to be done to the movement data for estimating
               the spectral arc length. [default = 4]
    fc       : float, optional
               The cutoff frequency for calculating the spectral arc length metric. [default = 20.]
    amp_th   : float, optional
               The amplitude threshold used for determining the cutoff frequency up to 
               which the spectral arc length is estimated. [default = 0.05]

    Returns
    -------
    sal      : float
               The spectral arc length estimate of the movement's smoothness.
    (f, Mf)  : tuple of two np.arrays
               The frequency (f) and the normalized magnitude spectrum (Mf) computed from the fft.
    (f_sel, Mf_sel) : tuple of two np.arrays
                      The portion of the spectrum that is selected for spectral arc length calculation.

    Notes
    -----
    This modified spectral arc length metric has been tested only for discrete movements.
    
    Examples
    --------
    >>> t = np.linspace(0, 1, 200)
    >>> move = np.exp(-5 * (t - 0.5)**2)
    >>> sal, _, _ = sparc(move, fs=200.)
    >>> '%.5f' % sal
    '-1.41403'
    """
    # Normalize movement: scale trajectory to have unit start-to-end displacement.
    # Ensure movement is a numpy array.
    movement = np.array(movement)
    amp = np.linalg.norm(movement[-1] - movement[0])
    if amp != 0:
        movement = movement / amp

    # Number of zeros to be padded.
    nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

    # Frequency vector.
    f = np.arange(0, fs, fs / nfft)
    # Normalized magnitude spectrum.
    Mf = abs(np.fft.fft(movement, nfft))
    Mf = Mf / max(Mf)

    # Select spectrum within the cutoff frequency fc.
    fc_inx = ((f <= fc) * 1).nonzero()
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]

    # Apply amplitude threshold for further selection.
    inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
    fc_inx = range(inx[0], inx[-1] + 1)
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]

    # Calculate spectral arc length.
    new_sal = -sum(np.sqrt(pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) +
                           pow(np.diff(Mf_sel), 2)))
    
    # 1. Time domain plot (original movement)
    t = np.arange(len(movement)) / fs
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 2, 1)
    plt.plot(t, movement)
    plt.title("1. Speed Profile (Time Domain)")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed")
    plt.grid(False)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 2. FFT and Normalization
    plt.subplot(3, 2, 2)
    plt.plot(f[:nfft // 2], Mf[:nfft // 2])
    plt.title("2. Full Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Magnitude")
    plt.grid(False)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    # 3. Cutoff Frequency
    plt.subplot(3, 2, 3)
    plt.plot(f_sel, Mf_sel)
    plt.title("3. Spectrum Below Cutoff (fc = 20 Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(False)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    # 4. Amplitude Threshold Filtering
    amp_mask = Mf_sel >= amp_th
    f_cut = f_sel[amp_mask]
    Mf_cut = Mf_sel[amp_mask]

    plt.subplot(3, 2, 4)
    plt.plot(f_sel, Mf_sel, color='lightgray', label='All under 20Hz')
    plt.plot(f_cut, Mf_cut, color='blue', label='Above threshold')
    plt.axhline(y=amp_th, color='red', linestyle='--', label='Threshold')
    plt.title("4. After Amplitude Thresholding")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(False)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 5. Spectral Arc Length Calculation
    df = np.diff(f_cut) / (f_cut[-1] - f_cut[0])  # Normalize frequency
    # dM = np.diff(Mf_cut)
    # arc_length = -np.sum(np.sqrt(df ** 2 + dM ** 2))

    plt.subplot(3, 2, 5)
    plt.plot(f_cut, Mf_cut, marker='o')
    for i in range(len(df)):
        plt.plot([f_cut[i], f_cut[i+1]], [Mf_cut[i], Mf_cut[i+1]], 'k--')
    plt.title("5. Arc Segments Used in SPARC")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(False)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.subplot(3, 2, 6)
    plt.text(0.1, 0.5, f"SPARC Value:\n{new_sal:.4f}", fontsize=20)
    plt.axis('off')
    plt.grid(False)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

    


    return new_sal, (f, Mf), (f_sel, Mf_sel)


speed = results['07/22/HW']['left'][1]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT54.csv']['traj_space']['LFIN'][1]
segments = test_windows_7['07/22/HW']['left']['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT54.csv'][3]
speed_segment = speed[segments[0]:segments[1]]
sparc_value, _,_ = sparc_Normalizing(speed_segment, fs=200)

speed = results['07/22/HW']['left'][1]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT28.csv']['traj_space']['LFIN'][1]
segments = test_windows_7['07/22/HW']['left']['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT28.csv'][5]
speed_segment = speed[segments[0]:segments[1]]
sparc_value, _,_ = sparc_Normalizing(speed_segment, fs=200)

speed = results['07/22/HW']['left'][1]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT56.csv']['traj_space']['LFIN'][1]
segments = test_windows_7['07/22/HW']['left']['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT56.csv'][13]
speed_segment = speed[segments[0]:segments[1]]
sparc_value, _,_ = sparc_Normalizing(speed_segment, fs=200)

speed = results['07/22/HW']['left'][1]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT14.csv']['traj_space']['LFIN'][1]
segments = test_windows_7['07/22/HW']['left']['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT14.csv'][11]
speed_segment = speed[segments[0]:segments[1]]
sparc_value, _,_ = sparc_Normalizing(speed_segment, fs=200)









import matplotlib.pyplot as plt




# ----------- PLOTTING (1 column x 4 rows) -----------------
# Collect results in a list for plotting
datasets = [
    ('HW_tBBT54.csv', 1, 3),
    ('HW_tBBT28.csv', 1, 5),
    ('HW_tBBT56.csv', 1, 13),
    ('HW_tBBT14.csv', 1, 11)
]

fig, axes = plt.subplots(4, 1, figsize=(4, 6.4))

for ax, (fname, idx, seg_idx) in zip(axes, datasets):
    speed = results['07/22/HW']['left'][idx][f'/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/{fname}']['traj_space']['LFIN'][1]
    segments = test_windows_7['07/22/HW']['left'][f'/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/{fname}'][seg_idx]
    speed_segment = speed[segments[0]:segments[1]]

    sparc_value, (f, Mf), (f_sel, Mf_sel) = sparc_Normalizing(speed_segment, fs=200)

    # Only plot "Cutoff Frequency"
    ax.plot(f_sel, Mf_sel, label=f"SPARC={sparc_value:.3f}", color='black', linewidth=3)
    # ax.set_title(f"{fname} (fc=20 Hz)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(False)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    # ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()

def sparc_Normalizing(movement, fs, padlevel=4, fc=20.0, amp_th=0.05):
    """
    Computes the SPARC (Spectral Arc Length) smoothness metric without plotting.
    Returns the SPARC value and spectrum components for later plotting.
    """
    movement = np.array(movement)
    amp = np.linalg.norm(movement[-1] - movement[0])
    if amp != 0:
        movement = movement / amp

    nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

    f = np.arange(0, fs, fs / nfft)
    Mf = abs(np.fft.fft(movement, nfft))
    Mf = Mf / max(Mf)

    # Select spectrum within cutoff
    fc_inx = (f <= fc).nonzero()
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]

    # Thresholding
    inx = (Mf_sel >= amp_th).nonzero()[0]
    fc_inx = range(inx[0], inx[-1] + 1)
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]

    # SPARC calculation
    sal = -np.sum(np.sqrt((np.diff(f_sel) / (f_sel[-1] - f_sel[0]))**2 +
                          (np.diff(Mf_sel))**2))

    return sal, (f, Mf), (f_sel, Mf_sel)


for fname, idx, seg_idx in datasets:
    fig, ax = plt.subplots(figsize=(2.5, 1.9))
    speed = results['07/22/HW']['left'][idx][f'/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/{fname}']['traj_space']['LFIN'][1]
    segments = test_windows_7['07/22/HW']['left'][f'/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/{fname}'][seg_idx]
    speed_segment = speed[segments[0]:segments[1]]

    sparc_value, (f, Mf), (f_sel, Mf_sel) = sparc_Normalizing(speed_segment, fs=200)

    ax.plot(f_sel, Mf_sel, label=f"SPARC={sparc_value:.3f}", color='black', linewidth=3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(False)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()



# # Function to compute and visualize LDLJ per segment
# def compute_ldlj(jerk, speed, x, y, z, segments, fs=200):
#     """
#     Computes the LDLJ for a given segment and provides visualizations.

#     Parameters
#     ----------
#     jerk (array-like): Jerk profile (1D array, in mm/s³).
#     speed (array-like): Speed profile (in mm/s).
#     x, y, z (array-like): 3D trajectory components (in mm).
#     segments (tuple): (start, end) indices for the segment.
#     fs (float): Sampling frequency (default 200 Hz).

#     Returns
#     -------
#     LDLJ (float): The computed LDLJ value.
#     """
#     import matplotlib.pyplot as plt

#     # --- Step 1: Extract segment jerk and time vectors ---
#     start, end = segments
#     jerk_segment = np.array(jerk[start:end])
#     duration = (end - start) / fs
#     t_orig = np.linspace(0, duration, num=len(jerk_segment))

#     # --- Step 2: Resample (warp) jerk to standardized 101 points ---
#     target_samples = 101
#     t_std = np.linspace(0, duration, num=target_samples)
#     warped_jerk = np.interp(t_std, t_orig, jerk_segment)

#     # --- Step 3: Compute integral of squared jerk ---
#     jerk_squared_integral = np.trapezoid(warped_jerk**2, t_std)

#     # --- Step 4: Get peak speed during this segment ---
#     vpeak = np.max(speed[start:end])

#     # --- Step 5: Dimensionless jerk and LDLJ ---
#     dimensionless_jerk = (duration**3 / (vpeak**2)) * jerk_squared_integral
#     LDLJ = -math.log(abs(dimensionless_jerk))

#     print(f"Duration: {duration:.3f} s")
#     print(f"Peak speed: {vpeak:.3f} mm/s")
#     print(f"Integral of squared jerk: {jerk_squared_integral:.3e}")
#     print(f"Dimensionless jerk: {dimensionless_jerk:.3e}")
#     print(f"LDLJ: {LDLJ:.3f}")

#     # --- Visualization ---
#     fig = plt.figure(figsize=(14, 10))

#     # 1. 3D trajectory with segment highlighted
#     ax1 = fig.add_subplot(221, projection='3d')
#     ax1.plot(x, y, z, label='Full trajectory', color='gray')
#     ax1.plot(x[start:end], y[start:end], z[start:end], linewidth=3, color='blue')
#     ax1.set_title('3D Trajectory (Segment Highlighted)')
#     ax1.set_xlabel('X (mm)')
#     ax1.set_ylabel('Y (mm)')
#     ax1.set_zlabel('Z (mm)')

#     # 2. Speed vs Time
#     t = np.arange(len(speed)) / fs
#     ax2 = fig.add_subplot(222)
#     # ax2.plot(t, speed, label='Speed Profile', color='gray')
#     ax2.plot(t[start:end], speed[start:end], linewidth=3, color='blue')
#     vpeak_idx = start + np.argmax(speed[start:end])
#     ax2.scatter(t[vpeak_idx], speed[vpeak_idx], c='red', label='Peak Speed')
#     ax2.set_title('Speed vs Time (mm/s)')
#     ax2.set_xlabel('Time (s)')
#     ax2.set_ylabel('Speed (mm/s)')
#     ax2.legend()
#     ax2.grid(False)
#     ax2.spines["top"].set_visible(False)
#     ax2.spines["right"].set_visible(False)

#     # 3. Jerk vs Time (Segment Emphasized)
#     ax3 = fig.add_subplot(223)
#     # ax3.plot(t, jerk, label='Jerk Profile', color='gray')
#     ax3.plot(t[start:end], jerk[start:end], linewidth=3, color='blue')
#     ax3.set_title('Jerk vs Time (mm/s³)')
#     ax3.set_xlabel('Time (s)')
#     ax3.set_ylabel('Jerk (mm/s³)')
#     ax3.grid(False)
#     ax3.spines["top"].set_visible(False)
#     ax3.spines["right"].set_visible(False)

#     # 4. Warped Jerk and Squared Jerk
#     ax4 = fig.add_subplot(224)
#     ax4.plot(t_std, warped_jerk, label='Warped Jerk', color='blue')
#     ax4.plot(t_std, warped_jerk**2, linestyle='--', label='Jerk Squared', color='orange')
#     ax4.fill_between(t_std, warped_jerk**2, alpha=0.3, color='orange')
#     ax4.set_title('Warped Jerk & Squared Jerk')
#     ax4.set_xlabel('Standardized Time (s)')
#     ax4.set_ylabel('Warped Jerk (mm/s³) \n& Jerk Squared (mm/s³)²')
#     ax4.legend()
#     ax4.text(0.95, 0.95, f"LDLJ = {LDLJ:.3f}", transform=ax4.transAxes, fontsize=20, ha='right', va='top')
#     ax4.grid(False)
#     ax4.spines["top"].set_visible(False)
#     ax4.spines["right"].set_visible(False)

#     plt.tight_layout()
#     plt.show()

#     return LDLJ

# # smooth
# jerk = results['07/22/HW']['left'][1]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT48.csv']['traj_space']['LFIN'][3]
# speed = results['07/22/HW']['left'][1]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT48.csv']['traj_space']['LFIN'][1]
# x = results['07/22/HW']['left'][1]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT48.csv']['traj_data']['LFIN_X']
# y = results['07/22/HW']['left'][1]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT48.csv']['traj_data']['LFIN_Y']
# z = results['07/22/HW']['left'][1]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT48.csv']['traj_data']['LFIN_Z']
# segments = test_windows_7['07/22/HW']['left']['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT48.csv'][7]

# # unsmooth
# jerk = results['07/22/HW']['left'][1]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT10.csv']['traj_space']['LFIN'][3]
# speed = results['07/22/HW']['left'][1]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT10.csv']['traj_space']['LFIN'][1]
# x = results['07/22/HW']['left'][1]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT10.csv']['traj_data']['LFIN_X']
# y = results['07/22/HW']['left'][1]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT10.csv']['traj_data']['LFIN_Y']
# z = results['07/22/HW']['left'][1]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT10.csv']['traj_data']['LFIN_Z']
# segments = test_windows_7['07/22/HW']['left']['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT10.csv'][8]

# # Compute LDLJ for the selected segment
# ldlj_value = compute_ldlj(jerk, speed, x, y, z, segments, fs=200)




