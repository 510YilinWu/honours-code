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


# --- Global Styles & Colors ---
# -----------------------
# Global Styles & Colors
# -----------------------
def setup_global_styles():
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Global colors and style constants.
    variable_colors = {
        "ballistic_LDLJ": "#1f78b4",   # dark blue
        "ballistic_sparc": "#e66101",  # dark orange
        "corrective_LDLJ": "#a6cee3",    # light blue
        "corrective_sparc": "#fdb863",   # light orange

        "durations (s)": "#4daf4a",     # green
        "distance (mm)": "#984ea3",     # purple
        "MotorAcuity": "#e7298a"        # magenta/pink
    }
    LINE_WIDTH = 2
    MARKER_SIZE = 6

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })

    # Helper functions for figure sizing and tick formatting.
    def mm_to_inch(width_mm, height_mm):
        return (width_mm / 25.4, height_mm / 25.4)

    def min_mid_max_ticks(ax, axis='both'):
        if axis in ('x', 'both'):
            xmin, xmax = ax.get_xlim()
            ax.set_xticks([xmin, (xmin + xmax) / 2, xmax])
        if axis in ('y', 'both'):
            ymin, ymax = ax.get_ylim()
            ax.set_yticks([ymin, (ymin + ymax) / 2, ymax])

    def format_axis(ax, x_is_categorical=False):
        if x_is_categorical:
            min_mid_max_ticks(ax, axis='y')
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        else:
            min_mid_max_ticks(ax, axis='both')
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    def smart_legend(fig, axes):
        if isinstance(axes, (list, np.ndarray)):
            handles, labels = [], []
            for ax in axes:
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)
            by_label = dict(zip(labels, handles))
            fig.legend(
                by_label.values(), by_label.keys(),
                loc="lower center",
                bbox_to_anchor=(0.5, -0.05),
                ncol=3,
                frameon=False
            )
        else:
            axes.legend(frameon=False, loc="upper right", fontsize=11)

    # Additional global style variables.
    hand_colors = {
        "dominant": "#7570b3",          # purple
        "non_dominant": "#66c2a5"         # teal
    }
    # Added order variable as requested.
    order = ["Non-dominant", "Dominant"]
    location_colors = [cm.tab20(i / 20) for i in range(16)]
    corr_cmap = cm.get_cmap("coolwarm")  # correlation colormap

    # Line and marker defaults.
    MARKER_SHAPE = 'o'
    LINE_WIDTH_DEFAULT = 2
    MARKER_SIZE_DEFAULT = 6
    figsize_mm = (90, 70)

    # Return all variables and functions in a dictionary.
    return {
        "variable_colors": variable_colors,
        "LINE_WIDTH": LINE_WIDTH,
        "MARKER_SIZE": MARKER_SIZE,
        "mm_to_inch": mm_to_inch,
        "min_mid_max_ticks": min_mid_max_ticks,
        "format_axis": format_axis,
        "smart_legend": smart_legend,
        "hand_colors": hand_colors,
        "order": order,
        "location_colors": location_colors,
        "corr_cmap": corr_cmap,
        "MARKER_SHAPE": MARKER_SHAPE,
        "LINE_WIDTH_DEFAULT": LINE_WIDTH_DEFAULT,
        "MARKER_SIZE_DEFAULT": MARKER_SIZE_DEFAULT,
        "figsize_mm": figsize_mm
    }

globals_dict = setup_global_styles()



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

# -------------------------------------------------------------------------------------------------------------------
# --- LOAD ALL SUBJECTS' tBBT ERROR FROM IMAGE, SAVE AS pickle file---
Block_Distance = utils4.load_selected_subject_errors(All_dates, DataProcess_folder)

# --- LOAD RESULTS FROM PICKLE FILE "processed_results.pkl" ---
results = utils1.load_selected_subject_results(All_dates, DataProcess_folder)



# # -------------------------------------------------------------------------------------------------------------------
# Calculate RMS reprojection error statistics
utils4.compute_rms_reprojection_error_stats()
# # -------------------------------------------------------------------------------------------------------------------
# Load sBBTResult from CSV into a DataFrame and compute right and left hand scores
sBBTResult = utils8.load_and_compute_sbbt_result()
# Swap and rename sBBTResult scores for specific subjects
sBBTResult = utils8.swap_and_rename_sbbt_result(sBBTResult)
sBBTResult_stats = utils8.compute_sbbt_result_stats(sBBTResult)
# utils8.plot_sbbt_boxplot(sBBTResult)
sBBT_combine_stat = utils8.analyze_sbbt_results(sBBTResult)


def plot_sbbt_boxplot(sBBTResult, globals_dict):
    """
    Plot a boxplot of sBBT scores by hand (Non-dominant vs Dominant) using global style settings.
    It overlays a swarmplot and draws dashed lines connecting each subject's pair of scores.
    
    Parameters:
        sBBTResult: DataFrame - indexed by subjects with columns "non_dominant" and "dominant".
        globals_dict: dict - Dictionary containing global style settings.
    """
    import matplotlib.pyplot as plt

    # Create figure using global figure size in millimeters converted to inches.
    fig, ax = plt.subplots(figsize=globals_dict["mm_to_inch"](*globals_dict["figsize_mm"]))

    # Prepare a DataFrame for sBBT scores per subject.
    df_scores = pd.DataFrame({
        "Subject": sBBTResult.index,
        "Non-dominant": sBBTResult["non_dominant"],
        "Dominant": sBBTResult["dominant"]
    })

    # Melt the DataFrame to long format.
    df_melt = df_scores.melt(id_vars="Subject", var_name="Hand", value_name="sBBT Score")

    # Use the global order and hand colors for consistent styling.
    order = globals_dict["order"]
    palette = [globals_dict["hand_colors"]["non_dominant"], globals_dict["hand_colors"]["dominant"]]

    # Plot the boxplot using the global palette and order.
    sns.boxplot(x="Hand", y="sBBT Score", data=df_melt, palette=palette, order=order, ax=ax)
    sns.swarmplot(
        x="Hand", y="sBBT Score", data=df_melt,
        color="black", size=globals_dict["MARKER_SIZE"],
        alpha=0.8, order=order, ax=ax
    )

    # Set tick parameters and label fonts using global style settings from plt.rcParams.
    ax.tick_params(axis='both', which='major', labelsize=plt.rcParams["xtick.labelsize"])
    ax.set_xlabel("Hand", fontsize=plt.rcParams["axes.labelsize"])
    ax.set_ylabel("sBBT Score", fontsize=plt.rcParams["axes.labelsize"])

    # Apply the global axis formatter.
    globals_dict["format_axis"](ax, x_is_categorical=True)

    # Add a smart legend using global settings.
    globals_dict["smart_legend"](fig, ax)

    plt.tight_layout()
    plt.show()

plot_sbbt_boxplot(sBBTResult, globals_dict)

# # -------------------------------------------------------------------------------------------------------------------
# # Load Motor Experiences from CSV into a dictionary
# MotorExperiences = utils7.load_motor_experiences("/Users/yilinwu/Desktop/Yilin-Honours/MotorExperience.csv")
# # Calculate demographic variables from MotorExperiences
# utils7.display_motor_experiences_stats(MotorExperiences)
# # Update MotorExperiences with weighted scores
# utils7.update_overall_h_total_weighted(MotorExperiences)
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

# -------------------------------------------------------------------------------------------------------------------
test_windows_7 = utils9.compute_test_window_7(results, reach_speed_segments, reach_metrics)
# -------------------------------------------------------------------------------------------------------------------
# ===============================
# Function 1: Calculate key events
# ===============================
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

# ===============================
# Function 2: Plot results
# ===============================
# def plot_trajectory_analysis(results_dict):
#     x = results_dict["x"]
#     y = results_dict["y"]
#     z = results_dict["z"]
#     v = results_dict["v"]
#     acc_along_vel = results_dict["acc_along_vel"]
#     # valid_peaks = results_dict["valid_peaks"]
#     global_peak_idx = results_dict["global_peak_idx"]
#     sub_movement_idx = results_dict["sub_movement_idx"]
#     end_of_ballistic_idx = results_dict["end_of_ballistic_idx"]
#     dt = results_dict["dt"]

#     fig = plt.figure(figsize=(14,6))

#     # 3D trajectory
#     ax3d = fig.add_subplot(1, 2, 1, projection='3d')
#     ax3d.plot(x, y, z, label='Trajectory', color='blue')

#     # for i in valid_peaks:
#     #     ax3d.scatter(x[i], y[i], z[i], color='pink', s=40, label='Candidate Peak' if i==valid_peaks[0] else "")
#     if global_peak_idx is not None:
#         ax3d.scatter(x[global_peak_idx], y[global_peak_idx], z[global_peak_idx],
#                      color='red', s=70, label='Global Peak')
#     if sub_movement_idx is not None:
#         ax3d.scatter(x[sub_movement_idx], y[sub_movement_idx], z[sub_movement_idx],
#                      color='orange', s=70, label='Sub-Movement End')
#     if end_of_ballistic_idx is not None:
#         ax3d.scatter(x[end_of_ballistic_idx], y[end_of_ballistic_idx], z[end_of_ballistic_idx],
#                      color='green', s=70, label='Ballistic Phase End')

#     ax3d.set_xlabel('X (mm)')
#     ax3d.set_ylabel('Y (mm)')
#     ax3d.set_zlabel('Z (mm)')
#     ax3d.set_title('3D Trajectory with Key Events')
#     ax3d.legend()
#     ax3d.grid(True)

#     # 2D velocity and acceleration
#     ax2d = fig.add_subplot(1, 2, 2)
#     time = np.arange(len(v)) * dt
#     ax2d.plot(time, v, label='Speed (mm/s)', color='blue')
#     ax2d.plot(time, acc_along_vel, label='Acc along Vel (mm/s²)', color='purple', alpha=0.7)

#     # for i in valid_peaks:
#     #     ax2d.axvline(i*dt, color='pink', linestyle='--', alpha=0.5)
#     if global_peak_idx is not None:
#         ax2d.axvline(global_peak_idx*dt, color='red', linestyle='--', label='Global Peak')
#     if sub_movement_idx is not None:
#         ax2d.axvline(sub_movement_idx*dt, color='orange', linestyle='--', label='Sub-Movement End')
#     if end_of_ballistic_idx is not None:
#         ax2d.axvline(end_of_ballistic_idx*dt, color='green', linestyle='--', label='Ballistic Phase End')

#     ax2d.set_xlabel('Time (s)')
#     ax2d.set_ylabel('Velocity / Acceleration')
#     ax2d.set_title('Speed and Acceleration along Velocity')
#     ax2d.legend()
#     ax2d.grid(True)

#     plt.tight_layout()
#     plt.show()

def plot_trajectory_analysis(results_dict):
    x = results_dict["x"]
    y = results_dict["y"]
    z = results_dict["z"]
    v = results_dict["v"]
    acc_along_vel = results_dict["acc_along_vel"]
    # global_peak_idx and sub_movement_idx removed
    end_of_ballistic_idx = results_dict["end_of_ballistic_idx"]
    dt = results_dict["dt"]

    fig = plt.figure(figsize=(11, 4))
    line_width = 4  # Increase line width
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.05)

    # 3D trajectory
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    if end_of_ballistic_idx is not None:
        ax3d.plot(x[:end_of_ballistic_idx + 1], y[:end_of_ballistic_idx + 1], z[:end_of_ballistic_idx + 1],
                  label='Trajectory (Ballistic)', color='blue', linestyle='-', linewidth=line_width)
        ax3d.plot(x[end_of_ballistic_idx:], y[end_of_ballistic_idx:], z[end_of_ballistic_idx:],
                  label='Trajectory (Corrective)', color='blue', linestyle='--', linewidth=line_width)
    else:
        ax3d.plot(x, y, z, label='Trajectory', color='blue', linewidth=line_width)

    if end_of_ballistic_idx is not None:
        ax3d.scatter(x[end_of_ballistic_idx], y[end_of_ballistic_idx], z[end_of_ballistic_idx],
                     color='cyan', s=100, label='Ballistic Phase End')

    ax3d.set_xlabel('X (mm)')
    ax3d.set_ylabel('Y (mm)')
    ax3d.set_zlabel('Z (mm)')
    ax3d.legend(frameon=False, loc='upper left', fontsize=10)
    ax3d.grid(True)
    ax3d.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax3d.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax3d.zaxis.set_major_locator(MaxNLocator(nbins=5))

    # 2D speed and acceleration
    ax2d_v = fig.add_subplot(gs[0, 1])
    ax2d_a = ax2d_v.twinx()
    time = np.arange(len(v)) * dt

    if end_of_ballistic_idx is not None:
        ax2d_v.plot(time[:end_of_ballistic_idx + 1], v[:end_of_ballistic_idx + 1],
                    label='Speed (Ballistic)', color='green', linestyle='-', linewidth=line_width)
        ax2d_v.plot(time[end_of_ballistic_idx:], v[end_of_ballistic_idx:],
                    label='Speed (Corrective)', color='green', linestyle='--', linewidth=line_width)
        ax2d_a.plot(time[:end_of_ballistic_idx + 1], acc_along_vel[:end_of_ballistic_idx + 1],
                    label='Acc along Vel (Ballistic)', color='orange', alpha=0.7, linestyle='-', linewidth=line_width)
        ax2d_a.plot(time[end_of_ballistic_idx:], acc_along_vel[end_of_ballistic_idx:],
                    label='Acc along Vel (Corrective)', color='orange', alpha=0.7, linestyle='--', linewidth=line_width)
    else:
        ax2d_v.plot(time, v, label='Speed', color='green', linewidth=line_width)
        ax2d_a.plot(time, acc_along_vel, label='Acc along Vel', color='orange', alpha=0.7, linewidth=line_width)

    if end_of_ballistic_idx is not None:
        ax2d_v.axvline(end_of_ballistic_idx * dt, color='cyan', linestyle='--', linewidth=line_width, label='Ballistic Phase End')

    ax2d_v.set_xlabel('Time (s)')
    ax2d_v.set_ylabel('Velocity (mm/s)', color='green')
    ax2d_a.set_ylabel('Acceleration along velocity (mm/s²)', color='orange')
    ax2d_v.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax2d_v.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax2d_a.yaxis.set_major_locator(MaxNLocator(nbins=5))

    lines_v, labels_v = ax2d_v.get_legend_handles_labels()
    lines_a, labels_a = ax2d_a.get_legend_handles_labels()
    # ax2d_v.legend(lines_v + lines_a, labels_v + labels_a, loc='upper right', frameon=False, fontsize=10)
    ax2d_v.grid(False)
    ax2d_a.grid(False)
    for spine in ['left', 'right']:
        ax2d_v.spines[spine].set_visible(True)
    for spine in ['left', 'right']:
        ax2d_a.spines[spine].set_visible(True)

    plt.tight_layout()
    plt.show()

# ===============================
# Parameters and trajectory data
# ===============================
def run_trajectory_analysis(subject, hand, file_path, seg_index):
    traj_data = results[subject][hand][1][file_path]['traj_data']
    # Choose the coordinate prefix based on hand if needed; here we use LFIN_ for left hand.
    coord_prefix = "LFIN_" if hand.lower() == "left" else "RFIN_"
    start_idx, end_idx = test_windows_7[subject][hand][file_path][seg_index]
    results_dict = analyze_trajectory(traj_data, coord_prefix, start_idx, end_idx)
    plot_trajectory_analysis(results_dict)

# run_trajectory_analysis("07/22/HW", "left", '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv', 2)

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
# -------------------------------------------------------------------------------------------------------------------

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

# -------------------------------------------------------------------------------------------------------------------
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
# --- GENERATE PLACEMENT LOCATION COLORS ---
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

placement_location_colors = generate_placement_colors(show_plot=True)
# -------------------------------------------------------------------------------------------------------------------
# Show how phase segmentation works on one example
def plot_traj_segment(subject, hand, file_path, seg_index,
                      reach_speed_segments, ballistic_phase, correction_phase, results,
                      marker="LFIN", frame_rate=200, figsize=(12, 9)):
    import numpy as np
    import matplotlib.pyplot as plt

    # Get reach-speed segment boundaries
    reach_start, reach_end = reach_speed_segments[subject][hand][file_path][seg_index]
    time_full = np.arange(reach_end - reach_start) / frame_rate

    # Full trajectory signals
    traj_space = results[subject][hand][1][file_path]['traj_space'][marker]
    position_full, velocity_full, acceleration_full, jerk_full = map(np.array, traj_space)

    # Phase boundaries
    ballistic_start, ballistic_end_idx = ballistic_phase[subject][hand][file_path][seg_index]
    corrective_start, corrective_end = correction_phase[subject][hand][file_path][seg_index]

    # Time axes
    ballistic_end_time = (ballistic_end_idx - reach_start) / frame_rate

    # Define units for each signal
    units = {
        "Position": "mm",
        "Velocity": "mm/s",
        "Acceleration": "mm/s²",
        "Jerk": "mm/s³"
    }

    fig, axs = plt.subplots(4, 1, figsize=figsize, sharex=True)
    signals = {"Position": position_full, "Velocity": velocity_full,
               "Acceleration": acceleration_full, "Jerk": jerk_full}
    colors = {"full": "dimgray", "ballistic": "blue", "corrective": "red"}

    for i, (label, signal) in enumerate(signals.items()):
        ax = axs[i]

        # Full reach trajectory
        ax.plot(time_full, signal[reach_start:reach_end], color=colors["full"],
                linestyle="--", linewidth=2)
        # Ballistic phase
        time_ballistic_full = (np.arange(ballistic_start, ballistic_end_idx) - reach_start) / frame_rate
        ax.plot(time_ballistic_full, signal[ballistic_start:ballistic_end_idx],
                color=colors["ballistic"], linestyle="-", linewidth=2)
        # Correction phase
        time_corrective_full = (np.arange(ballistic_end_idx, corrective_end) - reach_start) / frame_rate
        ax.plot(time_corrective_full, signal[ballistic_end_idx:corrective_end],
                color=colors["corrective"], linestyle="--", linewidth=2)
        # Ballistic end marker
        ax.scatter(ballistic_end_time, signal[ballistic_end_idx], color="cyan", s=100, marker="D")

        # Align the y-axis label and add unit using a dictionary lookup
        ax.set_ylabel(f"{label} \n({units[label]})", labelpad=10, fontsize=10)
        # Optionally, set fixed label coordinates for alignment (horizontally centered next to y-axis)
        ax.yaxis.set_label_coords(-0.15, 0.5)

        # Add legend on the first subplot only
        if i == 0:
            ax.legend(["Full Reach", "Ballistic", "Correction", "Ballistic End"],
                      frameon=False, fontsize=10)

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

plot_traj_segment("07/22/HW", "left",
                  '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv',
                  seg_index=2,
                  reach_speed_segments=reach_speed_segments,
                  ballistic_phase=ballistic_phase,
                  correction_phase=correction_phase,
                  results=results,
                  marker="LFIN",
                  frame_rate=200,
                  figsize=(6, 5))
# -------------------------------------------------------------------------------------------------------------------

# Show how phase segmentation works on multiple segments in 3D
def plot_phase_trajectory(subject, hand, file_path, seg_indices, ballistic_phase, correction_phase, results, figsize=(8, 6)):
    """
    Plots the ballistic and corrective phase trajectories of one or multiple trial segments in 3D.

    Parameters:
        subject (str): Subject identifier.
        hand (str): Hand identifier (e.g., "left" or "right").
        file_path (str): Path to the CSV file containing trajectory data.
        seg_indices (int or list): Segment index (or list of segment indices) to plot.
        ballistic_phase (dict): Nested dictionary of ballistic phase time windows.
        correction_phase (dict): Nested dictionary of corrective phase time windows.
        results (dict): The results dictionary containing trajectory data.
        figsize (tuple): Figure size, default is (8, 6).
    """
    
    # Ensure seg_indices is a list
    if not isinstance(seg_indices, list):
        seg_indices = [seg_indices]

    # Retrieve trajectory data using the appropriate marker
    traj_data = results[subject][hand][1][file_path]['traj_data']
    coord_prefix = "LFIN_" if hand.lower() == "left" else "RFIN_"
    coord_x = np.array(traj_data[coord_prefix + "X"])
    coord_y = np.array(traj_data[coord_prefix + "Y"])
    coord_z = np.array(traj_data[coord_prefix + "Z"])

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Select colors corresponding to each segment index
    colors = [placement_location_colors[idx] for idx in seg_indices]

    for idx, seg_index in enumerate(seg_indices):
        # Get start/end indices from the computed time windows for the chosen segment
        ballistic_start, ballistic_end_idx = ballistic_phase[subject][hand][file_path][seg_index]
        corrective_start, corrective_end = correction_phase[subject][hand][file_path][seg_index]

        # Plot ballistic phase trajectory
        ax.plot(
            coord_x[ballistic_start:ballistic_end_idx],
            coord_y[ballistic_start:ballistic_end_idx],
            coord_z[ballistic_start:ballistic_end_idx],
            color=colors[idx], linewidth=2,
            label=f'Placement {seg_index+1} Ballistic Phase' if idx == 0 else None
        )

        # Plot corrective phase trajectory
        ax.plot(
            coord_x[corrective_start:corrective_end],
            coord_y[corrective_start:corrective_end],
            coord_z[corrective_start:corrective_end],
            color=colors[idx], linestyle='--', linewidth=2,
            label=f'Placement {seg_index+1} Correction Phase' if idx == 0 else None
        )

        # Mark the start and end points of the overall segment
        ax.scatter(
            coord_x[ballistic_start], coord_y[ballistic_start], coord_z[ballistic_start],
            color='green', s=50, marker='o'
        )
        ax.scatter(
            coord_x[corrective_end - 1], coord_y[corrective_end - 1], coord_z[corrective_end - 1],
            color='red', s=50, marker='^'
        )

    # Label axes, set custom ticks, and add legend
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_xticks([-250, 0, 250])
    ax.set_yticks([-70, 0, 140])
    ax.set_zticks([800, 1200])
    ax.legend(frameon=False, loc=[0.5, 0.8])
    plt.tight_layout()
    plt.show()
plot_phase_trajectory(
    subject="07/22/HW",
    hand="left",
    file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv',
    seg_indices=[2],
    ballistic_phase=ballistic_phase,
    correction_phase=correction_phase,
    results=results,
    figsize=(7, 5)
)

# -------------------------------------------------------------------------------------------------------------------

def plot_traj_segment(subject, hand, file_path, seg_index,
                      ballistic_phase, correction_phase, results,
                      marker="LFIN", frame_rate=200, figsize=(12, 9)):
    """
    Plot Position, Velocity, Acceleration, and Jerk for a given trajectory segment,
    highlighting the ballistic end index with a time axis starting from 0.
    
    Parameters:
        figsize (tuple): Figure size as (width, height) in inches.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # Get ballistic and corrective phase boundaries
    ballistic_start, ballistic_end_idx = ballistic_phase[subject][hand][file_path][seg_index]
    corrective_start, corrective_end = correction_phase[subject][hand][file_path][seg_index]

    # Use segment from ballistic start to corrective end
    start_idx = ballistic_start
    end_idx = corrective_end

    # Retrieve trajectory signals
    traj_space = results[subject][hand][1][file_path]['traj_space'][marker]
    position_full = np.array(traj_space[0])
    velocity_full = np.array(traj_space[1])
    acceleration_full = np.array(traj_space[2])
    jerk_full = np.array(traj_space[3])

    # Offset time so that the segment starts at time 0
    segment_length = end_idx - start_idx
    time_values = np.arange(segment_length) / frame_rate
    ballistic_end_time = (ballistic_end_idx - start_idx) / frame_rate

    # Create subplots with adjustable figsize
    fig, axs = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # Plot Position
    axs[0].plot(time_values, position_full[start_idx:end_idx], color='blue', linewidth=2, label='Position')
    axs[0].scatter(ballistic_end_time, position_full[ballistic_end_idx], color='cyan', s=100, marker='D', label='Ballistic End')
    axs[0].set_ylabel("Position")
    axs[0].legend(frameon=False)

    # Plot Velocity
    axs[1].plot(time_values, velocity_full[start_idx:end_idx], color='green', linewidth=2, label='Velocity')
    axs[1].scatter(ballistic_end_time, velocity_full[ballistic_end_idx], color='cyan', s=100, marker='D', label='Ballistic End')
    axs[1].set_ylabel("Velocity")
    axs[1].legend(frameon=False)

    # Plot Acceleration
    axs[2].plot(time_values, acceleration_full[start_idx:end_idx], color='orange', linewidth=2, label='Acceleration')
    axs[2].scatter(ballistic_end_time, acceleration_full[ballistic_end_idx], color='cyan', s=100, marker='D', label='Ballistic End')
    axs[2].set_ylabel("Acceleration")
    axs[2].legend(frameon=False)

    # Plot Jerk
    axs[3].plot(time_values, jerk_full[start_idx:end_idx], color='red', linewidth=2, label='Jerk')
    axs[3].scatter(ballistic_end_time, jerk_full[ballistic_end_idx], color='cyan', s=100, marker='D', label='Ballistic End')
    axs[3].set_ylabel("Jerk")
    axs[3].set_xlabel("Time (s)")
    axs[3].legend(frameon=False)

    plt.tight_layout()
    plt.show()
plot_traj_segment("07/22/HW", "left",
                  '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT66.csv',
                  seg_index=15,
                  ballistic_phase=ballistic_phase,
                  correction_phase=correction_phase,
                  results=results,
                  marker="LFIN",
                  frame_rate=200,
                  figsize=(7, 5))

# -------------------------------------------------------------------------------------------------------------------
def overlay_all_segments(subject, hand, ballistic_phase, correction_phase, results, marker="LFIN", frame_rate=200, figsize=(7, 5)):
    """
    Overlay all trajectory segments across all trials for a given subject and hand.
    Segments are aligned to t=0 and x-axis is converted to time (seconds) using the given frame rate.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Create 4 subplots to display each metric, 4:3 aspect ratio
    fig, axs = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # Define a faded light grey color with transparency for trajectories
    faded_color = "lightgrey"
    plot_alpha = 0.3

    # Enhanced marker settings for ballistic end:
    ballistic_marker_style = {
        "color": "cyan",
        "s": 30,
        "marker": "o",
        "edgecolor": "blue",
        "linewidth": 0.5,
        "zorder": 10
    }

    # Loop over all file paths for the subject and hand
    for file_path in results[subject][hand][1]:
        traj_space = results[subject][hand][1][file_path]['traj_space'][marker]
        pos_full = np.array(traj_space[0])
        vel_full = np.array(traj_space[1])
        acc_full = np.array(traj_space[2])
        jerk_full = np.array(traj_space[3])

        num_segments = len(ballistic_phase[subject][hand][file_path])
        for seg_index in range(num_segments):
            bs, be = ballistic_phase[subject][hand][file_path][seg_index]
            cs, ce = correction_phase[subject][hand][file_path][seg_index]

            # Convert frame indices to time in seconds
            duration = ce - bs
            time_values = np.arange(duration) / frame_rate
            adjusted_be = (be - bs) / frame_rate  # ballistic end in seconds

            # Plot each signal
            axs[0].plot(time_values, pos_full[bs:ce], color=faded_color, alpha=plot_alpha)
            axs[0].scatter(adjusted_be, pos_full[be], **ballistic_marker_style)

            axs[1].plot(time_values, vel_full[bs:ce], color=faded_color, alpha=plot_alpha)
            axs[1].scatter(adjusted_be, vel_full[be], **ballistic_marker_style)

            axs[2].plot(time_values, acc_full[bs:ce], color=faded_color, alpha=plot_alpha)
            axs[2].scatter(adjusted_be, acc_full[be], **ballistic_marker_style)

            axs[3].plot(time_values, jerk_full[bs:ce], color=faded_color, alpha=plot_alpha)
            axs[3].scatter(adjusted_be, jerk_full[be], **ballistic_marker_style)

    # Set labels for each subplot
    axs[0].set_ylabel("Position")
    axs[1].set_ylabel("Velocity")
    axs[2].set_ylabel("Acceleration")
    axs[3].set_ylabel("Jerk")
    axs[3].set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
overlay_all_segments("07/22/HW", "left",
                     ballistic_phase=ballistic_phase,
                     correction_phase=correction_phase,
                     results=results,
                     marker="LFIN",
                     frame_rate=200,
                     figsize=(7, 6))

# -------------------------------------------------------------------------------------------------------------------
def plot_median_segment_with_phase_lines(subject, hand, ballistic_phase, correction_phase, results,
                                         marker="LFIN", frame_rate=200, figsize=(12, 9)):
    import numpy as np
    import matplotlib.pyplot as plt

    # Collect all segments
    pos_all, vel_all, acc_all, jerk_all = [], [], [], []
    ballistic_ends = []

    for file_path in results[subject][hand][1]:
        traj_space = results[subject][hand][1][file_path]['traj_space'][marker]
        pos_full, vel_full, acc_full, jerk_full = map(np.array, traj_space)

        num_segments = len(ballistic_phase[subject][hand][file_path])
        for seg_index in range(num_segments):
            bs, be = ballistic_phase[subject][hand][file_path][seg_index]
            cs, ce = correction_phase[subject][hand][file_path][seg_index]
            seg_length = ce - bs
            if seg_length > 0:
                pos_all.append(pos_full[bs:ce])
                vel_all.append(vel_full[bs:ce])
                acc_all.append(acc_full[bs:ce])
                jerk_all.append(jerk_full[bs:ce])
                ballistic_ends.append(be - bs)  # relative ballistic end within segment

    if len(pos_all) == 0:
        print("No segments found across trials.")
        return

    # Determine common length across segments
    common_length = min(len(seg) for seg in pos_all)
    time_values = np.arange(common_length) / frame_rate

    # Function to compute median and 5th–95th percentile range
    def aggregate_signal(signal_list):
        truncated = np.array([seg[:common_length] for seg in signal_list])
        median_signal = np.median(truncated, axis=0)
        lower_signal = np.percentile(truncated, 5, axis=0)
        upper_signal = np.percentile(truncated, 95, axis=0)
        return median_signal, lower_signal, upper_signal

    pos_med, pos_lower, pos_upper = aggregate_signal(pos_all)
    vel_med, vel_lower, vel_upper = aggregate_signal(vel_all)
    acc_med, acc_lower, acc_upper = aggregate_signal(acc_all)
    jerk_med, jerk_lower, jerk_upper = aggregate_signal(jerk_all)

    # Define signals and units
    signals = {"Position": (pos_med, pos_lower, pos_upper),
               "Velocity": (vel_med, vel_lower, vel_upper),
               "Acceleration": (acc_med, acc_lower, acc_upper),
               "Jerk": (jerk_med, jerk_lower, jerk_upper)}

    units = {"Position": "mm", "Velocity": "mm/s", "Acceleration": "mm/s²", "Jerk": "mm/s³"}
    colors = {"median_ballistic": "blue", "median_correction": "red", "range": "grey"}

    # Median ballistic end across all segments
    ballistic_end_time = np.median(ballistic_ends) / frame_rate
    ballistic_end_idx = int(ballistic_end_time * frame_rate)

    fig, axs = plt.subplots(4, 1, figsize=figsize, sharex=True)
    for i, (label, (median_signal, lower_signal, upper_signal)) in enumerate(signals.items()):
        ax = axs[i]
        # Shaded envelope for 5th–95th percentile
        ax.fill_between(time_values, lower_signal, upper_signal, color=colors["range"], alpha=0.3, label="5th–95th percentile")
        # Median ballistic phase
        ax.plot(time_values[:ballistic_end_idx], median_signal[:ballistic_end_idx],
                color=colors["median_ballistic"], linewidth=2, label="Median Ballistic")
        # Median corrective phase
        ax.plot(time_values[ballistic_end_idx:], median_signal[ballistic_end_idx:],
                color=colors["median_correction"], linewidth=2, label="Median Correction")
        # Ballistic end marker
        ax.scatter(ballistic_end_time, median_signal[ballistic_end_idx],
                   color="cyan", s=100, marker="o", label="Ballistic End")

        # Labels
        ax.set_ylabel(f"{label} \n({units[label]})", labelpad=10, fontsize=10)
        ax.yaxis.set_label_coords(-0.15, 0.5)
        # Legend only on first subplot to avoid clutter
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            filtered = [(h, l) for h, l in zip(handles, labels) if "5th–95th percentile" not in l]
            if filtered:
                handles, labels = zip(*filtered)
            else:
                handles, labels = [], []
            ax.legend(handles, labels, frameon=False, fontsize=10, loc=[0.7, 0.7])

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
plot_median_segment_with_phase_lines("07/22/HW", "left",
                                     ballistic_phase=ballistic_phase,
                                     correction_phase=correction_phase,
                                     results=results,
                                     marker="LFIN",
                                     frame_rate=200,
                                     figsize=(6, 5))
# -------------------------------------------------------------------------------------------------------------------

def overlay_all_segments(subject, hand, ballistic_phase, correction_phase, results, marker="LFIN", frame_rate=200, figsize=(8, 6)):
    """
    For a given subject and hand, this function aggregates all trial segments across the files,
    resamples/truncates them to the minimum common length, and computes the median trajectory
    along with an envelope computed by excluding outlier values (using the 5th and 95th percentiles)
    to represent the range of motion. It then plots Position, Velocity, Acceleration, and Jerk signals
    across all trials.
    
    Parameters:
        subject (str): Subject identifier (e.g., "07/22/HW").
        hand (str): Which hand ("left" or "right").
        ballistic_phase (dict): Nested dictionary with ballistic phase time windows.
        correction_phase (dict): Nested dictionary with corrective phase time windows.
        results (dict): Nested dictionary containing trajectory data.
        marker (str): Marker used to select trajectory signals ("LFIN" or "RFIN").
    """
    import matplotlib.pyplot as plt

    # Initialize lists to store segments from all files
    pos_all, vel_all, acc_all, jerk_all = [], [], [], []

    # Loop over all file paths for the subject and hand
    for file_path in results[subject][hand][1]:
        traj_space = results[subject][hand][1][file_path]['traj_space'][marker]
        pos_full = np.array(traj_space[0])
        vel_full = np.array(traj_space[1])
        acc_full = np.array(traj_space[2])
        jerk_full = np.array(traj_space[3])

        num_segments = len(ballistic_phase[subject][hand][file_path])
        for seg_index in range(num_segments):
            bs, be = ballistic_phase[subject][hand][file_path][seg_index]
            cs, ce = correction_phase[subject][hand][file_path][seg_index]
            seg_length = ce - bs
            if seg_length > 0:
                pos_all.append(pos_full[bs:ce])
                vel_all.append(vel_full[bs:ce])
                acc_all.append(acc_full[bs:ce])
                jerk_all.append(jerk_full[bs:ce])

    if len(pos_all) == 0:
        print("No segments found across trials.")
        return

    # Determine the common (minimum) length across all segments
    common_length = min([len(seg) for seg in pos_all])
    # Convert common frame indices to time (s) for a 200 Hz sampling rate
    time_values = np.arange(common_length) / frame_rate

    # Function to truncate each segment to the common length and compute median,
    # and envelope defined by the 5th and 95th percentiles (excluding extreme outliers)
    def aggregate_signal(signal_list):
        truncated = np.array([seg[:common_length] for seg in signal_list])
        median_signal = np.median(truncated, axis=0)
        lower_signal = np.percentile(truncated, 5, axis=0)
        upper_signal = np.percentile(truncated, 95, axis=0)
        return median_signal, lower_signal, upper_signal

    pos_med, pos_lower, pos_upper = aggregate_signal(pos_all)
    vel_med, vel_lower, vel_upper = aggregate_signal(vel_all)
    acc_med, acc_lower, acc_upper = aggregate_signal(acc_all)
    jerk_med, jerk_lower, jerk_upper = aggregate_signal(jerk_all)

    # Create 4 subplots for Position, Velocity, Acceleration, and Jerk with 4:3 ratio figure size
    fig, axs = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # Plot Position
    axs[0].plot(time_values, pos_med, color='blue', linewidth=2, label="Median Position")
    axs[0].fill_between(time_values, pos_lower, pos_upper, color='blue', alpha=0.2)
    # Plot Velocity
    axs[1].plot(time_values, vel_med, color='green', linewidth=2, label="Median Velocity")
    axs[1].fill_between(time_values, vel_lower, vel_upper, color='green', alpha=0.2)
    # Plot Acceleration
    axs[2].plot(time_values, acc_med, color='orange', linewidth=2, label="Median Acceleration")
    axs[2].fill_between(time_values, acc_lower, acc_upper, color='orange', alpha=0.2)
    # Plot Jerk
    axs[3].plot(time_values, jerk_med, color='red', linewidth=2, label="Median Jerk")
    axs[3].fill_between(time_values, jerk_lower, jerk_upper, color='red', alpha=0.2)

    # Set labels for each subplot
    axs[0].set_ylabel("Position")
    axs[1].set_ylabel("Velocity")
    axs[2].set_ylabel("Acceleration")
    axs[3].set_ylabel("Jerk")
    axs[3].set_xlabel("Time (s)")

    # Add legends to each subplot
    for ax in axs:
        ax.legend(loc="upper right", fontsize=10, frameon=False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
overlay_all_segments("07/22/HW", "left",
                        ballistic_phase=ballistic_phase,
                        correction_phase=correction_phase,
                        results=results,
                        marker="LFIN", 
                        frame_rate=200,
                        figsize=(7, 6))
# -------------------------------------------------------------------------------------------------------------------

def overlay_all_segments_3d(subject, hand, ballistic_phase, correction_phase, results, marker="RFIN", figsize=(12, 8), frame_rate=200):
    """
    For a given subject and hand, this function aggregates all trial segments across the files,
    extracts 3D kinematic signals (X, Y, Z), resamples/truncates them to the minimum common length,
    and computes the median trajectory with 5th–95th percentile envelopes.
    It then plots Position, Velocity, and Acceleration for each axis (X, Y, Z).
    
    Parameters:
        subject (str): Subject identifier (e.g., "07/22/HW").
        hand (str): Which hand ("left" or "right").
        ballistic_phase (dict): Nested dictionary with ballistic phase time windows.
        correction_phase (dict): Nested dictionary with corrective phase time windows.
        results (dict): Nested dictionary containing trajectory data.
        marker (str): Marker prefix ("RFIN" or "LFIN").
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Storage for each axis and kinematic measure
    pos_x_all, pos_y_all, pos_z_all = [], [], []
    vel_x_all, vel_y_all, vel_z_all = [], [], []
    acc_x_all, acc_y_all, acc_z_all = [], [], []

    # Loop over all file paths
    for file_path in results[subject][hand][1]:
        traj_data = results[subject][hand][1][file_path]['traj_data']

        num_segments = len(ballistic_phase[subject][hand][file_path])
        for seg_index in range(num_segments):
            bs, be = ballistic_phase[subject][hand][file_path][seg_index]
            cs, ce = correction_phase[subject][hand][file_path][seg_index]
            seg_length = ce - bs
            if seg_length > 0:
                # Extract signals for this segment
                x = np.array(traj_data[marker + "_X"][bs:ce])
                y = np.array(traj_data[marker + "_Y"][bs:ce])
                z = np.array(traj_data[marker + "_Z"][bs:ce])
                vx = np.array(traj_data[marker + "_VX"][bs:ce])
                vy = np.array(traj_data[marker + "_VY"][bs:ce])
                vz = np.array(traj_data[marker + "_VZ"][bs:ce])
                ax = np.array(traj_data[marker + "_AX"][bs:ce])
                ay = np.array(traj_data[marker + "_AY"][bs:ce])
                az = np.array(traj_data[marker + "_AZ"][bs:ce])

                # Append to storage
                pos_x_all.append(x)
                pos_y_all.append(y)
                pos_z_all.append(z)
                vel_x_all.append(vx)
                vel_y_all.append(vy)
                vel_z_all.append(vz)
                acc_x_all.append(ax)
                acc_y_all.append(ay)
                acc_z_all.append(az)

    if len(pos_x_all) == 0:
        print("No segments found across trials.")
        return

    # Common length across all trials
    common_length = min(len(seg) for seg in pos_x_all)
    time_values = np.arange(common_length) / frame_rate  # assuming 200 Hz

    # Helper to aggregate signals
    def aggregate_signal(signal_list):
        truncated = np.array([seg[:common_length] for seg in signal_list])
        median_signal = np.median(truncated, axis=0)
        lower_signal = np.percentile(truncated, 5, axis=0)
        upper_signal = np.percentile(truncated, 95, axis=0)
        return median_signal, lower_signal, upper_signal

    # Aggregate all signals
    pos_x_med, pos_x_low, pos_x_up = aggregate_signal(pos_x_all)
    pos_y_med, pos_y_low, pos_y_up = aggregate_signal(pos_y_all)
    pos_z_med, pos_z_low, pos_z_up = aggregate_signal(pos_z_all)

    vel_x_med, vel_x_low, vel_x_up = aggregate_signal(vel_x_all)
    vel_y_med, vel_y_low, vel_y_up = aggregate_signal(vel_y_all)
    vel_z_med, vel_z_low, vel_z_up = aggregate_signal(vel_z_all)

    acc_x_med, acc_x_low, acc_x_up = aggregate_signal(acc_x_all)
    acc_y_med, acc_y_low, acc_y_up = aggregate_signal(acc_y_all)
    acc_z_med, acc_z_low, acc_z_up = aggregate_signal(acc_z_all)

    # Plot: 3 rows (X, Y, Z), 3 columns (Position, Velocity, Acceleration)
    fig, axs = plt.subplots(3, 3, figsize=figsize, sharex=True)

    def plot_signal(ax, t, med, low, up, label, color):
        ax.plot(t, med, color=color, linewidth=2, label=label)
        ax.fill_between(t, low, up, color=color, alpha=0.2)
        ax.legend(loc="upper right", fontsize=8, frameon=False)

    # Position
    plot_signal(axs[0, 0], time_values, pos_x_med, pos_x_low, pos_x_up, "X Position", "blue")
    plot_signal(axs[1, 0], time_values, pos_y_med, pos_y_low, pos_y_up, "Y Position", "blue")
    plot_signal(axs[2, 0], time_values, pos_z_med, pos_z_low, pos_z_up, "Z Position", "blue")

    # Velocity
    plot_signal(axs[0, 1], time_values, vel_x_med, vel_x_low, vel_x_up, "X Velocity", "green")
    plot_signal(axs[1, 1], time_values, vel_y_med, vel_y_low, vel_y_up, "Y Velocity", "green")
    plot_signal(axs[2, 1], time_values, vel_z_med, vel_z_low, vel_z_up, "Z Velocity", "green")

    # Acceleration
    plot_signal(axs[0, 2], time_values, acc_x_med, acc_x_low, acc_x_up, "X Accel.", "orange")
    plot_signal(axs[1, 2], time_values, acc_y_med, acc_y_low, acc_y_up, "Y Accel.", "orange")
    plot_signal(axs[2, 2], time_values, acc_z_med, acc_z_low, acc_z_up, "Z Accel.", "orange")

    # Labels
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 1].set_xlabel("Time (s)")
    axs[2, 2].set_xlabel("Time (s)")

    axs[0, 0].set_ylabel("X")
    axs[1, 0].set_ylabel("Y")
    axs[2, 0].set_ylabel("Z")

    plt.tight_layout()
    plt.show()
overlay_all_segments_3d("07/22/HW", "left",
                        ballistic_phase=ballistic_phase,
                        correction_phase=correction_phase,
                        results=results,
                        marker="LFIN",
                        figsize=(7, 6),
                        frame_rate=200)

# -------------------------------------------------------------------------------------------------------------------
# def plot_phase_trajectory_combined(subject, hand, file_path, seg_indices,
#                                    ballistic_phase, correction_phase, results,
#                                    figsize=(7, 5), show_icon=False, icon_path=None,
#                                    placement_location_colors=None):
#     """
#     Plots ballistic and corrective phase trajectories in 3D for one or multiple segments.
#     Optionally overlays an icon at the start and end points, with customizable colors.

#     Parameters:
#         subject (str): Subject identifier.
#         hand (str): "left" or "right".
#         file_path (str): Path to trajectory CSV.
#         seg_indices (int or list): Segment index (or list of indices).
#         ballistic_phase (dict): Nested dict with ballistic phase time windows.
#         correction_phase (dict): Nested dict with corrective phase time windows.
#         results (dict): Trajectory data container.
#         figsize (tuple): Figure size.
#         show_icon (bool): Whether to overlay an icon at start/end.
#         icon_path (str): Path to the icon image (required if show_icon=True).
#         placement_location_colors (dict): Optional mapping from seg_index -> color.
#     """

#     if not isinstance(seg_indices, list):
#         seg_indices = [seg_indices]

#     # Get trajectory data
#     traj_data = results[subject][hand][1][file_path]['traj_data']
#     coord_prefix = "LFIN_" if hand.lower() == "left" else "RFIN_"
#     coord_x = np.array(traj_data[coord_prefix + "X"])
#     coord_y = np.array(traj_data[coord_prefix + "Y"])
#     coord_z = np.array(traj_data[coord_prefix + "Z"])

#     fig = plt.figure(figsize=figsize)
#     ax3d = fig.add_subplot(111, projection='3d')

#     for idx, seg_index in enumerate(seg_indices):
#         ballistic_start, ballistic_end_idx = ballistic_phase[subject][hand][file_path][seg_index]
#         corrective_start, corrective_end = correction_phase[subject][hand][file_path][seg_index]

#         # Choose color
#         if placement_location_colors is not None and seg_index in placement_location_colors:
#             color = placement_location_colors[seg_index]
#         else:
#             color = f"C{idx}"  # fallback to matplotlib default cycle


#         # Plot ballistic
#         ax3d.plot(
#             coord_x[ballistic_start:ballistic_end_idx],
#             coord_y[ballistic_start:ballistic_end_idx],
#             coord_z[ballistic_start:ballistic_end_idx],
#             color=color, linewidth=3, label=f'Segment {seg_index+1} Ballistic')

#         # Plot corrective
#         ax3d.plot(
#             coord_x[corrective_start:corrective_end],
#             coord_y[corrective_start:corrective_end],
#             coord_z[corrective_start:corrective_end],
#             color=color, linestyle='--', linewidth=3, label=f'Segment {seg_index+1} Corrective')
        
#         # Start & end points
#         ax3d.scatter(coord_x[ballistic_start], coord_y[ballistic_start], coord_z[ballistic_start],
#                      color='green', s=100, marker='o')
#         ax3d.scatter(coord_x[corrective_end - 1], coord_y[corrective_end - 1], coord_z[corrective_end - 1],
#                      color='red', s=100, marker='^')        
        
#         # Optionally add icons
#         if show_icon and icon_path:
#             import matplotlib.image as mpimg
#             img = mpimg.imread(icon_path)

#             def add_image(ax, xs, ys, zs, img, zoom=0.08):
#                 x2, y2, _ = proj_transform(xs, ys, zs, ax.get_proj())
#                 imagebox = OffsetImage(img, zoom=zoom)
#                 ab = AnnotationBbox(imagebox, (x2, y2), frameon=False, xycoords='data')
#                 ax.add_artist(ab)

#             add_image(ax3d, coord_x[ballistic_start]-50, coord_y[ballistic_start]+8, coord_z[ballistic_start]+60, img)
#             add_image(ax3d, coord_x[corrective_end]+40, coord_y[corrective_end]+6, coord_z[corrective_end]+50, img)


#     # Axes labels
#     ax3d.set_xlabel("X (mm)")
#     ax3d.set_ylabel("Y (mm)")
#     ax3d.set_zlabel("Z (mm)")
#     ax3d.legend(frameon=False, fontsize=12)

#     ax3d.set_xticks([-300, 0, 300])
#     ax3d.set_yticks([-60, -30, 0])
#     ax3d.set_zticks([700, 950, 1100])


#     plt.tight_layout()
#     plt.show()

# plot_phase_trajectory_combined(
#     subject="07/22/HW",
#     hand="left",
#     file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT66.csv',
#     seg_indices=[15],
#     ballistic_phase=ballistic_phase,
#     correction_phase=correction_phase,
#     results=results,
#     figsize=(5, 5),
#     show_icon=True,
#     icon_path="/Users/yilinwu/Desktop/HandHoldBlock1.png",
#     placement_location_colors=placement_location_colors
# )

# -------------------------------------------------------------------------------------------------------------------

def plot_phase_combined_single_figure(subject, hand, file_path, seg_indices,
                                      ballistic_phase, correction_phase, results,
                                      reach_speed_segments, placement_location_colors,
                                      marker="LFIN", frame_rate=200,
                                      figsize=(15, 8), show_icon=True, icon_path="/Users/yilinwu/Desktop/HandHoldBlock1.png",):
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
    signal_labels = ["Position", "Velocity", "Acceleration", "Jerk"]
    signals = [position_full, velocity_full, acceleration_full, jerk_full]
    units = ["mm", "mm/s", "mm/s²", "mm/s³"]
    colors_sig = {"full": "dimgray", "ballistic": "blue", "correction": "red"}

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
                  color=colors_sig["correction"], linestyle='--', linewidth=4,
                  label=f'Seg {seg_index+1} correction' if idx == 0 else None, zorder=1)
        # Start/end points
        ax3d.scatter(coord_x[ballistic_start], coord_y[ballistic_start], coord_z[ballistic_start],
                     color='white', edgecolors= "black",  s=100, marker='<', zorder=2)
        ax3d.scatter(coord_x[corrective_end-1], coord_y[corrective_end-1], coord_z[corrective_end-1],
                     color='white', edgecolors= "black", s=100, marker='>', zorder=2)
        ax3d.scatter(coord_x[ballistic_end_idx], coord_y[ballistic_end_idx], coord_z[ballistic_end_idx],
                     color='cyan', s=40, marker='o', zorder=2)

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



    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")
    ax3d.set_xticks([-250, 0, 250])
    ax3d.set_yticks([100, 115, 130])
    ax3d.set_zticks([850, 975, 1100])
    # ax3d.legend(frameon=False, loc=[0.5, 0.8])

    # --- Plot kinematic signals ---
    for i, (sig, label, unit) in enumerate(zip(signals, signal_labels, units)):
        ax = fig.add_subplot(gs[i, 3:])  # signals occupy right 3 columns
        ax.plot(time_full, sig[reach_start:reach_end], color=colors_sig["full"], linestyle="--", linewidth=4, zorder=1)

        ballistic_start, ballistic_end_idx = ballistic_phase[subject][hand][file_path][seg_indices[0]]
        corrective_start, corrective_end = correction_phase[subject][hand][file_path][seg_indices[0]]
        time_ballistic = (np.arange(ballistic_start, ballistic_end_idx) - reach_start) / frame_rate
        time_corrective = (np.arange(ballistic_end_idx, corrective_end) - reach_start) / frame_rate

        ax.plot(time_ballistic, sig[ballistic_start:ballistic_end_idx],
                color=colors_sig["ballistic"], linewidth=4, zorder=1)
        ax.plot(time_corrective, sig[ballistic_end_idx:corrective_end],
                color=colors_sig["correction"], linestyle="--", linewidth=4, zorder=1)
        ax.scatter((ballistic_end_idx - reach_start)/frame_rate, sig[ballistic_end_idx],
                   color="cyan", s=40, marker="o", zorder=2)

        ax.set_ylabel(f"{label}\n({unit})", labelpad=5, fontsize=10)
        ax.yaxis.set_label_coords(-0.18, 0.5)
        ax.relim()       # Recalculate limits
        ax.autoscale()   # Apply new limits

        if i == 0:
            ax.legend(["Full", "Ballistic", "Corrective", "Ballistic End"], frameon=False, fontsize=10)
        if i == len(signals) - 1:
            ax.set_xlabel("Time (s)")

    plt.tight_layout()
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
    figsize=(10, 4.5), 
    show_icon=True,
    icon_path="/Users/yilinwu/Desktop/HandHoldBlock1.png",
)

# -------------------------------------------------------------------------------------------------------------------

reach_TW_metrics_ballistic_phase = utils2.calculate_reach_metrics_for_time_windows_Normalizing(ballistic_phase, results)
reach_TW_metrics_correction_phase = utils2.calculate_reach_metrics_for_time_windows_Normalizing(correction_phase, results)
reach_TW_metrics_TW = utils2.calculate_reach_metrics_for_time_windows_Normalizing(test_windows_7, results)

reach_sparc_ballistic_phase = utils2.calculate_reach_sparc_Normalizing(ballistic_phase, results)
reach_sparc_correction_phase = utils2.calculate_reach_sparc_Normalizing(correction_phase, results)
reach_sparc_TW = utils2.calculate_reach_sparc_Normalizing(test_windows_7, results)

# -------------------------------------------------------------------------------------------------------------------
# Compute the minimal and maximal LDLJ values across segments (1 to 16) and print 
# the corresponding trial name and segment index

# def compute_ldlj_stats(reach_TW_metrics_ballistic_phase, subject="07/22/HW", hand="left"):
#     min_ldlj = float('inf')
#     max_ldlj = float('-inf')
#     min_trial = None
#     max_trial = None
#     min_segment = None
#     max_segment = None

#     # Create a list to hold all entries for further computations
#     ldlj_entries = []

#     for trial_name, trial_values in reach_TW_metrics_ballistic_phase['reach_LDLJ'][subject][hand].items():
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

# compute_ldlj_stats(reach_TW_metrics_ballistic_phase, subject="07/22/HW", hand="left")
# # -------------------------------------------------------------------------------------------------------------------

def calculate_median_of_medians_flexible(data_dict, metric=None):
    """
    Calculate median-of-medians per subject and hand for either metrics data or SPARC-like data.

    Args:
        data_dict (dict): nested dictionary
                          - metrics style: metric -> subject -> hand -> file -> list
                          - SPARC style: subject -> hand -> file -> list
        metric (str, optional): required if data_dict is metrics style.

    Returns:
        dict: subject -> {'left': median_of_medians, 'right': median_of_medians}
    """
    # Determine structure
    if metric is not None:  # metrics-style
        subjects_data = data_dict.get(metric, {})
    else:  # SPARC-style
        subjects_data = data_dict

    results = {}
    for subject, subject_data in subjects_data.items():
        results[subject] = {}
        for hand in ['left', 'right']:
            hand_data = subject_data.get(hand, {})
            medians = []
            for file_path, values in hand_data.items():
                try:
                    medians.append(float(np.median(values)))
                except Exception:
                    continue
            results[subject][hand] = float(np.median(medians)) if medians else None
    return results


TW_LDLJ = calculate_median_of_medians_flexible(reach_TW_metrics_TW, metric="reach_LDLJ")
ballistic_phase_LDLJ = calculate_median_of_medians_flexible(reach_TW_metrics_ballistic_phase, metric="reach_LDLJ")
correction_phase_LDLJ = calculate_median_of_medians_flexible(reach_TW_metrics_correction_phase, metric="reach_LDLJ")

median_sparc_TW = calculate_median_of_medians_flexible(reach_sparc_TW)
median_sparc_ballistic = calculate_median_of_medians_flexible(reach_sparc_ballistic_phase)
median_sparc_correction = calculate_median_of_medians_flexible(reach_sparc_correction_phase)



def plot_median_of_medians(ldlj_dicts, sparc_dicts, metric='reach_LDLJ', phases=None, hands=None, figsize=(12, 5)):
    """
    Compute median-of-medians and plot bar charts for LDLJ and SPARC data.

    Args:
        ldlj_dicts (list of dict): list of metric-style data_dicts for each phase
        sparc_dicts (list of dict): list of SPARC-style data_dicts for each phase
        metric (str): metric key for LDLJ data
        phases (list of str): phase names
        hands (list of str): hands to plot, e.g., ['left', 'right']
    """
    if phases is None:
        phases = ['TW', 'Ballistic', 'Correction']
    if hands is None:
        hands = ['left', 'right']

    def calculate_median_of_medians_flexible(data_dict, metric=None):
        """Calculate median-of-medians per subject and hand for either metrics or SPARC data."""
        if metric is not None:  # metrics-style
            subjects_data = data_dict.get(metric, {})
        else:  # SPARC-style
            subjects_data = data_dict

        results = {}
        for subject, subject_data in subjects_data.items():
            results[subject] = {}
            for hand in hands:
                hand_data = subject_data.get(hand, {})
                medians = []
                for file_path, values in hand_data.items():
                    try:
                        medians.append(float(np.median(values)))
                    except Exception:
                        continue
                results[subject][hand] = float(np.median(medians)) if medians else None
        return results

    # --- Compute medians ---
    ldlj_data = [calculate_median_of_medians_flexible(d, metric=metric) for d in ldlj_dicts]
    sparc_data = [calculate_median_of_medians_flexible(d) for d in sparc_dicts]

    # --- Plotting ---
    bar_width = 0.35
    x = np.arange(len(phases))
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    for ax, data, title, color in zip(axs, [ldlj_data, sparc_data], ['LDLJ', 'SPARC'], ['blue', 'green']):
        for i, hand in enumerate(hands):
            # Mean per hand across subjects
            means = [np.nanmean([d[s][hand] for s in d if d[s][hand] is not None]) for d in data]
            ax.bar(x + i*bar_width - bar_width/2, means, bar_width, label=f'{hand} hand', color=color, alpha=0.6 if hand=='right' else 1.0)

            # Overlay subject dots
            for j, phase_data in enumerate(data):
                for subject, values in phase_data.items():
                    if values[hand] is not None:
                        ax.plot(j + i*bar_width - bar_width/2, values[hand], 'ko', markersize=5)

        ax.set_xticks(x)
        ax.set_xticklabels(phases)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=10)

    # --- Custom y-ticks ---
    axs[0].set_yticks([0, -5, -10])  # adjust if needed
    axs[1].set_yticks([0, -1, -2])   # adjust if needed
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

plot_median_of_medians(
    ldlj_dicts=[reach_TW_metrics_TW, reach_TW_metrics_ballistic_phase, reach_TW_metrics_correction_phase],
    sparc_dicts=[reach_sparc_TW, reach_sparc_ballistic_phase, reach_sparc_correction_phase],
    metric="reach_LDLJ", figsize=(8, 4)
)


# import numpy as np
# import matplotlib.pyplot as plt

# def plot_median_of_medians_with_error(ldlj_dicts, sparc_dicts, metric='reach_LDLJ', phases=None, hands=None, figsize=(10, 5)):
#     """
#     Compute median-of-medians and plot bar charts with error bars for LDLJ and SPARC data.

#     Args:
#         ldlj_dicts (list of dict): list of metric-style data_dicts for each phase
#         sparc_dicts (list of dict): list of SPARC-style data_dicts for each phase
#         metric (str): metric key for LDLJ data
#         phases (list of str): phase names
#         hands (list of str): hands to plot, e.g., ['left', 'right']
#     """
#     if phases is None:
#         phases = ['TW', 'Ballistic', 'Correction']
#     if hands is None:
#         hands = ['left', 'right']

#     def calculate_median_of_medians_flexible(data_dict, metric=None):
#         """Calculate median-of-medians per subject and hand for either metrics or SPARC data."""
#         if metric is not None:  # metrics-style
#             subjects_data = data_dict.get(metric, {})
#         else:  # SPARC-style
#             subjects_data = data_dict

#         results = {}
#         for subject, subject_data in subjects_data.items():
#             results[subject] = {}
#             for hand in hands:
#                 hand_data = subject_data.get(hand, {})
#                 medians = []
#                 for file_path, values in hand_data.items():
#                     try:
#                         medians.append(float(np.median(values)))
#                     except Exception:
#                         continue
#                 results[subject][hand] = float(np.median(medians)) if medians else None
#         return results

#     # --- Compute medians ---
#     ldlj_data = [calculate_median_of_medians_flexible(d, metric=metric) for d in ldlj_dicts]
#     sparc_data = [calculate_median_of_medians_flexible(d) for d in sparc_dicts]

#     # --- Plotting ---
#     bar_width = 0.45
#     x = np.arange(len(phases))
#     fig, axs = plt.subplots(1, 2, figsize=figsize)

#     for ax, data, title, color in zip(axs, [ldlj_data, sparc_data], ['LDLJ', 'SPARC'], ['blue', 'green']):
#         for i, hand in enumerate(hands):
#             means = []
#             stds = []
#             for phase_data in data:
#                 hand_values = [phase_data[s][hand] for s in phase_data if phase_data[s][hand] is not None]
#                 means.append(np.nanmean(hand_values))
#                 stds.append(np.nanstd(hand_values))
#             ax.bar(x + i*bar_width - bar_width/2, means, bar_width, yerr=stds, capsize=5, label=f'{hand} hand', color=color, alpha=0.6 if hand=='right' else 1.0)

#         ax.set_xticks(x)
#         ax.set_xticklabels(phases)
#         ax.set_title(title)
#         ax.legend(frameon=False, fontsize=10, loc='lower right')

#     # --- Custom y-ticks ---
#     axs[0].set_yticks([0, -5, -10])  # adjust if needed
#     axs[1].set_yticks([0, -1, -2])   # adjust if needed
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.show()

# plot_median_of_medians_with_error(
#     ldlj_dicts=[reach_TW_metrics_TW, reach_TW_metrics_ballistic_phase, reach_TW_metrics_correction_phase],
#     sparc_dicts=[reach_sparc_TW, reach_sparc_ballistic_phase, reach_sparc_correction_phase],
#     metric="reach_LDLJ", figsize=(8, 4)
# )


# # -------------------------------------------------------------------------------------------------------------------
# def plot_trials_p_v_a_j_Nj_by_location(results, reach_speed_segments,
#                                            subject="07/22/HW", hand="left",
#                                            fs=200, target_samples=101, metrics=None,
#                                            mode="single", file_path=None):
#     """
#     Combined function to plot segmented signals with normalized jerk.
    
#     Options:
#       - mode="single": Plot only one selected trial.
#       - mode="all":    Overlay plots from all trials (all file paths).
    
#     Parameters:
#         results (dict): Dictionary with trajectory data.
#         reach_speed_segments (dict): Dictionary with segmentation ranges.
#         subject (str): Subject identifier.
#         hand (str): Hand identifier.
#         fs (int): Sampling rate.
#         target_samples (int): Number of samples for the normalized jerk signal.
#         metrics (list or None): List of metrics to plot.
#             Options include: 'pos', 'vel', 'acc', 'jerk', 'norm_jerk'.
#             If None, all metrics are plotted.
#         mode (str): "single" to plot one selected trial, "all" to overlay all trials.
#         file_path (str or None): Required if mode is "single"; path to the CSV file.
#     """
#     import matplotlib.pyplot as plt

#     # Default metrics if not provided.
#     if metrics is None:
#         metrics = ["pos", "vel", "acc", "jerk", "norm_jerk"]
    
#     # Select marker based on hand.
#     marker = 'RFIN' if hand == 'right' else 'LFIN'
    
#     # -------------------------- Mode 1: Single trial --------------------------
#     if mode == "single":
#         if file_path is None:
#             print("For mode 'single' you must provide a file_path.")
#             return

#         # Get segmentation ranges for the provided file.
#         seg_ranges = reach_speed_segments[subject][hand][file_path]
#         n_segments = len(seg_ranges)
        
#         # Create grid of subplots.
#         n_rows = int(np.ceil(np.sqrt(n_segments)))
#         n_cols = int(np.ceil(n_segments / n_rows))
#         fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
#         axs = np.array(axs).flatten()
        
#         # Extract signals from the selected file.
#         traj_data = results[subject][hand][1][file_path]['traj_space'][marker]
#         position = traj_data[0]
#         velocity = traj_data[1]
#         acceleration = traj_data[2]
#         jerk = traj_data[3]
        
#         for i, (start_idx, end_idx) in enumerate(seg_ranges):
#             seg_length = end_idx - start_idx
#             x_vals = np.arange(seg_length)
            
#             # Slice signals.
#             pos_seg = position[start_idx:end_idx]
#             vel_seg = velocity[start_idx:end_idx]
#             acc_seg = acceleration[start_idx:end_idx]
#             jerk_seg = jerk[start_idx:end_idx]
            
#             # Normalize jerk via interpolation.
#             duration = len(jerk_seg) / fs
#             t_orig = np.linspace(0, duration, num=len(jerk_seg))
#             t_std = np.linspace(0, duration, num=target_samples)
#             warped_jerk = np.interp(t_std, t_orig, jerk_seg)
            
#             # Mapping metric keys to value and plotting options.
#             available_metrics = {
#                 "pos":      {"data": pos_seg, "label": "Pos", "x": x_vals, "linestyle": "-"},
#                 "vel":      {"data": vel_seg, "label": "Vel", "x": x_vals, "linestyle": "-"},
#                 "acc":      {"data": acc_seg, "label": "Acc", "x": x_vals, "linestyle": "-"},
#                 "jerk":     {"data": jerk_seg, "label": "Jerk", "x": x_vals, "linestyle": "-"},
#                 "norm_jerk": {"data": warped_jerk, "label": "Norm Jerk", "x": np.linspace(0, 100, len(warped_jerk)), "linestyle": "--"}
#             }
            
#             ax = axs[i]
#             for key in metrics:
#                 if key in available_metrics:
#                     met = available_metrics[key]
#                     ax.plot(met["x"], met["data"], label=met["label"], linestyle=met["linestyle"], alpha=0.7)
            
#             ax.set_title(f"Segment {i+1}\nDuration: {duration:.2f}s")
#             ax.set_xlabel("Samples / %")
#             ax.set_ylabel("Signal")
#             ax.grid(True)
#             ax.legend(fontsize=8)
        
#         # Hide any unused subplots.
#         for j in range(i + 1, len(axs)):
#             axs[j].axis('off')
        
#         plt.tight_layout()
#         plt.show()
    
#     # -------------------------- Mode 2: All trials --------------------------
#     elif mode == "all":
#         # Get all file paths for the trial.
#         file_paths = list(results[subject][hand][1].keys())
#         if not file_paths:
#             print("No files found for the specified subject/hand/trial.")
#             return
        
#         # Use segmentation of the first file to setup grid.
#         seg_ranges = reach_speed_segments[subject][hand][file_paths[0]]
#         n_segments = len(seg_ranges)
        
#         # Create grid of subplots.
#         n_rows = int(np.ceil(np.sqrt(n_segments)))
#         n_cols = int(np.ceil(n_segments / n_rows))
#         fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
#         axs = np.array(axs).flatten()
        
#         # Loop over each segment index.
#         for seg_idx in range(n_segments):
#             ax = axs[seg_idx]
#             plotted_labels = {}
#             avg_duration = 0
#             count = 0
            
#             # Loop over each file for overlay.
#             for fp in file_paths:
#                 # Get segmentation range for current segment.
#                 start_idx, end_idx = reach_speed_segments[subject][hand][fp][seg_idx]
#                 seg_length = end_idx - start_idx
#                 x_vals = np.arange(seg_length)
                
#                 # Extract signals from current file.
#                 traj_data = results[subject][hand][1][fp]['traj_space'][marker]
#                 position = traj_data[0]
#                 velocity = traj_data[1]
#                 acceleration = traj_data[2]
#                 jerk = traj_data[3]
                
#                 pos_seg = position[start_idx:end_idx]
#                 vel_seg = velocity[start_idx:end_idx]
#                 acc_seg = acceleration[start_idx:end_idx]
#                 jerk_seg = jerk[start_idx:end_idx]
                
#                 # Normalize jerk.
#                 duration = len(jerk_seg) / fs
#                 avg_duration += duration
#                 count += 1
#                 t_orig = np.linspace(0, duration, num=len(jerk_seg))
#                 t_std = np.linspace(0, duration, num=target_samples)
#                 warped_jerk = np.interp(t_std, t_orig, jerk_seg)
                
#                 available_metrics = {
#                     "pos":       {"data": pos_seg, "label": "Pos", "x": x_vals, "linestyle": "-"},
#                     "vel":       {"data": vel_seg, "label": "Vel", "x": x_vals, "linestyle": "-"},
#                     "acc":       {"data": acc_seg, "label": "Acc", "x": x_vals, "linestyle": "-"},
#                     "jerk":      {"data": jerk_seg, "label": "Jerk", "x": x_vals, "linestyle": "-"},
#                     "norm_jerk": {"data": warped_jerk, "label": "Norm Jerk", "x": np.linspace(0, 100, len(warped_jerk)), "linestyle": "--"}
#                 }
                
#                 for key in metrics:
#                     if key in available_metrics:
#                         met = available_metrics[key]
#                         label = met["label"] if key not in plotted_labels else None
#                         plotted_labels[key] = True
#                         ax.plot(met["x"], met["data"], label=label, linestyle=met["linestyle"], alpha=0.7)
            
#             # Average duration over files.
#             avg_duration = avg_duration / count if count else 0
#             ax.set_title(f"Segment {seg_idx+1}\n(Avg duration: {avg_duration:.2f}s)")
#             ax.set_xlabel("Samples / %")
#             ax.set_ylabel("Signal")
#             ax.grid(True)
#             ax.legend(fontsize=8)
        
#         # Hide unused subplots.
#         for j in range(seg_idx + 1, len(axs)):
#             axs[j].axis('off')
        
#         plt.tight_layout()
#         plt.show()
    
#     else:
#         print("Invalid mode selected. Choose 'single' or 'all'.")

# # Example call for option 1 (single trial):
# plot_trials_p_v_a_j_Nj_by_location(results, ballistic_phase,
#                                        subject="07/22/HW", hand="left", 
#                                        fs=200, target_samples=101,
#                                        metrics=["vel"],
#                                        mode="single",
#                                        file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv')

# # Example call for option 2 (all trials):
# plot_trials_p_v_a_j_Nj_by_location(results, ballistic_phase,
#                                        subject="07/22/HW", hand="left", 
#                                        fs=200, target_samples=101,
#                                        metrics=["vel"],
#                                        mode="all")

# # Example call for option 2 (all trials):
# plot_trials_p_v_a_j_Nj_by_location(results, correction_phase,
#                                        subject="07/22/HW", hand="left", 
#                                        fs=200, target_samples=101,
#                                        metrics=["vel"],
#                                        mode="all")
# # -------------------------------------------------------------------------------------------------------------------
utils5.process_and_save_combined_metrics_acorss_phases(
    Block_Distance, reach_metrics,
    reach_TW_metrics_ballistic_phase, reach_TW_metrics_correction_phase,
    reach_sparc_ballistic_phase, reach_sparc_correction_phase,
    reach_TW_metrics_TW, reach_sparc_TW,
    All_dates, DataProcess_folder)

# # -------------------------------------------------------------------------------------------------------------------
# --- LOAD ALL COMBINED METRICS PER SUBJECT FROM PICKLE FILE ---
all_combined_metrics_acorss_phases = utils5.load_selected_subject_results_acorss_TWs(All_dates, DataProcess_folder)

# Swap and rename metrics for consistency
all_combined_metrics_acorss_phases = utils5.swap_and_rename_metrics(all_combined_metrics_acorss_phases, All_dates)

# Filter all_combined_metrics based on distance and count NaNs
filtered_metrics_acorss_phases, total_nan_acorss_phases, Nan_counts_per_subject_per_hand_acorss_phases, Nan_counts_per_index_acorss_phases = utils5.filter_combined_metrics_and_count_nan(all_combined_metrics_acorss_phases)

# Plot histograms and identify outliers
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
outliers = plot_histograms(filtered_metrics_acorss_phases, sd_multiplier=4, overlay_median=True, overlay_sd=False, overlay_iqr=True)

# Update filtered metrics and count NaN replacements based on distance and duration thresholds: distance_threshold=15, duration_threshold=1.6
updated_metrics_acorss_phases, Cutoff_counts_per_subject_per_hand_acorss_phases, Cutoff_counts_per_index_acorss_phases, total_nan_per_subject_hand_acorss_phases = utils5.update_filtered_metrics_and_count(filtered_metrics_acorss_phases)

# # -------------------------------------------------------------------------------------------------------------------
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


def plot_phase_metrics_custom(subject_medians, metrics_groups, phases=None, figsize=(10,5)):
    """
    Plots multiple metrics in subplots with specified colors, y-axis ticks, legend placement,
    hatching for non-dominant bars, and error bars (std dev across subjects).
    """
    if phases is None:
        phases = ["TW", "ballistic", "correction"]

    n_subplots = len(metrics_groups)
    fig, axes = plt.subplots(1, n_subplots, figsize=figsize)
    if n_subplots == 1:
        axes = [axes]  # make iterable

    # Define colors per phase
    phase_colors = {"TW": "white", "ballistic": "blue", "correction": "red"}

    # Define custom y-ticks per metric
    yticks_dict = {"ldlj": [0, -5, -10], "sparc": [0, -1, -2]}

    for ax, (metric_prefix, title) in zip(axes, metrics_groups):
        # Metric keys
        if metric_prefix.lower() in ["distance", "durations"]:
            metric_keys = {phase: metric_prefix for phase in phases}
        else:
            metric_keys = {phase: f"{phase}_{metric_prefix}" for phase in phases}

        # Aggregate values
        aggregated = {phase: {"dominant": [], "non_dominant": []} for phase in phases}
        for subject, hands in subject_medians.items():
            for hand, metrics in hands.items():
                for phase in phases:
                    key = metric_keys.get(phase)
                    if key in metrics:
                        aggregated[phase][hand].append(metrics[key])

        # Compute averages and standard deviations
        averages = {phase: {} for phase in phases}
        std_devs = {phase: {} for phase in phases}
        for phase in phases:
            for hand in ["dominant", "non_dominant"]:
                vals = aggregated[phase][hand]
                averages[phase][hand] = np.nanmean(vals) if vals else np.nan
                std_devs[phase][hand] = np.nanstd(vals) if vals else 0

        # Bar plot
        labels = phases
        x = np.arange(len(labels))
        width = 0.35

        for i, phase in enumerate(phases):
            dom_val = averages[phase]["dominant"]
            dom_err = std_devs[phase]["dominant"]
            nondom_val = averages[phase]["non_dominant"]
            nondom_err = std_devs[phase]["non_dominant"]

            # Dominant bars: solid fill with error bar
            ax.bar(x[i] - width/2, dom_val, width, color=phase_colors[phase], edgecolor='black',
                   yerr=dom_err, capsize=5, label='Dominant' if i == 0 else "")
            # Non-dominant bars: hatched with error bar
            ax.bar(x[i] + width/2, nondom_val, width, color=phase_colors[phase], edgecolor='black',
                   hatch='//', yerr=nondom_err, capsize=5, label='Non-dominant' if i == 0 else "")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric_prefix.upper() if metric_prefix.lower() not in ["distance", "durations"] else metric_prefix.capitalize())
        
        # Set custom y-ticks if defined
        if metric_prefix.lower() in yticks_dict:
            ax.set_yticks(yticks_dict[metric_prefix.lower()])

        # Legend at bottom-right
        if metric_prefix.lower() == "ldlj":
            ax.legend(loc='lower left', frameon=False)

    plt.tight_layout()
    plt.show()

plot_phase_metrics_custom(subject_medians, metrics_groups=[("LDLJ", "LDLJ"), ("sparc", "SPARC")], figsize=(8,4))


def plot_median_metrics(subject_medians, hand_order=["non_dominant", "dominant"], figsize=(12, 6)):
    """
    Plots boxplots of median Duration and Distance per hand using the subject_medians dictionary.
    
    The subject_medians dictionary is expected to have the structure:
      { subject: { hand: { 'durations': median_duration, 'distance': median_distance, ... } } }
    
    Parameters:
        subject_medians (dict): Dictionary containing median metrics per subject.
        hand_order (list): List specifying the order of hand categories.
        figsize (tuple): Figure size for the plot (in inches).
    """
    # Convert the subject_medians dictionary into a DataFrame.
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
    
    # Map hand keys to nicely formatted labels
    hand_labels = {"dominant": "Dominant", "non_dominant": "Non-dominant"}
    df["Hand"] = df["Hand"].map(hand_labels)
    
    # Update hand_order to match the new labels
    hand_order_labels = [hand_labels[h] for h in hand_order]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Define custom box colors for consistency (light grey palette)
    box_colors = ["#D3D3D3", "#F0F0F0"]
    
    # Plot for Duration
    sns.boxplot(
        data=df, x="Hand", y="Duration", 
        order=hand_order_labels, ax=axes[0],
        palette=box_colors
    )
    sns.swarmplot(
        data=df, x="Hand", y="Duration",
        order=hand_order_labels, ax=axes[0],
        color='black', size=6
    )
    axes[0].set_xlabel("Hand")
    axes[0].set_ylabel("Duration (s)")
    axes[0].set_ylim(0, 1.2)  # Set y-axis limit for Duration plot
    axes[0].set_yticks([0, 0.6, 1.2])  # Custom y-ticks for Duration plot    
    # Plot for Distance
    sns.boxplot(
        data=df, x="Hand", y="Distance",
        order=hand_order_labels, ax=axes[1],
        palette=box_colors
    )
    sns.swarmplot(
        data=df, x="Hand", y="Distance",
        order=hand_order_labels, ax=axes[1],
        color='black', size=6
    )
    axes[1].set_xlabel("Hand")
    axes[1].set_ylabel("Error Distance (mm)")
    axes[1].set_ylim(0, 5)  # Set y-axis limit for Distance plot
    axes[1].set_yticks([0, 2.5, 5])  # Custom y-ticks for Distance plot    
    plt.tight_layout()
    plt.show()
plot_median_metrics(subject_medians, hand_order=["non_dominant", "dominant"], figsize=(8, 4))

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

def plot_trials_mean_median_of_reach_indices(stats, subject, hand, metric_x, metric_y, stat_type="avg", use_unique_colors=False, figsize=(8,6), marker_style='D'):
    """
    Overlays scatter plots for all reach indices in a single plot using either mean or median statistics.
    For each reach index (1 to 16), it uses the colors provided by placement_location_colors.
    Calculates and returns the Spearman correlation for the overlayed points.

    Parameters:
        stats (dict): Statistics (mean or median) for all subjects and hands.
        subject (str): Subject identifier.
        hand (str): Hand ('non_dominant' or 'dominant').
        metric_x (str): Metric for x-axis.
        metric_y (str): Metric for y-axis.
        stat_type (str): Type of statistics to use ("mean" or "median").
        use_unique_colors (bool): Ignored in this implementation.
        figsize (tuple): Figure size.
        marker_style (str): Marker style for points (default: 'D' for diamond).

    Returns:
        tuple: Spearman correlation and p-value for the overlayed points.
    """
    plt.figure(figsize=figsize)

    x_values = []
    y_values = []

    import matplotlib.colors as mcolors

    # For each reach index (0 to 15), use placement_location_colors to determine the marker color.
    for reach_index in range(16):
        duration = stats[subject][hand][reach_index][f"{stat_type}_duration"]
        distance = stats[subject][hand][reach_index][f"{stat_type}_distance"]
        if not np.isnan(duration) and not np.isnan(distance):
            x_values.append(duration)
            y_values.append(distance)
            color = placement_location_colors[reach_index]
            dark_color = tuple(c * 0.8 for c in mcolors.to_rgb(color))
            
            # Add label for legend
            plt.scatter(duration, distance, facecolors='none', edgecolors=dark_color, s=300,
                        zorder=5, alpha=1.0, marker=marker_style, label= reach_index+1)
            plt.text(duration, distance, f"{reach_index+1}", fontsize=10, color=dark_color, ha="center", va="center")

    if len(x_values) > 1 and len(y_values) > 1:
        spearman_corr, p_value = spearmanr(x_values, y_values)
    else:
        spearman_corr, p_value = np.nan, np.nan

    plt.ylabel('Median distance : Good → Bad (mm)', fontsize=16)
    plt.ylim(1.2, 3.8)
    plt.yticks([1.2, 2.5, 3.8], fontsize=14)
    plt.xlabel('Median duration : Fast → Slow (s)', fontsize=16)
    plt.xlim(0.65, 0.95)
    plt.xticks([0.65, 0.8, 0.95], fontsize=14)

    print(f"Overlay of Reach Statistics ({subject}, {hand.capitalize()}, {stat_type.capitalize()})\n"
              f"Spearman Corr: {spearman_corr:.2f}, P-value: {p_value:.2f}")

    plt.grid(False)
    plt.tight_layout()
    plt.show()

    return spearman_corr, p_value

plot_trials_mean_median_of_reach_indices(median_stats, '07/22/HW', 'dominant', 'durations', 'distance', stat_type="median", use_unique_colors=True, figsize=(6,5), marker_style='D')

# Calculate Spearman correlation, p-value, data points, and hyperbolic fit parameters (a, b) for durations vs distances for each subject and hand across reach indices
def calculate_duration_distance_trials_mean_median_of_reach_indices(stats, selected_subjects=None, stat_type="avg"):
    """
    Calculates Spearman correlation, p-value, data points, and hyperbolic fit parameters (a, b)
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

                # Calculate Spearman correlation
                if len(x_values) > 1 and len(y_values) > 1:
                    spearman_corr, p_value = spearmanr(x_values, y_values)
                else:
                    spearman_corr, p_value = np.nan, np.nan

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
                    "spearman_corr": spearman_corr,
                    "p_value": p_value,
                    "data_points": len(x_values),
                    "hyperbolic_fit_a": a,
                    "hyperbolic_fit_b": b
                }

    return results

# Calculate Spearman correlation, p-value, data points, and hyperbolic fit parameters
SAT_corr_acorss_results = calculate_duration_distance_trials_mean_median_of_reach_indices(median_stats, stat_type='median')

# Create a box plot of Spearman correlations for durations vs distances across all subjects,
def boxplot_spearman_corr_trials_mean_median_of_reach_indices(results, figsize=(8,6), orientation="horizontal"):
    """
    Plots a box plot of Spearman correlations for durations vs. distances across all subjects,
    with separate boxes for non-dominant and dominant hands. Also performs Fisher's z-test
    to compare the two categories, and applies the Wilcoxon signed-rank test on each hand
    (after Fisher r-to-z transform) to test whether the distribution of correlations across
    subjects differs from 0.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import norm, wilcoxon

    # Gather correlations and corresponding hand labels
    correlations = []
    hand_labels_raw = []
    for subject, hands_data in results.items():
        for hand, metrics in hands_data.items():
            if "spearman_corr" in metrics and not (
                metrics["spearman_corr"] is None or
                (isinstance(metrics["spearman_corr"], float) and np.isnan(metrics["spearman_corr"]))
            ):
                correlations.append(metrics["spearman_corr"])
                hand_labels_raw.append(hand)

    # Create DataFrame for plotting
    df = pd.DataFrame({"Hand": hand_labels_raw, "Spearman Correlation": correlations})

    # Map hand keys to nicely formatted labels
    hand_label_map = {"dominant": "Dominant", "non_dominant": "Non-dominant"}
    df["Hand"] = df["Hand"].map(hand_label_map)

    # Define consistent hand order and custom box colors
    hand_order_labels = ["Non-dominant", "Dominant"]
    box_colors = {"Non-dominant": "#D3D3D3", "Dominant": "#F0F0F0"}

    # Plot
    plt.figure(figsize=figsize)
    if orientation.lower() == "horizontal":
        sns.boxplot(
            y="Hand", x="Spearman Correlation", data=df,
            order=hand_order_labels, palette=box_colors
        )
        sns.swarmplot(
            y="Hand", x="Spearman Correlation", data=df,
            order=hand_order_labels, color="black", size=6
        )
        plt.xlim(-1, 1)
        plt.xticks([-1, 0, 1], fontsize=12)
        plt.xlabel("Spearman Correlation")
        plt.ylabel("Hand")
    else:
        sns.boxplot(
            x="Hand", y="Spearman Correlation", data=df,
            order=hand_order_labels, palette=box_colors
        )
        sns.swarmplot(
            x="Hand", y="Spearman Correlation", data=df,
            order=hand_order_labels, color="black", size=6
        )
        plt.ylim(-1, 1)
        plt.yticks([-1, 0, 1], fontsize=12)
        plt.xlabel("Hand")
        plt.ylabel("Spearman Correlation")

    plt.tight_layout()
    plt.show()

    # Summary stats
    for hand in hand_order_labels:
        values = df[df["Hand"] == hand]["Spearman Correlation"]
        median = values.median()
        iqr = values.quantile(0.75) - values.quantile(0.25)
        print(f"{hand}: Median = {median:.2f}, IQR = {iqr:.2f}")

    # Fisher’s z-test for difference between two independent correlations
    vals1 = df[df["Hand"] == "Non-dominant"]["Spearman Correlation"].values
    vals2 = df[df["Hand"] == "Dominant"]["Spearman Correlation"].values

    if len(vals1) > 3 and len(vals2) > 3:
        r1, r2 = np.mean(vals1), np.mean(vals2)  # use group mean correlations
        n1, n2 = len(vals1), len(vals2)

        # Fisher z-transform
        z1 = 0.5 * np.log((1 + r1) / (1 - r1))
        z2 = 0.5 * np.log((1 + r2) / (1 - r2))

        # Standard error
        se = np.sqrt(1/(n1 - 3) + 1/(n2 - 3))

        # z statistic
        z_stat = (z1 - z2) / se
        p_val = 2 * (1 - norm.cdf(abs(z_stat)))

        print(f"\nFisher's z-test comparing categories:")
        print(f" Non-dominant mean r = {r1:.3f}, n={n1}")
        print(f" Dominant mean r     = {r2:.3f}, n={n2}")
        print(f" z = {z_stat:.3f}, p = {p_val:.4f}")
    else:
        print("\nNot enough data for Fisher’s z-test (need n > 3 in each group).")

    # Apply Wilcoxon signed-rank test for each hand to test if the distribution differs from 0
    for hand in hand_order_labels:
        vals = df[df["Hand"] == hand]["Spearman Correlation"].values
        if len(vals) > 0:
            try:
                stat, p = wilcoxon(vals)
                print(f"Wilcoxon signed-rank test for {hand}: statistic = {stat:.3f}, p = {p:.4f}")
            except Exception as e:
                print(f"Wilcoxon test for {hand} could not be performed: {str(e)}")
        else:
            print(f"No data for Wilcoxon test for {hand}.")

boxplot_spearman_corr_trials_mean_median_of_reach_indices(SAT_corr_acorss_results, figsize=(4,3), orientation="vertical")

# # -------------------------------------------------------------------------------------------------------------------
def plot_reach_scatter_and_spearman(subject, hand, reach_index, figsize=(6,4)):
    """
    Plots a scatter plot of durations vs. distances for a given subject, hand, and reach index
    across trials and calculates the Spearman correlation and p-value.
    
    Parameters:
        subject (str): Subject identifier (e.g., "07/22/HW")
        hand (str): Hand identifier (e.g., "non_dominant")
        reach_index (int): Index of the reach (0-indexed)
        
    Returns:
        tuple: Spearman correlation coefficient and p-value
    """
    durations = []
    distances = []
    trial_names = []

    # Gather duration and distance values for the chosen reach across trials
    for trial, rep_durations in updated_metrics_acorss_phases[subject][hand]['durations'].items():
        idx = reach_index  # Adjust if reach_index is 1-indexed (subtract 1)
        duration = rep_durations[idx]
        distance = updated_metrics_acorss_phases[subject][hand]['distance'][trial][idx]
        durations.append(duration)
        distances.append(distance)
        trial_names.append(trial)
    
    # Calculate Spearman correlation
    corr, pval = spearmanr(durations, distances)
    
    # Create the scatter plot
    plt.figure(figsize=figsize)
    plt.scatter(
        durations,
        distances,
        color=placement_location_colors[reach_index],  # use color from your mapping
        s=50,
        label=f"Reach {reach_index}"
    )
    plt.xlabel("Duration (s)")
    plt.xticks([0.5, 0.85, 1.2], fontsize=12)
    plt.ylabel("Distance (mm)")
    plt.yticks([0, 5.5, 11], fontsize=12)
    # plt.title(f"Spearman r: {corr:.2f}, p: {pval:.3f}")
    
    # remove grid
    plt.grid(False)
        
    plt.tight_layout()
    plt.show()
    
    return corr, pval

corr_value, p_value = plot_reach_scatter_and_spearman("07/22/HW", "non_dominant", 0, figsize=(4,3))


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

def heatmap_spearman_correlation_reach_indices(results, hand="both", simplified=False, return_medians=False, overlay_median=False):
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
        
    Returns:
        dict or None: If return_medians is True, returns a dictionary with keys corresponding to each hand 
                      (or the chosen hand) and values as dictionaries with 'column_medians' and 'row_medians'.
    """
    import matplotlib.pyplot as plt
    reach_indices = list(range(16))
    medians = {}
    
    if hand == "both":
        if simplified:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        else:
            fig, axes = plt.subplots(2, 1, figsize=(12, len(results) * 0.5))
        
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
                cmap="coolwarm",
                cbar=True,
                xticklabels=list(range(1, 17)),
                yticklabels=[] if simplified else subjects,
                vmin=-1,
                vmax=1,
                ax=ax
            )
            hand_order_labels = ["Non-dominant", "Dominant"]
            ax.set_title(f"{hand_order_labels[idx]} Hand", fontsize=20 if not simplified else 14)
            ax.set_xlabel("Placement location", fontsize=18)
            ax.set_xticklabels(range(1, 17), fontsize=10, rotation=0)
            ax.set_ylabel("Subjects", fontsize=18)
            if overlay_median:
                import matplotlib.patches as patches
                for i, subject in enumerate(df.index):
                    # Calculate row median from non-NaN values
                    row_data = df.loc[subject].dropna()
                    if row_data.empty:
                        continue
                    median_val = np.median(row_data.values)
                    # Find the column index with the value closest to the row median
                    col_idx = np.argmin(np.abs(df.loc[subject].values - median_val))
                    # Overlay a green rectangle to highlight the cell
                    ax.add_patch(patches.Rectangle((col_idx, i), 1, 1, fill=False, edgecolor='green', lw=2))
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
        df = pd.DataFrame(data, index=subjects, columns=reach_indices)
        fig, ax = plt.subplots(figsize=(6, 6) if simplified else (12, len(subjects) * 0.5))
        sns.heatmap(
            df,
            annot=not simplified,
            fmt=".2f",
            cmap="coolwarm",
            cbar=True,
            xticklabels=list(range(1, 17)),
            yticklabels=[] if simplified else subjects,
            vmin=-1,
            vmax=1,
            ax=ax
        )
        ax.set_title(f"{hand.replace('_', '-').title()} Hand", fontsize=20 if not simplified else 16)
        ax.set_xlabel("Placement location", fontsize=18 if not simplified else 18)
        ax.set_xticklabels(range(1, 17), fontsize=14, rotation=0)
        ax.set_ylabel("Subjects", fontsize=18 if not simplified else 18)
        ax.set_yticklabels([] if simplified else ax.get_yticklabels())
        if overlay_median:
            import matplotlib.patches as patches
            for i, subject in enumerate(df.index):
                # Calculate row median from non-NaN values
                row_data = df.loc[subject].dropna()
                if row_data.empty:
                    continue
                median_val = np.median(row_data.values)
                # Find the column index with the value closest to the row median
                col_idx = np.argmin(np.abs(df.loc[subject].values - median_val))
                # Overlay a green rectangle to highlight the cell
                ax.add_patch(patches.Rectangle((col_idx, i), 1, 1, fill=False, edgecolor='green', lw=2))
        plt.tight_layout()
        plt.show()
        if return_medians:
            medians[hand] = {
                "column_medians": df.median(axis=0).to_dict(),
                "row_medians": df.median(axis=1).to_dict()
            }
    
    if return_medians:
        return medians

# Plot heatmap for Spearman correlations for each reach indices (1 to 16) for each subject and hand
medians = heatmap_spearman_correlation_reach_indices(SAT_corr_within_results, hand="non_dominant", simplified=True, return_medians=True, overlay_median=True)

def boxplot_spearman_corr_with_stats_reach_indices_by_subject(results, figsize=(8,6)):
    """
    Creates a box plot of median Spearman correlations for each subject,
    separated by non_dominant and dominant hands. Uses the same figure color
    and format as boxplot_spearman_corr_trials_mean_median_of_reach_indices.
    
    Parameters:
        results (dict): Results containing Spearman correlations for each subject and hand.
    """
    import matplotlib.pyplot as plt

    median_corr_non = []
    median_corr_dom = []
    # Collect median correlation for each subject for each hand.
    for subject in results.keys():
        for hand in ['non_dominant', 'dominant']:
            if hand in results[subject]:
                # Gather correlations for reach_indices 0-15.
                correlations = [
                    results[subject][hand].get(reach_index, {}).get("spearman_corr", np.nan)
                    for reach_index in range(16)
                ]
                # Remove NaNs.
                correlations = [r for r in correlations if not np.isnan(r)]
                if correlations:
                    med_corr = np.median(correlations)
                    if hand == 'non_dominant':
                        median_corr_non.append(med_corr)
                    elif hand == 'dominant':
                        median_corr_dom.append(med_corr)

    # Create a DataFrame for plotting.
    data = []
    for x in median_corr_non:
        data.append({"Hand": "Non-dominant", "Median Spearman Correlation": x})
    for x in median_corr_dom:
        data.append({"Hand": "Dominant", "Median Spearman Correlation": x})
    df_plot = pd.DataFrame(data)
    
    # Define hand order and custom palette.
    hand_order = ["Non-dominant", "Dominant"]
    palette = {"Non-dominant": "#D3D3D3", "Dominant": "#F0F0F0"}
    
    # Create the box plot with overlayed swarm plot.
    plt.figure(figsize=figsize)
    sns.boxplot(x="Hand", y="Median Spearman Correlation", data=df_plot,
                order=hand_order, palette=palette)
    sns.swarmplot(x="Hand", y="Median Spearman Correlation", data=df_plot,
                  order=hand_order, color='black', size=6)
    plt.xlabel("Hand", fontsize=13)
    plt.ylabel("Median Spearman Correlation", fontsize=13)
    plt.ylim(-1, 1)
    plt.yticks([-1, 0, 1], fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Use Fisher's z-test to compare the two groups if sufficient data is available.
    if (len(median_corr_non) > 3) and (len(median_corr_dom) > 3):
        # Compute the group means.
        r1 = np.mean(median_corr_non)
        r2 = np.mean(median_corr_dom)
        n1 = len(median_corr_non)
        n2 = len(median_corr_dom)
        # Fisher's r-to-z transformation.
        z1 = np.arctanh(r1)
        z2 = np.arctanh(r2)
        # Standard error.
        se = np.sqrt(1/(n1 - 3) + 1/(n2 - 3))
        # z statistic.
        z_stat = (z1 - z2) / se
        # Two-tailed p-value.
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
        print(f"Fisher's z-test: z = {z_stat:.2f}, p = {p_value:.4f}")
    else:
        print("Not enough data for Fisher's z-test (need >3 subjects per group).")
    
    # Apply Wilcoxon signed-rank test for each hand to test whether the median correlation 
    # distribution significantly differs from 0.
    if len(median_corr_non) > 0:
        try:
            stat_non, p_non = wilcoxon(median_corr_non)
            print(f"Wilcoxon test for Non-dominant: statistic = {stat_non:.2f}, p = {p_non:.4f}")
        except Exception as e:
            print("Wilcoxon test for Non-dominant could not be performed.")
    else:
        print("No data available for Non-dominant Wilcoxon test.")
    
    if len(median_corr_dom) > 0:
        try:
            stat_dom, p_dom = wilcoxon(median_corr_dom)
            print(f"Wilcoxon test for Dominant: statistic = {stat_dom:.2f}, p = {p_dom:.4f}")
        except Exception as e:
            print("Wilcoxon test for Dominant could not be performed.")
    else:
        print("No data available for Dominant Wilcoxon test.")

boxplot_spearman_corr_with_stats_reach_indices_by_subject(SAT_corr_within_results, figsize=(4,3))

# # -------------------------------------------------------------------------------------------------------------------
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

    # --- Prepare dictionaries for raw and z-scored data ---
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

    # --- Z-scored Scatter Plots Per Reach Index with Diagonal and Perpendicular Distances ---
    if show_plots:
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
        for i in range(num_reaches):
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
                # The projection of (x, y) onto the line x = y is:
                proj_x = (x_val + y_val) / 2
                proj_y = proj_x  # Since the line is x = y

                # The distance from (0,0) to the projection point along the line (with sign)
                # is given by the dot product of the projection point with the unit vector along (1,1)
                projection_distance = (x_val + y_val) / math.sqrt(2)

                ax.plot([x_val, proj_x], [y_val, proj_y], color='magenta', linestyle=':', linewidth=0.8)
                ax.scatter(proj_x, proj_y, color='orange', marker='x', s=30)
                perp_distances.append(projection_distance)

            zscore_perp_distances[i + 1] = perp_distances
            ax.legend()
            ax.grid(True)
        fig.suptitle(f"{subject} - {hand.capitalize()}: Z-scored Duration vs Distance", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    else:
        for i in range(num_reaches):
            perp_distances = []
            for j, (x_val, y_val) in enumerate(zip(z_durations_matrix[:, i], z_distance_matrix[:, i])):
                projection_distance = (x_val + y_val) / math.sqrt(2)
                perp_distances.append(projection_distance)
            zscore_perp_distances[i + 1] = perp_distances

    # Save the raw data along with the z-scored data and perpendicular distances in the global structure.
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

def compare_raw_vs_processed(subject, hand, reach_index, show_plots=True, figsize=(12,6)):
    """
    Create a side-by-side comparison of raw vs processed scatter plots
    for a single reach (reach_index).
    """
    global updated_metrics_zscore

    # Extract saved data
    raw_durations = updated_metrics_zscore[subject][hand]['durations'][reach_index]
    raw_distance = updated_metrics_zscore[subject][hand]['distance'][reach_index]
    zscore_durations = updated_metrics_zscore[subject][hand]['zscore_durations'][reach_index]
    zscore_distance = updated_metrics_zscore[subject][hand]['zscore_distance'][reach_index]
    perp_distances = updated_metrics_zscore[subject][hand]['MotorAcuity'][reach_index]

    markersize = 50
    if show_plots:
        fig, axs = plt.subplots(1, 2, figsize=figsize)

        # --- Left: Raw Data ---
        axs[0].scatter(raw_durations, raw_distance, color='blue', s=markersize)
        axs[0].set_xlabel("Duration (s)", fontsize=14)
        axs[0].set_ylabel("Distance (mm)", fontsize=14)
        axs[0].set_xlim(0.6, 0.9)
        axs[0].set_ylim(0, 6)
        axs[0].set_xticks([0.6, 0.75, 0.9])
        axs[0].set_yticks([0, 3, 6])
        axs[0].tick_params(axis='both', labelsize=12)
        axs[0].grid(False)

        # --- Right: Processed Data ---
        axs[1].scatter(zscore_durations, zscore_distance, color='grey', label="Z-scored Data", s=markersize, alpha=0.4)
        # axs[1].axhline(y=0, color='lightgrey', linestyle='--', linewidth=0.8)
        # axs[1].axvline(x=0, color='lightgrey', linestyle='--', linewidth=0.8)

        # Add diagonal line
        min_lim = min(min(zscore_durations), min(zscore_distance))
        max_lim = max(max(zscore_durations), max(zscore_distance))
        x_vals = np.linspace(min_lim, max_lim, 100)
        axs[1].plot(x_vals, x_vals, color='lightgrey', linestyle='--', label='45° line', linewidth=2, zorder=1)

        # Overlay perpendicular distances
        for x_val, y_val, proj_dist in zip(zscore_durations, zscore_distance, perp_distances):
            proj_x = (x_val + y_val) / 2
            proj_y = proj_x
            axs[1].plot([x_val, proj_x], [y_val, proj_y], color='lightgrey', linestyle=':', linewidth=1.5, zorder=1)
            axs[1].scatter(proj_x, proj_y, color='red', marker='x', s=50, zorder=2)

        axs[1].set_xlabel("Z-scored Duration", fontsize=14)
        axs[1].set_ylabel("Z-scored Distance", fontsize=14)
        axs[1].set_xlim(-2.2, 2.2)
        axs[1].set_ylim(-2.2, 2.2)
        axs[1].set_xticks([-2, 0, 2])
        axs[1].set_yticks([-2, 0, 2])
        axs[1].tick_params(axis='both', labelsize=12)
        axs[1].grid(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# Example call
compare_raw_vs_processed("07/22/HW", "non_dominant", reach_index=10, show_plots=True, figsize=(8,4))
# # -------------------------------------------------------------------------------------------------------------------

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


def plot_median_metrics(subject_medians, overall_median_motor_acuity, hand_order=["non_dominant", "dominant"], figsize=(12, 6)):
    """
    Plots boxplots of median Duration and Distance per hand using the subject_medians dictionary,
    and adds a third subplot for the subjects' median MotorAcuity using overall_median_motor_acuity.

    The subject_medians dictionary is expected to have the structure:
      { subject: { hand: { 'durations': median_duration, 'distance': median_distance, ... } } }
    The overall_median_motor_acuity dictionary is expected to have the structure:
      { subject: { hand: overall_median, ... } }

    Parameters:
        subject_medians (dict): Dictionary containing median Duration and Distance per subject.
        overall_median_motor_acuity (dict): Dictionary containing overall median MotorAcuity per subject and hand.
        hand_order (list): List specifying the order of hand categories.
        figsize (tuple): Figure size for the overall plot (in inches).
    """
    # Convert subject_medians into DataFrame for Duration and Distance.
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
    
    # Map hand keys to nicely formatted labels and update order.
    hand_labels = {"dominant": "Dominant", "non_dominant": "Non-dominant"}
    df["Hand"] = df["Hand"].map(hand_labels)
    hand_order_labels = [hand_labels[h] for h in hand_order]
    
    # Convert overall_median_motor_acuity into a DataFrame.
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
    
    # Create a figure with three subplots side-by-side.
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot for Duration.
    sns.boxplot(
        data=df, x="Hand", y="Duration", 
        order=hand_order_labels, ax=axes[0],
        palette=["#D3D3D3", "#F0F0F0"]
    )
    sns.swarmplot(
        data=df, x="Hand", y="Duration",
        order=hand_order_labels, ax=axes[0],
        color='black', size=6
    )
    axes[0].set_xlabel("Hand")
    axes[0].set_ylabel("Duration (s)")
    axes[0].set_ylim(0, 1.2)
    axes[0].set_yticks([0, 0.6, 1.2])
    
    # Plot for Distance.
    sns.boxplot(
        data=df, x="Hand", y="Distance",
        order=hand_order_labels, ax=axes[1],
        palette=["#D3D3D3", "#F0F0F0"]
    )
    sns.swarmplot(
        data=df, x="Hand", y="Distance",
        order=hand_order_labels, ax=axes[1],
        color='black', size=6
    )
    axes[1].set_xlabel("Hand")
    axes[1].set_ylabel("Error Distance (mm)")
    axes[1].set_ylim(0, 5)
    axes[1].set_yticks([0, 2.5, 5])
    
    # Plot for MotorAcuity.
    sns.boxplot(
        data=df_z, x="Hand", y="MotorAcuity",
        order=hand_order_labels, ax=axes[2],
        palette=["#D3D3D3", "#F0F0F0"]
    )
    sns.swarmplot(
        data=df_z, x="Hand", y="MotorAcuity",
        order=hand_order_labels, ax=axes[2],
        color='black', size=6
    )
    axes[2].set_xlabel("Hand")
    axes[2].set_ylabel("Motor Acuity")
    axes[2].set_ylim(-0.25, 0)
    
    plt.tight_layout()
    plt.show()

plot_median_metrics(subject_medians, overall_median_motor_acuity, hand_order=["non_dominant", "dominant"], figsize=(8, 4))



# # -------------------------------------------------------------------------------------------------------------------
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

## -------------------------------------------------------------------------------------------------------------------

def plot_overall_median_correlations(saved_heatmaps, plot_type='bar', overlay_points=True):
    import matplotlib.pyplot as plt

    # Extract the row medians dictionaries for each variable and phase.
    # For "durations":
    tw_durations = saved_heatmaps[('TW_LDLJ', 'durations')]['medians']['non_dominant']['row_medians']
    ballistic_durations = saved_heatmaps[('ballistic_LDLJ', 'durations')]['medians']['non_dominant']['row_medians']
    correction_durations = saved_heatmaps[('correction_LDLJ', 'durations')]['medians']['non_dominant']['row_medians']

    # For "distance":
    tw_distance = saved_heatmaps[('TW_LDLJ', 'distance')]['medians']['non_dominant']['row_medians']
    ballistic_distance = saved_heatmaps[('ballistic_LDLJ', 'distance')]['medians']['non_dominant']['row_medians']
    correction_distance = saved_heatmaps[('correction_LDLJ', 'distance')]['medians']['non_dominant']['row_medians']

    # For "MotorAcuity":
    tw_motoracuity = saved_heatmaps[('TW_LDLJ', 'MotorAcuity')]['medians']['non_dominant']['row_medians']
    ballistic_motoracuity = saved_heatmaps[('ballistic_LDLJ', 'MotorAcuity')]['medians']['non_dominant']['row_medians']
    correction_motoracuity = saved_heatmaps[('correction_LDLJ', 'MotorAcuity')]['medians']['non_dominant']['row_medians']

    # Helper function: compute median and symmetric error (IQR/2)
    def compute_median_and_error(data_dict):
        values = np.array(list(data_dict.values()))
        median = np.median(values)
        iqr = np.percentile(values, 75) - np.percentile(values, 25)
        error = iqr / 2.0
        return median, error

    # Compute overall medians and error bars for each phase.
    duration_tw, err_duration_tw = compute_median_and_error(tw_durations)
    duration_ball, err_duration_ball = compute_median_and_error(ballistic_durations)
    duration_corr, err_duration_corr = compute_median_and_error(correction_durations)

    distance_tw, err_distance_tw = compute_median_and_error(tw_distance)
    distance_ball, err_distance_ball = compute_median_and_error(ballistic_distance)
    distance_corr, err_distance_corr = compute_median_and_error(correction_distance)

    motoracuity_tw, err_motoracuity_tw = compute_median_and_error(tw_motoracuity)
    motoracuity_ball, err_motoracuity_ball = compute_median_and_error(ballistic_motoracuity)
    motoracuity_corr, err_motoracuity_corr = compute_median_and_error(correction_motoracuity)

    duration_vals = [duration_tw, duration_ball, duration_corr]
    duration_errs = [err_duration_tw, err_duration_ball, err_duration_corr]

    distance_vals = [distance_tw, distance_ball, distance_corr]
    distance_errs = [err_distance_tw, err_distance_ball, err_distance_corr]

    motoracuity_vals = [motoracuity_tw, motoracuity_ball, motoracuity_corr]
    motoracuity_errs = [err_motoracuity_tw, err_motoracuity_ball, err_motoracuity_corr]

    categories = ['TW', 'Ballistic', 'Correction']
    # Define the colors: white for TW, blue for ballistic and red for correction.
    colors = ['white', 'blue', 'red']

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # --- Durations subplot ---
    ax = axs[0]
    if plot_type == 'bar':
        ax.bar(categories, duration_vals, color=colors, yerr=duration_errs, capsize=5, edgecolor='black')
        ax.set_title("Durations")
        ax.set_ylabel("Median Correlation")
        # ax.set_ylim(-1, 1)                # Set y-axis limits
        # ax.set_yticks([-1, 0, 1])         # Set y-axis tick labels
        if overlay_points:
            for i, data_dict in enumerate([tw_durations, ballistic_durations, correction_durations]):
                pts = np.array(list(data_dict.values()))
                jitter = np.random.uniform(-0.1, 0.1, size=pts.shape)
                ax.scatter(np.full(pts.shape, i) + jitter, pts, color='black', zorder=5, alpha=0.5)
    elif plot_type == 'box':
        data = [list(tw_durations.values()), list(ballistic_durations.values()), list(correction_durations.values())]
        bp = ax.boxplot(data, positions=range(len(categories)), patch_artist=True)
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_title("Durations")
        ax.set_ylabel("Median Correlation")
        # ax.set_ylim(-1, 1)
        # ax.set_yticks([-1, 0, 1])
        if overlay_points:
            for i, arr in enumerate(data):
                pts = np.array(arr)
                jitter = np.random.uniform(-0.1, 0.1, size=pts.shape)
                ax.scatter(np.full(pts.shape, i) + jitter, pts, color='black', zorder=5, alpha=0.5)

    # --- Distance subplot ---
    ax = axs[1]
    if plot_type == 'bar':
        ax.bar(categories, distance_vals, color=colors, yerr=distance_errs, capsize=5, edgecolor='black')
        ax.set_title("Distance")
        # ax.set_ylabel("Median Correlation")
        # ax.set_ylim(-1, 1)
        # ax.set_yticks([-1, 0, 1])
        if overlay_points:
            for i, data_dict in enumerate([tw_distance, ballistic_distance, correction_distance]):
                pts = np.array(list(data_dict.values()))
                jitter = np.random.uniform(-0.1, 0.1, size=pts.shape)
                ax.scatter(np.full(pts.shape, i) + jitter, pts, color='black', zorder=5, alpha=0.5)
    elif plot_type == 'box':
        data = [list(tw_distance.values()), list(ballistic_distance.values()), list(correction_distance.values())]
        bp = ax.boxplot(data, positions=range(len(categories)), patch_artist=True)
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_title("Distance")
        # ax.set_ylabel("Median Correlation")
        # ax.set_ylim(-1, 1)
        # ax.set_yticks([-1, 0, 1])
        if overlay_points:
            for i, arr in enumerate(data):
                pts = np.array(arr)
                jitter = np.random.uniform(-0.1, 0.1, size=pts.shape)
                ax.scatter(np.full(pts.shape, i) + jitter, pts, color='black', zorder=5, alpha=0.5)

    # --- MotorAcuity subplot ---
    ax = axs[2]
    if plot_type == 'bar':
        ax.bar(categories, motoracuity_vals, color=colors, yerr=motoracuity_errs, capsize=5, edgecolor='black')
        ax.set_title("MotorAcuity")
        # ax.set_ylabel("Median Correlation")
        # ax.set_ylim(-1, 1)
        # ax.set_yticks([-1, 0, 1])
        if overlay_points:
            for i, data_dict in enumerate([tw_motoracuity, ballistic_motoracuity, correction_motoracuity]):
                pts = np.array(list(data_dict.values()))
                jitter = np.random.uniform(-0.1, 0.1, size=pts.shape)
                ax.scatter(np.full(pts.shape, i) + jitter, pts, color='black', zorder=5, alpha=0.5)
    elif plot_type == 'box':
        data = [list(tw_motoracuity.values()), list(ballistic_motoracuity.values()), list(correction_motoracuity.values())]
        bp = ax.boxplot(data, positions=range(len(categories)), patch_artist=True)
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_title("MotorAcuity")
        # ax.set_ylabel("Median Correlation")
        # ax.set_ylim(-1, 1)
        # ax.set_yticks([-1, 0, 1])
        if overlay_points:
            for i, arr in enumerate(data):
                pts = np.array(arr)
                jitter = np.random.uniform(-0.1, 0.1, size=pts.shape)
                ax.scatter(np.full(pts.shape, i) + jitter, pts, color='black', zorder=5, alpha=0.5)

    plt.tight_layout()
    plt.show()

plot_overall_median_correlations(saved_heatmaps, plot_type='bar', overlay_points=False)

def plot_grouped_median_correlations(saved_heatmaps, hand='non_dominant', overlay_points=True, figuresize=(10, 5)):  
    import numpy as np
    import matplotlib.pyplot as plt

    # Extract data for the specified hand
    tw_durations = saved_heatmaps[('TW_LDLJ', 'durations')]['medians'][hand]['row_medians']
    ballistic_durations = saved_heatmaps[('ballistic_LDLJ', 'durations')]['medians'][hand]['row_medians']
    correction_durations = saved_heatmaps[('correction_LDLJ', 'durations')]['medians'][hand]['row_medians']

    tw_distance = saved_heatmaps[('TW_LDLJ', 'distance')]['medians'][hand]['row_medians']
    ballistic_distance = saved_heatmaps[('ballistic_LDLJ', 'distance')]['medians'][hand]['row_medians']
    correction_distance = saved_heatmaps[('correction_LDLJ', 'distance')]['medians'][hand]['row_medians']

    tw_motoracuity = saved_heatmaps[('TW_LDLJ', 'MotorAcuity')]['medians'][hand]['row_medians']
    ballistic_motoracuity = saved_heatmaps[('ballistic_LDLJ', 'MotorAcuity')]['medians'][hand]['row_medians']
    correction_motoracuity = saved_heatmaps[('correction_LDLJ', 'MotorAcuity')]['medians'][hand]['row_medians']

    # Helper function: compute median and symmetric error (IQR/2)
    def compute_median_and_error(data_dict):
        values = np.array(list(data_dict.values()))
        median = np.median(values)
        iqr = np.percentile(values, 75) - np.percentile(values, 25)
        error = iqr / 2.0
        return median, error

    # Compute medians and errors for each metric and phase
    duration_vals = []
    duration_errs = []
    for data in [tw_durations, ballistic_durations, correction_durations]:
        m, e = compute_median_and_error(data)
        duration_vals.append(m)
        duration_errs.append(e)

    distance_vals = []
    distance_errs = []
    for data in [tw_distance, ballistic_distance, correction_distance]:
        m, e = compute_median_and_error(data)
        distance_vals.append(m)
        distance_errs.append(e)

    motoracuity_vals = []
    motoracuity_errs = []
    for data in [tw_motoracuity, ballistic_motoracuity, correction_motoracuity]:
        m, e = compute_median_and_error(data)
        motoracuity_vals.append(m)
        motoracuity_errs.append(e)

    # Combine into arrays for plotting
    all_vals = np.array([duration_vals, distance_vals, motoracuity_vals])  # shape (3 groups, 3 bars each)
    all_errs = np.array([duration_errs, distance_errs, motoracuity_errs])

    categories = ['Durations', 'Distance', 'MotorAcuity']
    bar_labels = ['TW', 'Ballistic', 'Correction']
    colors = ['white', 'blue', 'red']
    
    # If non_dominant hand, add hatch pattern '//'
    hatch_pattern = '//' if hand == 'non_dominant' else ''

    x = np.arange(len(categories))  # x locations for groups
    width = 0.25  # width of each bar

    fig, ax = plt.subplots(figsize=figuresize)

    for i in range(3):
        ax.bar(x + i * width - width, all_vals[:, i], width, yerr=all_errs[:, i],
               capsize=5, color=colors[i], edgecolor='black', label=bar_labels[i],
               hatch=hatch_pattern)

        if overlay_points:
            # List all dictionaries in order: for each category: durations, then distance, then MotorAcuity
            data_dicts = [tw_durations, ballistic_durations, correction_durations,
                          tw_distance, ballistic_distance, correction_distance,
                          tw_motoracuity, ballistic_motoracuity, correction_motoracuity]
            for j in range(len(categories)):
                idx = j * 3 + i
                pts = np.array(list(data_dicts[idx].values()))
                jitter = np.random.uniform(-0.05, 0.05, size=pts.shape)
                ax.scatter(np.full(pts.shape, x[j] + i * width - width) + jitter, pts,
                           color='black', zorder=5, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=14)
    ax.set_ylabel("Median Correlation", fontsize=14)
    ax.legend(fontsize=12, frameon=False, loc='lower right')
    plt.tight_layout()
    plt.show()

# Example call for non_dominant hand:
plot_grouped_median_correlations(saved_heatmaps, hand='non_dominant', overlay_points=False, figuresize=(4, 3))

def plot_grouped_median_correlations(saved_heatmaps, hand='dominant', overlay_points=True, figuresize=(10, 5), plot_both_hands=False):
    import numpy as np
    import matplotlib.pyplot as plt

    hands_to_plot = ['dominant', 'non_dominant'] if plot_both_hands else [hand]

    # Helper function: compute median and symmetric error (IQR/2)
    def compute_median_and_error(data_dict):
        values = np.array(list(data_dict.values()))
        median = np.median(values)
        iqr = np.percentile(values, 75) - np.percentile(values, 25)
        error = iqr / 2.0
        return median, error

    # Extract and compute medians/errors
    all_vals = []
    all_errs = []
    for h in hands_to_plot:
        duration_vals, duration_errs = [], []
        distance_vals, distance_errs = [], []
        motoracuity_vals, motoracuity_errs = [], []

        for data in [saved_heatmaps[('TW_LDLJ', 'durations')]['medians'][h]['row_medians'],
                     saved_heatmaps[('ballistic_LDLJ', 'durations')]['medians'][h]['row_medians'],
                     saved_heatmaps[('correction_LDLJ', 'durations')]['medians'][h]['row_medians']]:
            m, e = compute_median_and_error(data)
            duration_vals.append(m)
            duration_errs.append(e)

        for data in [saved_heatmaps[('TW_LDLJ', 'distance')]['medians'][h]['row_medians'],
                     saved_heatmaps[('ballistic_LDLJ', 'distance')]['medians'][h]['row_medians'],
                     saved_heatmaps[('correction_LDLJ', 'distance')]['medians'][h]['row_medians']]:
            m, e = compute_median_and_error(data)
            distance_vals.append(m)
            distance_errs.append(e)

        for data in [saved_heatmaps[('TW_LDLJ', 'MotorAcuity')]['medians'][h]['row_medians'],
                     saved_heatmaps[('ballistic_LDLJ', 'MotorAcuity')]['medians'][h]['row_medians'],
                     saved_heatmaps[('correction_LDLJ', 'MotorAcuity')]['medians'][h]['row_medians']]:
            m, e = compute_median_and_error(data)
            motoracuity_vals.append(m)
            motoracuity_errs.append(e)

        all_vals.append([duration_vals, distance_vals, motoracuity_vals])
        all_errs.append([duration_errs, distance_errs, motoracuity_errs])

    all_vals = np.array(all_vals)  # shape (hands, groups, bars)
    all_errs = np.array(all_errs)

    categories = ['Durations', 'Distance', 'MotorAcuity']
    bar_labels = ['TW', 'Ballistic', 'Correction']
    colors = ['white', 'blue', 'red']
    hatches = [None, '//']  # dominant = None, non_dominant = '//'

    n_hands = len(hands_to_plot)
    x = np.arange(len(categories))  # x locations for groups
    width = 0.25  # width of each bar
    fig, ax = plt.subplots(figsize=figuresize)

    for i in range(3):  # for each bar type (TW, Ballistic, Correction)
        for h_idx, h in enumerate(hands_to_plot):
            ax.bar(
                x + i * width - width/2 + h_idx * width * 0.6,  # slight offset for hands
                all_vals[h_idx, :, i],
                width * 0.6,
                yerr=all_errs[h_idx, :, i],
                capsize=5,
                color=colors[i],
                edgecolor='black',
                label=f"{bar_labels[i]} ({h})" if i == 0 else "",
                hatch=hatches[h_idx]
            )


            if overlay_points:
                data_dicts = [
                    saved_heatmaps[('TW_LDLJ', 'durations')]['medians'][h]['row_medians'],
                    saved_heatmaps[('ballistic_LDLJ', 'durations')]['medians'][h]['row_medians'],
                    saved_heatmaps[('correction_LDLJ', 'durations')]['medians'][h]['row_medians'],
                    saved_heatmaps[('TW_LDLJ', 'distance')]['medians'][h]['row_medians'],
                    saved_heatmaps[('ballistic_LDLJ', 'distance')]['medians'][h]['row_medians'],
                    saved_heatmaps[('correction_LDLJ', 'distance')]['medians'][h]['row_medians'],
                    saved_heatmaps[('TW_LDLJ', 'MotorAcuity')]['medians'][h]['row_medians'],
                    saved_heatmaps[('ballistic_LDLJ', 'MotorAcuity')]['medians'][h]['row_medians'],
                    saved_heatmaps[('correction_LDLJ', 'MotorAcuity')]['medians'][h]['row_medians']
                ]
                for j in range(len(categories)):
                    idx = j * 3 + i
                    pts = np.array(list(data_dicts[idx].values()))
                    jitter = np.random.uniform(-0.02, 0.02, size=pts.shape)
                    ax.scatter(
                        np.full(pts.shape, x[j] + i * width - width/2 + h_idx * width * 0.6) + jitter,
                        pts,
                        color='black',
                        zorder=5,
                        alpha=0.5
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=14)
    ax.set_ylabel("Median Correlation", fontsize=14)
    ax.legend(fontsize=12, frameon=False, loc='lower right')
    plt.tight_layout()
    plt.show()
# Example call for both hands:
plot_grouped_median_correlations(saved_heatmaps, hand='non_dominant', overlay_points=False, figuresize=(4, 3), plot_both_hands=False)

#-------------------------------------------------------------------------------------------------------------------

def plot_overall_median_correlations(saved_heatmaps, plot_type='bar', overlay_points=True, metric_type='LDLJ'):
    import matplotlib.pyplot as plt
    # Extract the row medians dictionaries for each variable and phase using the chosen metric type.
    # For "durations":
    tw_durations = saved_heatmaps[(f"TW_{metric_type}", 'durations')]['medians']['non_dominant']['row_medians']
    ballistic_durations = saved_heatmaps[(f"ballistic_{metric_type}", 'durations')]['medians']['non_dominant']['row_medians']
    correction_durations = saved_heatmaps[(f"correction_{metric_type}", 'durations')]['medians']['non_dominant']['row_medians']

    # For "distance":
    tw_distance = saved_heatmaps[(f"TW_{metric_type}", 'distance')]['medians']['non_dominant']['row_medians']
    ballistic_distance = saved_heatmaps[(f"ballistic_{metric_type}", 'distance')]['medians']['non_dominant']['row_medians']
    correction_distance = saved_heatmaps[(f"correction_{metric_type}", 'distance')]['medians']['non_dominant']['row_medians']

    # For "MotorAcuity":
    tw_motoracuity = saved_heatmaps[(f"TW_{metric_type}", 'MotorAcuity')]['medians']['non_dominant']['row_medians']
    ballistic_motoracuity = saved_heatmaps[(f"ballistic_{metric_type}", 'MotorAcuity')]['medians']['non_dominant']['row_medians']
    correction_motoracuity = saved_heatmaps[(f"correction_{metric_type}", 'MotorAcuity')]['medians']['non_dominant']['row_medians']

    # Helper function: compute median and symmetric error (IQR/2)
    def compute_median_and_error(data_dict):
        values = np.array(list(data_dict.values()))
        median = np.median(values)
        iqr = np.percentile(values, 75) - np.percentile(values, 25)
        error = iqr / 2.0
        return median, error

    # Compute overall medians and error bars for each phase.
    duration_tw, err_duration_tw = compute_median_and_error(tw_durations)
    duration_ball, err_duration_ball = compute_median_and_error(ballistic_durations)
    duration_corr, err_duration_corr = compute_median_and_error(correction_durations)

    distance_tw, err_distance_tw = compute_median_and_error(tw_distance)
    distance_ball, err_distance_ball = compute_median_and_error(ballistic_distance)
    distance_corr, err_distance_corr = compute_median_and_error(correction_distance)

    motoracuity_tw, err_motoracuity_tw = compute_median_and_error(tw_motoracuity)
    motoracuity_ball, err_motoracuity_ball = compute_median_and_error(ballistic_motoracuity)
    motoracuity_corr, err_motoracuity_corr = compute_median_and_error(correction_motoracuity)

    duration_vals = [duration_tw, duration_ball, duration_corr]
    duration_errs = [err_duration_tw, err_duration_ball, err_duration_corr]

    distance_vals = [distance_tw, distance_ball, distance_corr]
    distance_errs = [err_distance_tw, err_distance_ball, err_distance_corr]

    motoracuity_vals = [motoracuity_tw, motoracuity_ball, motoracuity_corr]
    motoracuity_errs = [err_motoracuity_tw, err_motoracuity_ball, err_motoracuity_corr]

    categories = ['TW', 'Ballistic', 'Correction']
    # Define the colors: white for TW, blue for ballistic and red for correction.
    colors = ['white', 'blue', 'red']

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # --- Durations subplot ---
    ax = axs[0]
    if plot_type == 'bar':
        ax.bar(categories, duration_vals, color=colors, yerr=duration_errs, capsize=5, edgecolor='black')
        ax.set_title("Durations")
        ax.set_ylabel("Median Correlation")
        if overlay_points:
            for i, data_dict in enumerate([tw_durations, ballistic_durations, correction_durations]):
                pts = np.array(list(data_dict.values()))
                jitter = np.random.uniform(-0.1, 0.1, size=pts.shape)
                ax.scatter(np.full(pts.shape, i) + jitter, pts, color='black', zorder=5, alpha=0.5)
    elif plot_type == 'box':
        data = [list(tw_durations.values()), list(ballistic_durations.values()), list(correction_durations.values())]
        bp = ax.boxplot(data, positions=range(len(categories)), patch_artist=True)
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_title("Durations")
        ax.set_ylabel("Median Correlation")
        if overlay_points:
            for i, arr in enumerate(data):
                pts = np.array(arr)
                jitter = np.random.uniform(-0.1, 0.1, size=pts.shape)
                ax.scatter(np.full(pts.shape, i) + jitter, pts, color='black', zorder=5, alpha=0.5)

    # --- Distance subplot ---
    ax = axs[1]
    if plot_type == 'bar':
        ax.bar(categories, distance_vals, color=colors, yerr=distance_errs, capsize=5, edgecolor='black')
        ax.set_title("Distance")
        if overlay_points:
            for i, data_dict in enumerate([tw_distance, ballistic_distance, correction_distance]):
                pts = np.array(list(data_dict.values()))
                jitter = np.random.uniform(-0.1, 0.1, size=pts.shape)
                ax.scatter(np.full(pts.shape, i) + jitter, pts, color='black', zorder=5, alpha=0.5)
    elif plot_type == 'box':
        data = [list(tw_distance.values()), list(ballistic_distance.values()), list(correction_distance.values())]
        bp = ax.boxplot(data, positions=range(len(categories)), patch_artist=True)
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_title("Distance")
        if overlay_points:
            for i, arr in enumerate(data):
                pts = np.array(arr)
                jitter = np.random.uniform(-0.1, 0.1, size=pts.shape)
                ax.scatter(np.full(pts.shape, i) + jitter, pts, color='black', zorder=5, alpha=0.5)

    # --- MotorAcuity subplot ---
    ax = axs[2]
    if plot_type == 'bar':
        ax.bar(categories, motoracuity_vals, color=colors, yerr=motoracuity_errs, capsize=5, edgecolor='black')
        ax.set_title("MotorAcuity")
        if overlay_points:
            for i, data_dict in enumerate([tw_motoracuity, ballistic_motoracuity, correction_motoracuity]):
                pts = np.array(list(data_dict.values()))
                jitter = np.random.uniform(-0.1, 0.1, size=pts.shape)
                ax.scatter(np.full(pts.shape, i) + jitter, pts, color='black', zorder=5, alpha=0.5)
    elif plot_type == 'box':
        data = [list(tw_motoracuity.values()), list(ballistic_motoracuity.values()), list(correction_motoracuity.values())]
        bp = ax.boxplot(data, positions=range(len(categories)), patch_artist=True)
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_title("MotorAcuity")
        if overlay_points:
            for i, arr in enumerate(data):
                pts = np.array(arr)
                jitter = np.random.uniform(-0.1, 0.1, size=pts.shape)
                ax.scatter(np.full(pts.shape, i) + jitter, pts, color='black', zorder=5, alpha=0.5)

    plt.tight_layout()
    plt.show()

plot_overall_median_correlations(saved_heatmaps, plot_type='bar', overlay_points=False, metric_type='sparc')


def plot_grouped_median_correlations(saved_heatmaps, hand='non_dominant', overlay_points=True, figuresize=(10, 5), metric='ldlj'):
    """
    Plots grouped median correlations from saved_heatmaps for either LDLJ or SPARC metrics.
    
    Parameters:
        saved_heatmaps (dict): Dictionary containing heatmap data keyed by (phase_metric, variable).
        hand (str): 'non_dominant' or 'dominant'.
        overlay_points (bool): Whether to overlay individual subject points.
        figuresize (tuple): Figure size in inches.
        metric (str): Choose between "ldlj" or "sparc" (case-insensitive).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    prefix = metric
    # Extract data for the specified hand using the chosen metric prefix
    tw_durations = saved_heatmaps[(f"TW_{prefix}", 'durations')]['medians'][hand]['row_medians']
    ballistic_durations = saved_heatmaps[(f"ballistic_{prefix}", 'durations')]['medians'][hand]['row_medians']
    correction_durations = saved_heatmaps[(f"correction_{prefix}", 'durations')]['medians'][hand]['row_medians']

    tw_distance = saved_heatmaps[(f"TW_{prefix}", 'distance')]['medians'][hand]['row_medians']
    ballistic_distance = saved_heatmaps[(f"ballistic_{prefix}", 'distance')]['medians'][hand]['row_medians']
    correction_distance = saved_heatmaps[(f"correction_{prefix}", 'distance')]['medians'][hand]['row_medians']

    tw_motoracuity = saved_heatmaps[(f"TW_{prefix}", 'MotorAcuity')]['medians'][hand]['row_medians']
    ballistic_motoracuity = saved_heatmaps[(f"ballistic_{prefix}", 'MotorAcuity')]['medians'][hand]['row_medians']
    correction_motoracuity = saved_heatmaps[(f"correction_{prefix}", 'MotorAcuity')]['medians'][hand]['row_medians']

    # Helper function: compute median and symmetric error (IQR/2)
    def compute_median_and_error(data_dict):
        values = np.array(list(data_dict.values()))
        median = np.median(values)
        iqr = np.percentile(values, 75) - np.percentile(values, 25)
        error = iqr / 2.0
        return median, error

    # Compute medians and errors for each metric and phase
    duration_vals = []
    duration_errs = []
    for data in [tw_durations, ballistic_durations, correction_durations]:
        m, e = compute_median_and_error(data)
        duration_vals.append(m)
        duration_errs.append(e)

    distance_vals = []
    distance_errs = []
    for data in [tw_distance, ballistic_distance, correction_distance]:
        m, e = compute_median_and_error(data)
        distance_vals.append(m)
        distance_errs.append(e)

    motoracuity_vals = []
    motoracuity_errs = []
    for data in [tw_motoracuity, ballistic_motoracuity, correction_motoracuity]:
        m, e = compute_median_and_error(data)
        motoracuity_vals.append(m)
        motoracuity_errs.append(e)

    # Combine into arrays for plotting (rows: metrics - durations, distance, motoracuity)
    all_vals = np.array([duration_vals, distance_vals, motoracuity_vals])
    all_errs = np.array([duration_errs, distance_errs, motoracuity_errs])

    categories = ['Durations', 'Distance', 'MotorAcuity']
    bar_labels = ['TW', 'Ballistic', 'Correction']
    colors = ['white', 'blue', 'red']
    
    # If non_dominant hand, add hatch pattern '//'
    hatch_pattern = '//' if hand == 'non_dominant' else ''

    x = np.arange(len(categories))  # x locations for groups
    width = 0.25  # width of each bar

    fig, ax = plt.subplots(figsize=figuresize)

    for i in range(3):
        ax.bar(x + i * width - width, all_vals[:, i], width, yerr=all_errs[:, i],
               capsize=5, color=colors[i], edgecolor='black', label=bar_labels[i],
               hatch=hatch_pattern)

        if overlay_points:
            # Order: durations, then distance, then MotorAcuity
            data_dicts = [
                tw_durations, ballistic_durations, correction_durations,
                tw_distance, ballistic_distance, correction_distance,
                tw_motoracuity, ballistic_motoracuity, correction_motoracuity
            ]
            for j in range(len(categories)):
                idx = j * 3 + i
                pts = np.array(list(data_dicts[idx].values()))
                jitter = np.random.uniform(-0.05, 0.05, size=pts.shape)
                ax.scatter(np.full(pts.shape, x[j] + i * width - width) + jitter, pts,
                           color='black', zorder=5, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=14)
    ax.set_ylabel("Median Correlation", fontsize=14)
    ax.legend(fontsize=12, frameon=False, loc='best')
    plt.tight_layout()
    plt.show()

# Example call for non_dominant hand with LDLJ metric:
plot_grouped_median_correlations(saved_heatmaps, hand='dominant', overlay_points=False, figuresize=(4, 3), metric='sparc')


def plot_grouped_median_correlations(saved_heatmaps, hand='dominant', overlay_points=True, figuresize=(10, 5), plot_both_hands=False, metric='LDLJ'):
    import numpy as np
    import matplotlib.pyplot as plt

    hands_to_plot = ['dominant', 'non_dominant'] if plot_both_hands else [hand]

    # Helper function: compute median and symmetric error (IQR/2)
    def compute_median_and_error(data_dict):
        values = np.array(list(data_dict.values()))
        median = np.median(values)
        iqr = np.percentile(values, 75) - np.percentile(values, 25)
        error = iqr / 2.0
        return median, error

    # Extract and compute medians/errors
    all_vals = []
    all_errs = []
    for h in hands_to_plot:
        duration_vals, duration_errs = [], []
        distance_vals, distance_errs = [], []
        motoracuity_vals, motoracuity_errs = [], []

        for data in [saved_heatmaps[(f"TW_{metric}", 'durations')]['medians'][h]['row_medians'],
                     saved_heatmaps[(f"ballistic_{metric}", 'durations')]['medians'][h]['row_medians'],
                     saved_heatmaps[(f"correction_{metric}", 'durations')]['medians'][h]['row_medians']]:
            m, e = compute_median_and_error(data)
            duration_vals.append(m)
            duration_errs.append(e)

        for data in [saved_heatmaps[(f"TW_{metric}", 'distance')]['medians'][h]['row_medians'],
                     saved_heatmaps[(f"ballistic_{metric}", 'distance')]['medians'][h]['row_medians'],
                     saved_heatmaps[(f"correction_{metric}", 'distance')]['medians'][h]['row_medians']]:
            m, e = compute_median_and_error(data)
            distance_vals.append(m)
            distance_errs.append(e)

        for data in [saved_heatmaps[(f"TW_{metric}", 'MotorAcuity')]['medians'][h]['row_medians'],
                     saved_heatmaps[(f"ballistic_{metric}", 'MotorAcuity')]['medians'][h]['row_medians'],
                     saved_heatmaps[(f"correction_{metric}", 'MotorAcuity')]['medians'][h]['row_medians']]:
            m, e = compute_median_and_error(data)
            motoracuity_vals.append(m)
            motoracuity_errs.append(e)

        all_vals.append([duration_vals, distance_vals, motoracuity_vals])
        all_errs.append([duration_errs, distance_errs, motoracuity_errs])

    all_vals = np.array(all_vals)  # shape (hands, groups, bars)
    all_errs = np.array(all_errs)

    categories = ['Durations', 'Distance', 'MotorAcuity']
    bar_labels = ['TW', 'Ballistic', 'Correction']
    colors = ['white', 'blue', 'red']
    hatches = [None, '//']  # dominant = None, non_dominant = '//'

    n_hands = len(hands_to_plot)
    x = np.arange(len(categories))  # x locations for groups
    width = 0.25  # width of each bar
    fig, ax = plt.subplots(figsize=figuresize)

    for i in range(3):  # for each bar type (TW, Ballistic, Correction)
        for h_idx, h in enumerate(hands_to_plot):
            ax.bar(
                x + i * width - width/2 + h_idx * width * 0.6,  # slight offset for hands
                all_vals[h_idx, :, i],
                width * 0.6,
                yerr=all_errs[h_idx, :, i],
                capsize=5,
                color=colors[i],
                edgecolor='black',
                label=f"{bar_labels[i]} ({h})" if i == 0 else "",
                hatch=hatches[h_idx]
            )

            if overlay_points:
                data_dicts = [
                    saved_heatmaps[(f"TW_{metric}", 'durations')]['medians'][h]['row_medians'],
                    saved_heatmaps[(f"ballistic_{metric}", 'durations')]['medians'][h]['row_medians'],
                    saved_heatmaps[(f"correction_{metric}", 'durations')]['medians'][h]['row_medians'],
                    saved_heatmaps[(f"TW_{metric}", 'distance')]['medians'][h]['row_medians'],
                    saved_heatmaps[(f"ballistic_{metric}", 'distance')]['medians'][h]['row_medians'],
                    saved_heatmaps[(f"correction_{metric}", 'distance')]['medians'][h]['row_medians'],
                    saved_heatmaps[(f"TW_{metric}", 'MotorAcuity')]['medians'][h]['row_medians'],
                    saved_heatmaps[(f"ballistic_{metric}", 'MotorAcuity')]['medians'][h]['row_medians'],
                    saved_heatmaps[(f"correction_{metric}", 'MotorAcuity')]['medians'][h]['row_medians']
                ]
                for j in range(len(categories)):
                    idx = j * 3 + i
                    pts = np.array(list(data_dicts[idx].values()))
                    jitter = np.random.uniform(-0.02, 0.02, size=pts.shape)
                    ax.scatter(
                        np.full(pts.shape, x[j] + i * width - width/2 + h_idx * width * 0.6) + jitter,
                        pts,
                        color='black',
                        zorder=5,
                        alpha=0.5
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=14)
    ax.set_ylabel("Median Correlation", fontsize=14)
    ax.legend(fontsize=12, frameon=False, loc='lower right' if metric=='LDLJ' else 'best')
    plt.tight_layout()
    plt.show()
# Example call for both hands:
plot_grouped_median_correlations(saved_heatmaps, hand='non_dominant', overlay_points=False, figuresize=(4, 3), plot_both_hands=True, metric='sparc')







def create_metrics_dataframe(updated_metrics_acorss_phases, updated_metrics_zscore_by_trial, sBBTResult, MotorExperiences):
    """
    Creates a DataFrame from the combined metrics stored in the updated_metrics_acorss_phases dictionary.
    It loops over each subject, hand, trial, and location (assumed to be 16) and collects the available
    metrics into a list of dictionaries. If a 'durations' value is NaN, a message is printed and that record
    is skipped.

    Parameters:
        updated_metrics_acorss_phases (dict): A nested dictionary that contains metrics per subject, hand, and trial.
        updated_metrics_zscore_by_trial (dict): A nested dictionary containing z-scored metrics, including MotorAcuity,
                                        per subject, hand, and trial.
        sBBTResult (dict): Dictionary containing sBBTResult values per subject and hand.
        MotorExperiences (dict): Dictionary containing for each subject their attributes:
            Gender, Age, handedness, physical_h_total_weighted, musical_h_total_weighted,
            digital_h_total_weighted, overall_h_total_weighted.

    Returns:
        df (DataFrame): A pandas DataFrame with columns:
            ['Subject', 'Hand', 'Trial', 'Location', 'durations',
             'cartesian_distances', 'path_distances', 'v_peaks',
             'TW1_acc_peaks', 'TW1_jerk_peaks',
             'TW2_1_acc_peaks', 'TW2_1_jerk_peaks',
             'TW2_2_acc_peaks', 'TW2_2_jerk_peaks',
             'TW3_acc_peaks', 'TW3_jerk_peaks',
             'TW4_acc_peaks', 'TW4_jerk_peaks',
             'TW5_acc_peaks', 'TW5_jerk_peaks',
             'TW6_acc_peaks', 'TW6_jerk_peaks',
             'TW1_LDLJ', 'TW2_1_LDLJ', 'TW2_2_LDLJ', 'TW3_LDLJ', 'TW4_LDLJ', 'TW5_LDLJ', 'TW6_LDLJ',
             'TW1_sparc', 'TW2_1_sparc', 'TW2_2_sparc', 'TW3_sparc', 'TW4_sparc', 'TW5_sparc', 'TW6_sparc',
             'distance', 'MotorAcuity', 'sBBTResult',
             'Gender', 'Age', 'handedness', 'physical_h_total_weighted',
             'musical_h_total_weighted', 'digital_h_total_weighted', 'overall_h_total_weighted']
    """

    rows = []
    for subject in updated_metrics_acorss_phases:

        # Use only the last part of the subject key (e.g. "06/19/CZ" becomes "CZ")
        subject_name = subject.split('/')[-1]

        # Extract motor experience info per subject (one value each)
        motor_data = MotorExperiences.get(subject_name, {})
        gender = motor_data.get("Gender", np.nan)
        age = motor_data.get("Age", np.nan)
        handedness_attr = motor_data.get("handedness", np.nan)
        physical_h_total_weighted = motor_data.get("physical_h_total_weighted", np.nan)
        musical_h_total_weighted = motor_data.get("musical_h_total_weighted", np.nan)
        digital_h_total_weighted = motor_data.get("digital_h_total_weighted", np.nan)
        overall_h_total_weighted = motor_data.get("overall_h_total_weighted", np.nan)

        for hand in updated_metrics_acorss_phases[subject]:
            # Attempt to get the sBBTResult value for the subject and hand.
            try:
                # Get the value from the "non_dominant" column for the row where Subject matches subject_name
                sbbt_val = sBBTResult.loc[sBBTResult["Subject"] == subject_name, hand].values
            except KeyError:
                sbbt_val = np.nan

            for trial in updated_metrics_acorss_phases[subject][hand]['durations']:
                for loc in range(16):
                    duration_val = updated_metrics_acorss_phases[subject][hand]['durations'][trial][loc]
                    if not np.isnan(duration_val):
                        rows.append({
                            'Subject': subject,
                            'Hand': hand,
                            'Trial': trial,
                            'Location': loc + 1,  # Location index (1 to 16)
                            
                            # Motor experiences (one value per subject)
                            'Gender': gender,
                            'Age': age,
                            'handedness': handedness_attr,
                            'physical_h_total_weighted': physical_h_total_weighted,
                            'musical_h_total_weighted': musical_h_total_weighted,
                            'digital_h_total_weighted': digital_h_total_weighted,
                            'overall_h_total_weighted': overall_h_total_weighted,
                            
                            # sBBTResult per subject per hand
                            'sBBTResult': sbbt_val,
                            
                            # Dependent variables: task performance metrics
                            'durations': duration_val,
                            'distance': updated_metrics_acorss_phases[subject][hand]['distance'][trial][loc],
                            'MotorAcuity': updated_metrics_zscore_by_trial[subject][hand]['MotorAcuity'][trial][loc],

                            # Reach metrics
                            'cartesian_distances': updated_metrics_acorss_phases[subject][hand]['cartesian_distances'][trial][loc],
                            'path_distances': updated_metrics_acorss_phases[subject][hand]['path_distances'][trial][loc],
                            'v_peaks': updated_metrics_acorss_phases[subject][hand]['v_peaks'][trial][loc],
                            'v_peak_indices': updated_metrics_acorss_phases[subject][hand]['v_peak_indices'][trial][loc],
                            
                            # Ballistic metrics
                            'ballistic_acc_peaks': updated_metrics_acorss_phases[subject][hand]['ballistic_acc_peaks'][trial][loc],
                            'ballistic_jerk_peaks': updated_metrics_acorss_phases[subject][hand]['ballistic_jerk_peaks'][trial][loc],
                            'ballistic_LDLJ': updated_metrics_acorss_phases[subject][hand]['ballistic_LDLJ'][trial][loc],
                            'ballistic_sparc': updated_metrics_acorss_phases[subject][hand]['ballistic_sparc'][trial][loc],
                            
                            # Corrective metrics
                            'corrective_acc_peaks': updated_metrics_acorss_phases[subject][hand]['corrective_acc_peaks'][trial][loc],
                            'corrective_jerk_peaks': updated_metrics_acorss_phases[subject][hand]['corrective_jerk_peaks'][trial][loc],
                            'corrective_LDLJ': updated_metrics_acorss_phases[subject][hand]['corrective_LDLJ'][trial][loc],
                            'corrective_sparc': updated_metrics_acorss_phases[subject][hand]['corrective_sparc'][trial][loc]
                        })

                    else:
                        print(f"Skipping NaN for Subject: {subject}, Hand: {hand}, Trial: {trial}, Location: {loc+1}")
    df = pd.DataFrame(rows)

    # Calculate Dis_to_subject and Dis_to_partition based on the Location column.
    # For Dis_to_subject: Locations 1-4 -> 1, 5-8 -> 2, 9-12 -> 3, 13-16 -> 4.
    # For Dis_to_partition: Locations that are 1,5,9,13 -> 1; 2,6,10,14 -> 2; 3,7,11,15 -> 3; 4,8,12,16 -> 4.
    df['Dis_to_subject'] = ((df['Location'] - 1) // 4) + 1
    df['Dis_to_partition'] = ((df['Location'] - 1) % 4) + 1

    # Convert the selected columns to numeric values in the DataFrame using pd.to_numeric.
    cols = [
        "physical_h_total_weighted",
        "musical_h_total_weighted",
        "digital_h_total_weighted",
        "overall_h_total_weighted",
        "Age"
    ]
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure sBBTResult is a single numeric value (extract from array if necessary)
    df["sBBTResult"] = df["sBBTResult"].apply(lambda x: x[0] if isinstance(x, np.ndarray) and len(x) > 0 else np.nan)
    
    
    # Apply log transformation to 'distance' and 'durations' columns to reduce skewness.
    df["distance_log"] = np.log1p(df["distance"])
    df["durations_log"] = np.log1p(df["durations"])

    # Define the path where the DataFrame will be saved as a pickle file.
    output_pickle_file = "/Users/yilinwu/Desktop/honours data/DataProcess/df.pkl"

    # Save the DataFrame 'df' as a pickle file.
    with open(output_pickle_file, "wb") as f:
        pickle.dump(df, f)
    print(f"DataFrame saved as pickle file at: {output_pickle_file}")

    return df

# Create DataFrame from updated metrics across test windows
df = create_metrics_dataframe(updated_metrics_acorss_phases, updated_metrics_zscore_by_trial, sBBTResult, MotorExperiences)


# ---










# -------------------------------------------------------------------------------------------------------------------
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import math
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.signal import find_peaks
from matplotlib.ticker import MaxNLocator
from scipy.stats import spearmanr
from scipy.stats import wilcoxon
from scipy.stats import norm
from scipy.stats import wilcoxon, norm
import pickle
# -----------------------
# Global Styles & Colors
# -----------------------
variable_colors = {
    "ballistic_LDLJ": "#a6cee3",    # light blue
    "ballistic_sparc": "#fdb863",   # light orange
    "corrective_LDLJ": "#1f78b4",   # dark blue
    "corrective_sparc": "#e66101",  # dark orange
    "durations (s)": "#4daf4a",     # green
    "distance (mm)": "#984ea3",     # purple
    "MotorAcuity": "#e7298a"        # magenta/pink
}
LINE_WIDTH = 2
MARKER_SIZE = 6

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

def mm_to_inch(width_mm, height_mm):
    return (width_mm / 25.4, height_mm / 25.4)

def min_mid_max_ticks(ax, axis='both'):
    if axis in ('x', 'both'):
        xmin, xmax = ax.get_xlim()
        ax.set_xticks([xmin, (xmin+xmax)/2, xmax])
    if axis in ('y', 'both'):
        ymin, ymax = ax.get_ylim()
        ax.set_yticks([ymin, (ymin+ymax)/2, ymax])

def format_axis(ax, x_is_categorical=False):
    if x_is_categorical:
        min_mid_max_ticks(ax, axis='y')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    else:
        min_mid_max_ticks(ax, axis='both')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

def smart_legend(fig, axes):
    if isinstance(axes, np.ndarray):
        handles, labels = [], []
        for ax in axes.ravel():
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(), by_label.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=3,
            frameon=False
        )
    else:
        axes.legend(frameon=False, loc="upper right", fontsize=11)

hand_colors = {
    "dominant": "#7570b3",          # purple
    "non_dominant": "#66c2a5"       # teal
}
location_colors = [cm.tab20(i/20) for i in range(16)]
corr_cmap = cm.get_cmap("coolwarm")  # correlation colormap

# Line and marker defaults
MARKER_SHAPE = 'o'
LINE_WIDTH_DEFAULT = 2
MARKER_SIZE_DEFAULT = 6
figsize_mm = (90, 70)



# -----------------------
# 1. Scatter Plot
# -----------------------
def scatter_plot(x, y, xlabel="X", ylabel="Y", color=variable_colors["ballistic_LDLJ"], figsize=figsize_mm):
    fig, ax = plt.subplots(figsize=mm_to_inch(*figsize))
    ax.scatter(x, y, color=color, s=MARKER_SIZE, marker=MARKER_SHAPE)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    format_axis(ax)
    smart_legend(fig, ax)
    plt.tight_layout()
    plt.show()

# -----------------------
# 2. Histogram
# -----------------------
def histogram(values, xlabel="Value", ylabel="Count", color=variable_colors["MotorAcuity"], bins=10, figsize=figsize_mm):
    fig, ax = plt.subplots(figsize=mm_to_inch(*figsize))
    ax.hist(values, bins=bins, color=color, edgecolor='black')
    # Overlay points at y=0
    ax.scatter(values, np.zeros_like(values), color='black', s=MARKER_SIZE)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    format_axis(ax)
    smart_legend(fig, ax)
    plt.tight_layout()
    plt.show()

# -----------------------
# 3. Box Plot
# -----------------------
def box_plot(categories, values, xlabel="Category", ylabel="Value", color=variable_colors["durations (s)"], figsize=figsize_mm):
    fig, ax = plt.subplots(figsize=mm_to_inch(*figsize))
    if isinstance(color, list):
        palette = color
    else:
        palette = [color]*len(np.unique(categories))
    sns.boxplot(x=categories, y=values, palette=palette, ax=ax)
    sns.swarmplot(x=categories, y=values, color='black', size=MARKER_SIZE, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    format_axis(ax, x_is_categorical=True)
    smart_legend(fig, ax)
    plt.tight_layout()
    plt.show()

# -----------------------
# 4. Violin Plot
# -----------------------
def violin_plot(categories, values, xlabel="Category", ylabel="Value", color=variable_colors["durations (s)"], figsize=figsize_mm):
    fig, ax = plt.subplots(figsize=mm_to_inch(*figsize))
    sns.violinplot(x=categories, y=values, palette=[color]*len(np.unique(categories)), ax=ax)
    sns.stripplot(x=categories, y=values, color='black', size=MARKER_SIZE, ax=ax, jitter=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    format_axis(ax, x_is_categorical=True)
    smart_legend(fig, ax)
    plt.tight_layout()
    plt.show()

# -----------------------
# 5. 3D Plot
# -----------------------
def plot_3d(x, y, z, xlabel="X", ylabel="Y", zlabel="Z", color=variable_colors["ballistic_LDLJ"], figsize=figsize_mm):
    fig = plt.figure(figsize=mm_to_inch(*figsize))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, color=color, linewidth=LINE_WIDTH)
    ax.scatter(x, y, z, color='black', s=MARKER_SIZE)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.tight_layout()
    plt.show()



# -----------------------
# Box Plot of sBBT Scores by Hand using the box_plot function
# Aggregate the data so that each subject contributes one value (using the mean if there are multiple entries)
df_unique = df.groupby(['Subject', 'Hand'], as_index=False)['sBBTResult'].mean()

# Ensure a consistent order for the categories
df_unique['Hand'] = pd.Categorical(df_unique['Hand'], categories=["dominant", "non_dominant"], ordered=True)

# Call the box_plot function.
# Here we supply a list of colors for the palette to match the "dominant" and "non_dominant" order.
box_plot(
    categories=df_unique['Hand'],
    values=df_unique['sBBTResult'],
    xlabel="Hand",
    ylabel="sBBT score",
    color=[hand_colors["dominant"], hand_colors["non_dominant"]]
)


# -----------------------
# Pairplot-like 4x4 correlation matrix
# -----------------------
vars_to_plot = ['Var1','Var2','Var3','Var4']
n_vars = len(vars_to_plot)
fig, axes = plt.subplots(n_vars, n_vars, figsize=(12,12), sharex='col', sharey='row')

for i, var_row in enumerate(vars_to_plot):
    for j, var_col in enumerate(vars_to_plot):
        ax = axes[i,j]
        if i==j:
            # Histogram on diagonal
            for cat in df['Category'].unique():
                ax.hist(df[df['Category']==cat][var_col], bins=8,
                        color=category_colors[cat], alpha=0.7, label=cat)
        else:
            # Scatter overlay
            for cat in df['Category'].unique():
                ax.scatter(df[df['Category']==cat][var_col],
                           df[df['Category']==cat][var_row],
                           color=category_colors[cat], s=20)
        # Formatting
        if i==n_vars-1:
            ax.set_xlabel(var_col)
        else:
            ax.set_xticklabels([])
        if j==0:
            ax.set_ylabel(var_row)
        else:
            ax.set_yticklabels([])

# Legend outside
handles = [plt.Line2D([0],[0], marker='o', color='w', label=cat,
                      markerfacecolor=color, markersize=6) 
           for cat, color in category_colors.items()]
fig.legend(handles=handles, labels=category_colors.keys(),
           loc='lower center', ncol=3, frameon=False)

plt.tight_layout(rect=[0,0.05,1,1])
plt.show()


# Generate synthetic data for the speed-accuracy trade-off line plot.

# Create synthetic data for a straight line with duration in [0.5, 2] seconds
x = np.linspace(0.5, 2.0, 100)  # Movement Duration (s)
y = -10 * x + 20               # Movement Error (mm): 15 mm at 0.5 s to 0 mm at 2.0 s

# Create the figure and axis using the global size conversion (mm_to_inch, figsize_mm)
fig, ax = plt.subplots(figsize=mm_to_inch(*figsize_mm))
ax.plot(x, y, color='blue', linewidth=LINE_WIDTH)

ax.set_xlabel("Movement duration (s)")
ax.set_ylabel("Error distance (mm)")

# Set custom x-axis ticks: left tick labeled "fast" and right tick labeled "slow"
ax.set_xticks([x[0], x[-1]])
ax.set_xticklabels(["fast", "slow"])

# Set custom y-axis ticks: bottom tick labeled "accurate" in green and top tick labeled "inaccurate" in red
# Note: For our data, y[-1] is 0 mm (accurate) and y[0] is 15 mm (inaccurate)
ax.set_yticks([y[-1], y[0]])
ax.set_yticklabels(["accurate", "inaccurate"])
for tick, color in zip(ax.get_yticklabels(), ["green", "red"]):
    tick.set_color(color)
for tick, color in zip(ax.get_xticklabels(), ["green", "red"]):
    tick.set_color(color)
    
plt.tight_layout()
plt.show()
