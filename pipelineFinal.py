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

import pickle
import math
import numpy as np
from scipy.stats import zscore
from scipy.stats import wilcoxon
from scipy.stats import spearmanr
from scipy.stats import chisquare
from scipy.stats import circmean, rayleigh

import pandas as pd
import seaborn as sns

import pingouin as pg
from scipy.signal import find_peaks

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import proj3d
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib as mpl
import math
from scipy.signal import find_peaks
from scipy.signal import find_peaks, savgol_filter


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
utils8.plot_sbbt_boxplot(sBBTResult)
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
# Load Motor Experiences from CSV into a dictionary
MotorExperiences = utils7.load_motor_experiences("/Users/yilinwu/Desktop/Yilin-Honours/MotorExperience.csv")
# Calculate demographic variables from MotorExperiences
utils7.display_motor_experiences_stats(MotorExperiences)
# Update MotorExperiences with weighted scores
utils7.update_overall_h_total_weighted(MotorExperiences)
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
# -------------------------------------------------------------------------------------------------------------------

ballistic_end = utils9.calculate_phase_indices_all_files(results, test_windows_7)




def compute_two_time_windows(test_windows, ballistic_end):
    """
    Compute ballistic and corrective phases using test_windows indices and ballistic_end indices.

    Ballistic phase: from the start of test window 7 to the computed ballistic_end.
    Corrective phase: from ballistic_end to the end of test window 7.

    Parameters:
        test_windows (dict): Nested dictionary with test window indices.
                             For each subject, hand, file_path, and segment, it provides a tuple (start, end).
        ballistic_end (dict): Nested dictionary with ballistic end indices, keyed similarly.

    Returns:
        ballistic_phase (dict): Dictionary with lists of (start, ballistic_end) tuples.
        corrective_phase (dict): Dictionary with lists of (ballistic_end, end) tuples.
    """
    ballistic_phase = {
        subject: {
            hand: {
                file_path: [
                    (tw[0], ballistic_end[subject][hand][file_path][seg])
                    for seg, tw in enumerate(test_windows[subject][hand][file_path])
                ]
                for file_path in test_windows[subject][hand]
            }
            for hand in test_windows[subject]
        }
        for subject in test_windows
    }

    corrective_phase = {
        subject: {
            hand: {
                file_path: [
                    (ballistic_end[subject][hand][file_path][seg], tw[1])
                    for seg, tw in enumerate(test_windows[subject][hand][file_path])
                ]
                for file_path in test_windows[subject][hand]
            }
            for hand in test_windows[subject]
        }
        for subject in test_windows
    }

    return ballistic_phase, corrective_phase

ballistic_phase, corrective_phase = compute_two_time_windows(test_windows_7, ballistic_end)



ballistic_end["07/22/HW"]["left"]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT66.csv'][15]

ballistic_phase["07/22/HW"]["left"]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT66.csv'][15]
corrective_phase["07/22/HW"]["left"]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT66.csv'][15]
test_windows_7["07/22/HW"]["left"]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT66.csv'][15]


# Select a subject, hand, file and segment to plot
subject = "07/22/HW"
hand = "left"
file_path = '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT66.csv'
seg_index = 14

# Get the start/end indices from the computed time windows for the chosen segment
ballistic_start, ballistic_end_idx = ballistic_phase[subject][hand][file_path][seg_index]
corrective_start, corrective_end = corrective_phase[subject][hand][file_path][seg_index]

# Retrieve the trajectory coordinates from results using the appropriate marker (LFIN for left hand)
traj_data = results[subject][hand][1][file_path]['traj_data']
coord_prefix = "LFIN_"
coord_x = np.array(traj_data[coord_prefix + "X"])
coord_y = np.array(traj_data[coord_prefix + "Y"])
coord_z = np.array(traj_data[coord_prefix + "Z"])

# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot ballistic phase trajectory
ax.plot(coord_x[ballistic_start:ballistic_end_idx],
    coord_y[ballistic_start:ballistic_end_idx],
    coord_z[ballistic_start:ballistic_end_idx],
    color='cyan', linewidth=2, label='Ballistic Phase')

# Plot corrective phase trajectory
ax.plot(coord_x[corrective_start:corrective_end],
    coord_y[corrective_start:corrective_end],
    coord_z[corrective_start:corrective_end],
    color='orange', linewidth=2, label='Corrective Phase')

# Mark the start and end points of the overall segment
ax.scatter(coord_x[ballistic_start], coord_y[ballistic_start], coord_z[ballistic_start],
       color='green', s=50, label='Start')
ax.scatter(coord_x[corrective_end-1], coord_y[corrective_end-1], coord_z[corrective_end-1],
       color='red', s=50, label='End')

# Label axes and add legend
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.legend()

plt.tight_layout()
plt.show()





# Select a subject, hand, file and segment to plot
subject = "07/22/HW"
hand = "left"
file_path = '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT66.csv'
seg_index = 15

# Get the phase window indices from the computed time windows for the chosen segment
ballistic_start, ballistic_end_idx = ballistic_phase[subject][hand][file_path][seg_index]
corrective_start, corrective_end = corrective_phase[subject][hand][file_path][seg_index]
# Use the overall segment from the start of the ballistic phase to the end of the corrective phase
start_idx = ballistic_start
end_idx = corrective_end

# Retrieve the trajectory signals corresponding to position, velocity, acceleration and jerk
# (Using the 'traj_space' key and the marker "LFIN" for the left hand)
marker = "LFIN"
traj_space = results[subject][hand][1][file_path]['traj_space'][marker]
position_full = np.array(traj_space[0])
velocity_full = np.array(traj_space[1])
acceleration_full = np.array(traj_space[2])
jerk_full = np.array(traj_space[3])

# Create a time index based on sample indices
time_values = np.arange(start_idx, end_idx)

import matplotlib.pyplot as plt

# Create subplots for Position, Velocity, Acceleration and Jerk
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# Plot Position and overlay ballistic_end as a point
axs[0].plot(time_values, position_full[start_idx:end_idx], color='blue', linewidth=2, label='Position')
axs[0].scatter(ballistic_end_idx, position_full[ballistic_end_idx], color='cyan', s=100, marker='D', label='Ballistic End')
axs[0].set_ylabel("Position")
axs[0].legend()

# Plot Velocity and overlay ballistic_end as a point
axs[1].plot(time_values, velocity_full[start_idx:end_idx], color='green', linewidth=2, label='Velocity')
axs[1].scatter(ballistic_end_idx, velocity_full[ballistic_end_idx], color='cyan', s=100, marker='D', label='Ballistic End')
axs[1].set_ylabel("Velocity")
axs[1].legend()

# Plot Acceleration and overlay ballistic_end as a point
axs[2].plot(time_values, acceleration_full[start_idx:end_idx], color='orange', linewidth=2, label='Acceleration')
axs[2].scatter(ballistic_end_idx, acceleration_full[ballistic_end_idx], color='cyan', s=100, marker='D', label='Ballistic End')
axs[2].set_ylabel("Acceleration")
axs[2].legend()

# Plot Jerk and overlay ballistic_end as a point
axs[3].plot(time_values, jerk_full[start_idx:end_idx], color='red', linewidth=2, label='Jerk')
axs[3].scatter(ballistic_end_idx, jerk_full[ballistic_end_idx], color='cyan', s=100, marker='D', label='Ballistic End')
axs[3].set_ylabel("Jerk")
axs[3].set_xlabel("Sample index")
axs[3].legend()

plt.tight_layout()
plt.show()


























# all_phase_data = utils9.calculate_phase_indices_all_files(
#     results, reach_speed_segments, test_windows_7,
#     fs=200, target_samples=101, dis_phases=0.3
# )

utils9.plot_3d_trajectory_icon_all_phase_data(results, reach_speed_segments, test_windows_7,
                                       subject="07/22/HW", hand="left",
                                       file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT10.csv',
                                       seg_index=9, fs=200, target_samples=101, dis_phases=0.3,
                                       show_icon=True)

utils9.plot_trajectory_all_phase_data(results, reach_speed_segments, test_windows_7, all_phase_data,
                         subject="07/22/HW", hand="left",
                         file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT66.csv',
                         seg_index=15)

# utils9.plot_3d_trajectory_video_with_icon_all_phase_data(
#     results, reach_speed_segments, test_windows_7, all_phase_data,
#     subject="07/22/HW", hand="left",
#     file_path="/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv",
#     seg_index=3,
#     fs=200, target_samples=101, dis_phases=0.3,
#     video_save_path="trajectory_with_icon_3_new.mp4"
# )




def plot_3d_trajectory_icon_all_phase_data(results, reach_speed_segments, test_windows,
                                           subject, hand, file_path, seg_index=3,
                                           fs=200, target_samples=101, dis_phases=0.3,
                                           show_icon=True):
    """
    Computes phase indices directly from results and plots a 3D trajectory with phase colors.
    This modified version calls calculate_phase_indices internally.
    Optionally shows icons if show_icon is True.
    
    Uses figure settings that have been halved.
    
    Args:
        results: The results dictionary.
        reach_speed_segments: Reach segments (used as the first set for segmentation).
        test_windows: The second segmentation parameter.
        subject, hand, file_path: Identifiers for selecting the correct data.
        seg_index: Index of the segment to process.
        fs: Sampling frequency.
        target_samples: Number of samples for interpolation.
        dis_phases: Distance threshold for phase computation.
        show_icon: Boolean flag for displaying icons.
    """
    # Compute data directly from results.
    data = utils9.calculate_phase_indices(results, reach_speed_segments, test_windows,
                                   subject=subject, hand=hand, file_path=file_path,
                                   fs=fs, target_samples=target_samples,
                                   seg_index=seg_index, dis_phases=dis_phases)

    # Halved figure size.
    fig_size = (10/2, 8/2)  # (5, 4)
    fig3d = plt.figure(figsize=fig_size)
    ax3d = fig3d.add_subplot(111, projection='3d')

    # Plot phase lines with halved linewidths.
    ax3d.plot(
        data["coord_x_full"][data["start_seg"]:data["latency_end_idx"]],
        data["coord_y_full"][data["start_seg"]:data["latency_end_idx"]],
        data["coord_z_full"][data["start_seg"]:data["latency_end_idx"]],
        color='magenta', linewidth=2, label='Latency Phase'
    )
    ax3d.plot(
        data["coord_x_full"][data["latency_end_idx"]:data["ballistic_end"]],
        data["coord_y_full"][data["latency_end_idx"]:data["ballistic_end"]],
        data["coord_z_full"][data["latency_end_idx"]:data["ballistic_end"]],
        color='cyan', linewidth=2, label='Ballistic Phase'
    )
    ax3d.plot(
        data["coord_x_full"][data["ballistic_end"]:data["verification_start_idx"]],
        data["coord_y_full"][data["ballistic_end"]:data["verification_start_idx"]],
        data["coord_z_full"][data["ballistic_end"]:data["verification_start_idx"]],
        color='green', linewidth=2, label='Correction Phase'
    )
    ax3d.plot(
        data["coord_x_full"][data["verification_start_idx"]:data["end_seg"]],
        data["coord_y_full"][data["verification_start_idx"]:data["end_seg"]],
        data["coord_z_full"][data["verification_start_idx"]:data["end_seg"]],
        color='orange', linewidth=2, label='Verification Phase'
    )

    # Helper function to add an image at a 3D point.
    def add_image(ax, xs, ys, zs, img, zoom=0.1):
        x2, y2, _ = proj_transform(xs, ys, zs, ax.get_proj())
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, (x2, y2), frameon=False, xycoords='data')
        ax.add_artist(ab)

    # If the option is enabled, load the PNG image and add icons with halved zoom.
    if show_icon:
        img = mpimg.imread("/Users/yilinwu/Desktop/HandHoldBlock.png")
        add_image(ax3d,
                  data["coord_x_full"][data["start_seg"]],
                  data["coord_y_full"][data["start_seg"]],
                  data["coord_z_full"][data["start_seg"]] + 4/2,
                  img, zoom=0.08/2)
        add_image(ax3d,
                  data["coord_x_full"][data["end_seg"]],
                  data["coord_y_full"][data["end_seg"]],
                  data["coord_z_full"][data["end_seg"]] + 4/2,
                  img, zoom=0.08/2)

    # Mark start and end points with scatter markers using halved sizes.
    ax3d.scatter(
        data["coord_x_full"][data["start_seg"]],
        data["coord_y_full"][data["start_seg"]],
        data["coord_z_full"][data["start_seg"]],
        color='black', s=25, marker='o', label='Start'
    )
    ax3d.scatter(
        data["coord_x_full"][data["end_seg"]],
        data["coord_y_full"][data["end_seg"]],
        data["coord_z_full"][data["end_seg"]],
        color='red', s=25, marker='o', label='End'
    )

    offset = 10/2  # halved offset value: 5
    ax3d.text(
        data["coord_x_full"][data["start_seg"]] + offset,
        data["coord_y_full"][data["start_seg"]],
        data["coord_z_full"][data["start_seg"]],
        'Start', color='black',
        fontdict={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax3d.text(
        data["coord_x_full"][data["end_seg"]] + offset,
        data["coord_y_full"][data["end_seg"]],
        data["coord_z_full"][data["end_seg"]],
        'End', color='black',
        fontdict={'fontsize': 11, 'fontweight': 'bold'}
    )

    ax3d.set_xlabel("X (mm)", fontsize=11)
    ax3d.set_ylabel("Y (mm)", fontsize=11)
    ax3d.set_zlabel("Z (mm)", fontsize=11)
    ax3d.legend(loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize=11, frameon=False)
    ax3d.grid(True)

    # Set ticks for the x axis as -max, 0, max.
    xmin, xmax = ax3d.get_xlim()
    max_abs_x = math.ceil(max(abs(xmin), abs(xmax)))
    ax3d.set_xticks([-max_abs_x, 0, max_abs_x])

    # Set ticks for the y axis as -max, 0, max.
    ymin, ymax = ax3d.get_ylim()
    max_abs_y = math.ceil(max(abs(ymin), abs(ymax)))
    ax3d.set_yticks([-max_abs_y, 0, max_abs_y])

    # For the z axis, use the actual min, middle, and max tick values rounded to integers.
    zmin, zmax = ax3d.get_zlim()
    zmin_int = int(math.floor(zmin))
    zmax_int = int(math.ceil(zmax))
    zmiddle_int = int(round((zmin_int + zmax_int) / 2))
    ax3d.set_zticks([zmin_int, zmiddle_int, zmax_int])

    plt.tight_layout()
    plt.show()


plot_3d_trajectory_icon_all_phase_data(results, reach_speed_segments, test_windows_7,
                                       subject="07/22/HW", hand="left",
                                       file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT66.csv',
                                       seg_index=15, fs=200, target_samples=101, dis_phases=0.3,
                                       show_icon=False)



















# -------------------------------------------------------------------------------------------------------------------
def compute_two_time_windows(all_phase_data):
    """
    Compute two time windows using phase indices from all_phase_data.

    Returns:
        window1 (dict): Nested dictionary with lists of (start, end) tuples.
        window2 (dict): Nested dictionary with lists of (start, end) tuples.
    """
    window1 = {
        subject: {
            hand: {
                file_path: [
                    (phase_data["latency_end_idx"], phase_data["ballistic_end"])
                    for seg, phase_data in sorted(trial_data.items())
                ]
                for file_path, trial_data in all_phase_data[subject][hand].items()
            }
            for hand in all_phase_data[subject]
        }
        for subject in all_phase_data
    }

    window2 = {
        subject: {
            hand: {
                file_path: [
                    (phase_data["ballistic_end"], phase_data["verification_start_idx"])
                    for seg, phase_data in sorted(trial_data.items())
                ]
                for file_path, trial_data in all_phase_data[subject][hand].items()
            }
            for hand in all_phase_data[subject]
        }
        for subject in all_phase_data
    }

    return window1, window2

ballistic_phase, corrective_phase = compute_two_time_windows(all_phase_data)

# -------------------------------------------------------------------------------------------------------------------

reach_TW_metrics_ballistic_phase = utils2.calculate_reach_metrics_for_time_windows_Normalizing(ballistic_phase, results)
reach_TW_metrics_corrective_phase = utils2.calculate_reach_metrics_for_time_windows_Normalizing(corrective_phase, results)

reach_sparc_ballistic_phase = utils2.calculate_reach_sparc_Normalizing(ballistic_phase, results)
reach_sparc_corrective_phase = utils2.calculate_reach_sparc_Normalizing(corrective_phase, results)

# -------------------------------------------------------------------------------------------------------------------
# Compute the minimal and maximal LDLJ values across segments (1 to 16) and print 
# the corresponding trial name and segment index

min_ldlj = float('inf')
max_ldlj = float('-inf')
min_trial = None
max_trial = None
min_segment = None
max_segment = None

# Create a list to hold all entries for further computations
ldlj_entries = []

for trial_name, trial_values in reach_TW_metrics_ballistic_phase['reach_LDLJ']["07/22/HW"]["left"].items():
    for seg in range(1, 17):  # iteration over segments 1 to 16
        value = trial_values[seg - 1]
        ldlj_entries.append((value, trial_name, seg))
        if value < min_ldlj:
            min_ldlj = value
            min_trial = trial_name
            min_segment = seg
        if value > max_ldlj:
            max_ldlj = value
            max_trial = trial_name
            max_segment = seg

print("Minimum LDLJ value across trials:", min_ldlj)
print("Found in trial:", min_trial, "at segment index:", min_segment)
print("Maximum LDLJ value across trials:", max_ldlj)
print("Found in trial:", max_trial, "at segment index:", max_segment)

# Compute the median LDLJ value across segments from all trials

all_values = [entry[0] for entry in ldlj_entries]
median_ldlj = np.median(all_values)

closest_diff = float('inf')
median_trial = None
median_segment = None

for value, trial_name, seg in ldlj_entries:
    diff = abs(value - median_ldlj)
    if diff < closest_diff:
        closest_diff = diff
        median_trial = trial_name
        median_segment = seg

print("Median LDLJ value across trials:", median_ldlj)
print("Closest value found in trial:", median_trial, "at segment index:", median_segment)

# Find the bottom 5 (smallest) and top 5 (largest) LDLJ values
# Sorting in ascending order: bottom 5 are the smallest values,
# top 5 are the largest values.

sorted_entries = sorted(ldlj_entries, key=lambda x: x[0])
bottom_5 = sorted_entries[:5]
top_5 = sorted_entries[-5:]  # highest 5 values

print("\nBottom 5 LDLJ values:")
for value, trial_name, seg in bottom_5:
    print(f"LDLJ: {value}, Trial: {trial_name}, Segment: {seg}")

print("\nTop 5 LDLJ values:")
for value, trial_name, seg in top_5:
    print(f"LDLJ: {value}, Trial: {trial_name}, Segment: {seg}")







def plot_3d_trajectory_icon_all_phase_data(results, reach_speed_segments, test_windows,
                                           subject, hand, file_paths, seg_indices,
                                           fs=200, target_samples=101, dis_phases=0.3,
                                           show_icon=True):
    """
    Computes phase indices directly from results and plots 3D trajectories for each selected combination
    of file_path and seg_index. This version calls calculate_phase_indices internally for each combination.
    Optionally shows icons if show_icon is True.
    
    Uses halved figure settings and applies common x, y, z axis ranges across subplots based on the maximum 
    range among all segments.

    The title for each subplot is obtained from:
           reach_TW_metrics_ballistic_phase['reach_LDLJ'][subject][hand][file_path][seg_index]
           
    Args:
        results: The results dictionary.
        reach_speed_segments: Reach segments (used as the first set for segmentation).
        test_windows: The second segmentation parameter.
        subject, hand: Identifiers for selecting the correct data.
        file_paths: List of file paths (one per subplot).
        seg_indices: List of segment indices corresponding to each file path.
        fs: Sampling frequency.
        target_samples: Number of samples for interpolation.
        dis_phases: Distance threshold for phase computation.
        show_icon: Boolean flag for displaying icons.
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    num_plots = len(seg_indices)
    # Halved individual figure size (width, height)
    single_fig_size = (10/2, 8/2)  # (5, 4)
    # Increase width for multiple subplots
    overall_size = (single_fig_size[0] * num_plots, single_fig_size[1])
    
    fig3d, axes = plt.subplots(1, num_plots, subplot_kw={'projection': '3d'}, figsize=overall_size)
    if num_plots == 1:
        axes = [axes]

    # Lists to store axis limits for all subplots.
    x_ranges = []
    y_ranges = []
    z_mins = []
    z_maxs = []

    # Loop over each file path and segment index pair.
    for ax3d, file_path, seg_index in zip(axes, file_paths, seg_indices):
        # Compute data for the given file and segment.
        data = utils9.calculate_phase_indices(results, reach_speed_segments, test_windows,
                                               subject=subject, hand=hand, file_path=file_path,
                                               fs=fs, target_samples=target_samples,
                                               seg_index=seg_index, dis_phases=dis_phases)

        # Plot the four phases with halved linewidths.
        ax3d.plot(
            data["coord_x_full"][data["start_seg"]:data["latency_end_idx"]],
            data["coord_y_full"][data["start_seg"]:data["latency_end_idx"]],
            data["coord_z_full"][data["start_seg"]:data["latency_end_idx"]],
            color='magenta', linewidth=2, label='Latency Phase'
        )
        ax3d.plot(
            data["coord_x_full"][data["latency_end_idx"]:data["ballistic_end"]],
            data["coord_y_full"][data["latency_end_idx"]:data["ballistic_end"]],
            data["coord_z_full"][data["latency_end_idx"]:data["ballistic_end"]],
            color='cyan', linewidth=2, label='Ballistic Phase'
        )
        ax3d.plot(
            data["coord_x_full"][data["ballistic_end"]:data["verification_start_idx"]],
            data["coord_y_full"][data["ballistic_end"]:data["verification_start_idx"]],
            data["coord_z_full"][data["ballistic_end"]:data["verification_start_idx"]],
            color='green', linewidth=2, label='Correction Phase'
        )
        ax3d.plot(
            data["coord_x_full"][data["verification_start_idx"]:data["end_seg"]],
            data["coord_y_full"][data["verification_start_idx"]:data["end_seg"]],
            data["coord_z_full"][data["verification_start_idx"]:data["end_seg"]],
            color='orange', linewidth=2, label='Verification Phase'
        )

        # Helper function to add an image at a 3D point.
        def add_image(ax, xs, ys, zs, img, zoom=0.1):
            x2, y2, _ = proj_transform(xs, ys, zs, ax.get_proj())
            imagebox = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(imagebox, (x2, y2), frameon=False, xycoords='data')
            ax.add_artist(ab)

        # Add icons if enabled.
        if show_icon:
            img = mpimg.imread("/Users/yilinwu/Desktop/HandHoldBlock.png")
            add_image(ax3d,
                      data["coord_x_full"][data["start_seg"]],
                      data["coord_y_full"][data["start_seg"]],
                      data["coord_z_full"][data["start_seg"]] + 4/2,
                      img, zoom=0.08/2)
            add_image(ax3d,
                      data["coord_x_full"][data["end_seg"]],
                      data["coord_y_full"][data["end_seg"]],
                      data["coord_z_full"][data["end_seg"]] + 4/2,
                      img, zoom=0.08/2)

        # Mark the start and end points.
        ax3d.scatter(
            data["coord_x_full"][data["start_seg"]],
            data["coord_y_full"][data["start_seg"]],
            data["coord_z_full"][data["start_seg"]],
            color='black', s=25, marker='o', label='Start'
        )
        ax3d.scatter(
            data["coord_x_full"][data["end_seg"]],
            data["coord_y_full"][data["end_seg"]],
            data["coord_z_full"][data["end_seg"]],
            color='red', s=25, marker='o', label='End'
        )

        offset = 10/2  # halved offset value.
        ax3d.text(
            data["coord_x_full"][data["start_seg"]] + offset,
            data["coord_y_full"][data["start_seg"]],
            data["coord_z_full"][data["start_seg"]],
            'Start', color='black',
            fontdict={'fontsize': 11, 'fontweight': 'bold'}
        )
        ax3d.text(
            data["coord_x_full"][data["end_seg"]] + offset,
            data["coord_y_full"][data["end_seg"]],
            data["coord_z_full"][data["end_seg"]],
            'End', color='black',
            fontdict={'fontsize': 11, 'fontweight': 'bold'}
        )

        ax3d.set_xlabel("X (mm)", fontsize=11)
        ax3d.set_ylabel("Y (mm)", fontsize=11)
        ax3d.set_zlabel("Z (mm)", fontsize=11)
        # Use the reach_TW_metrics_ballistic_phase lookup as the title.
        title_val = reach_TW_metrics_ballistic_phase['reach_LDLJ'][subject][hand][file_path][seg_index]
        
        ax3d.text(
            np.max(data["coord_x_full"])/2,
            np.max(data["coord_y_full"])/2,
            np.max(data["coord_z_full"]),
            f"ballistic LDLJ:\n {title_val:.2f}",
            fontsize=12, verticalalignment='top', horizontalalignment='right', color='black'
        )
        
        ax3d.grid(True)

        # Gather local axis limits.
        xmin, xmax = ax3d.get_xlim()
        x_ranges.append(max(abs(xmin), abs(xmax)))
        ymin, ymax = ax3d.get_ylim()
        y_ranges.append(max(abs(ymin), abs(ymax)))
        zmin, zmax = ax3d.get_zlim()
        z_mins.append(zmin)
        z_maxs.append(zmax)

    # Determine global axis ranges across all subplots.
    global_range_x = math.ceil(max(x_ranges))
    global_range_y = math.ceil(max(y_ranges))
    global_zmin = math.floor(min(z_mins))
    global_zmax = math.ceil(max(z_maxs))
    global_zmiddle = int(round((global_zmin + global_zmax) / 2))

    # Apply the same axis limits to all subplots.
    for ax in axes:
        ax.set_xlim([-global_range_x, global_range_x])
        ax.set_xticks([-global_range_x, 0, global_range_x])
        ax.set_ylim([-global_range_y, global_range_y])
        ax.set_yticks([-global_range_y, 0, global_range_y])
        ax.set_zlim(global_zmin, global_zmax)
        ax.set_zticks([global_zmin, global_zmiddle, global_zmax])
        
        # Only show z-axis tick labels on the right-most subplot.
        if ax != axes[-1]:
            ax.set_zticklabels([])
    
    # Create a global legend outside the subplots.
    handles, labels = axes[0].get_legend_handles_labels()
    fig3d.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=6, frameon=False)
    plt.subplots_adjust(wspace=-0.35)  # further reduces whitespace
    plt.tight_layout()
    plt.show()


plot_3d_trajectory_icon_all_phase_data(results, reach_speed_segments, test_windows_7,
                                       subject="07/22/HW", hand="left",
                                       file_paths=[
                                           '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT30.csv',
                                           '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT48.csv',
                                           '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT48.csv'
                                       ],
                                       seg_indices=[14, 13, 7],
                                       fs=200, target_samples=101, dis_phases=0.3,
                                       show_icon=False)


def plot_trajectory_all_phase_data(results, reach_speed_segments, test_windows_7, all_phase_data,
                             subject, hand, file_path, seg_index):
    """
    Plots 2D subplots overlaying Position, Velocity, Acceleration,
    Original Jerk and Trajectory (X, Y, Z) for two segmentations.
    For the first segmentation, indices come from reach_speed_segments;
    for the second segmentation, indices come from test_windows_7.
    The phase markers (Latency End, Ballistic End, Verification Start) are taken from all_phase_data.
    Also finds local minima of velocity in the velocity subplot.
    """

    # Select marker based on hand.
    marker = 'RFIN' if hand == 'right' else 'LFIN'
    
    # Extract trajectory data from results.
    traj_data = results[subject][hand][1][file_path]['traj_space'][marker]
    position_full = traj_data[0]
    velocity_full = traj_data[1]
    acceleration_full = traj_data[2]
    jerk_full = traj_data[3]
    
    # Extract coordinate arrays.
    traj_data_full = results[subject][hand][1][file_path]['traj_data']
    coord_prefix = "RFIN_" if hand == 'right' else "LFIN_"
    coord_x_full = np.array(traj_data_full[coord_prefix + "X"])
    coord_y_full = np.array(traj_data_full[coord_prefix + "Y"])
    coord_z_full = np.array(traj_data_full[coord_prefix + "Z"])
    
    # Get segmentation indices for segment 1 from reach_speed_segments.
    seg_range1 = reach_speed_segments[subject][hand][file_path][seg_index]
    start_idx1, end_idx1 = seg_range1
    
    # Get segmentation indices for segment 2 from test_windows_7.
    seg_range2 = test_windows_7[subject][hand][file_path][seg_index]
    start_idx2, end_idx2 = seg_range2

    # Extract trajectory segments.
    pos_seg1 = position_full[start_idx1:end_idx1]
    vel_seg1 = velocity_full[start_idx1:end_idx1]
    acc_seg1 = acceleration_full[start_idx1:end_idx1]
    jerk_seg1 = np.array(jerk_full[start_idx1:end_idx1])
    
    pos_seg2 = position_full[start_idx2:end_idx2]
    vel_seg2 = velocity_full[start_idx2:end_idx2]
    acc_seg2 = acceleration_full[start_idx2:end_idx2]
    jerk_seg2 = np.array(jerk_full[start_idx2:end_idx2])
    
    # Create frame index lists.
    x_vals1 = list(range(start_idx1, end_idx1))
    x_vals2 = list(range(start_idx2, end_idx2))
    
    # Get phase markers from all_phase_data.
    phase = all_phase_data[subject][hand][file_path][seg_index]
    latency_end_idx = phase["latency_end_idx"]
    ballistic_end   = phase["ballistic_end"]
    verification_start_idx = phase["verification_start_idx"]
    
    # # Find local minima in velocity for both segments.
    # peaks1, _ = find_peaks(-np.array(vel_seg1))
    # peaks2, _ = find_peaks(-np.array(vel_seg2))
    
    peaks1, _ = find_peaks(-np.array(vel_seg1), prominence=100)
    peaks2, _ = find_peaks(-np.array(vel_seg2), prominence=100)

    print(len(peaks1), len(peaks2))

    
    # Plotting the subplots.
    fig, axs = plt.subplots(4, 1, figsize=(6, 10))
    
    tittle_fontsize = 10
    legend_fontsize = 10
    label_fontsize = 10

    # Overlay Position.
    axs[0].plot(x_vals1, pos_seg1, color='blue', linewidth=2, label='Segment 1')
    axs[0].plot(x_vals2, pos_seg2, color='lime', linestyle='--', linewidth=2, label='Segment 2')
    axs[0].set_title('Position', fontsize=tittle_fontsize)
    
    # Overlay Velocity.
    axs[1].plot(x_vals1, vel_seg1, color='blue', linewidth=2, label='Segment 1')
    axs[1].plot(x_vals2, vel_seg2, color='lime', linestyle='--', linewidth=2, label='Segment 2')
    # Mark local minima on segment 1.
    axs[1].scatter(np.array(x_vals1)[peaks1], np.array(vel_seg1)[peaks1],
                   color='red', marker='o', s=50, label='Local min (Seg 1)')
    # Mark local minima on segment 2.
    axs[1].scatter(np.array(x_vals2)[peaks2], np.array(vel_seg2)[peaks2],
                   color='magenta', marker='x', s=50, label='Local min (Seg 2)')
    axs[1].set_title('Velocity', fontsize=tittle_fontsize)
    
    # Overlay Acceleration.
    axs[2].plot(x_vals1, acc_seg1, color='blue', linewidth=2, label='Segment 1')
    axs[2].plot(x_vals2, acc_seg2, color='lime', linestyle='--', linewidth=2, label='Segment 2')
    axs[2].set_title('Acceleration', fontsize=tittle_fontsize)
    
    # Overlay Original Jerk.
    axs[3].plot(x_vals1, jerk_seg1, color='blue', linewidth=2, label='Segment 1')
    axs[3].plot(x_vals2, jerk_seg2, color='lime', linestyle='--', linewidth=2, label='Segment 2')
    axs[3].set_title('Original Jerk', fontsize=tittle_fontsize)

    # Overlay phase markers on each subplot.
    for ax in axs:
        ax.axvline(latency_end_idx, color='magenta', linestyle=':', label='Latency End')
        ax.axvline(ballistic_end, color='cyan', linestyle=':', label='Ballistic End')
        ax.axvline(verification_start_idx, color='orange', linestyle=':', label='Verification Start')
        # Only add legend for the bottom plot.
        if ax == axs[3]:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), fontsize=legend_fontsize, ncol=len(ax.get_legend_handles_labels()[0]))
    plt.tight_layout()
    plt.show()

plot_trajectory_all_phase_data(results, reach_speed_segments, test_windows_7, all_phase_data,
                             subject="07/22/HW", hand="left",
                             file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT30.csv',
                             seg_index=14)

# # -------------------------------------------------------------------------------------------------------------------
def plot_trials_p_v_a_j_Nj_by_location(results, reach_speed_segments,
                                           subject="07/22/HW", hand="left",
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
        traj_data = results[subject][hand][1][file_path]['traj_space'][marker]
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
        file_paths = list(results[subject][hand][1].keys())
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
                traj_data = results[subject][hand][1][fp]['traj_space'][marker]
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
plot_trials_p_v_a_j_Nj_by_location(results, ballistic_phase,
                                       subject="07/22/HW", hand="left", 
                                       fs=200, target_samples=101,
                                       metrics=["vel"],
                                       mode="single",
                                       file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv')

# Example call for option 2 (all trials):
plot_trials_p_v_a_j_Nj_by_location(results, ballistic_phase,
                                       subject="07/22/HW", hand="left", 
                                       fs=200, target_samples=101,
                                       metrics=["vel"],
                                       mode="all")

# Example call for option 2 (all trials):
plot_trials_p_v_a_j_Nj_by_location(results, corrective_phase,
                                       subject="07/22/HW", hand="left", 
                                       fs=200, target_samples=101,
                                       metrics=["vel"],
                                       mode="all")
# # -------------------------------------------------------------------------------------------------------------------
utils5.process_and_save_combined_metrics_acorss_phases(
    Block_Distance, reach_metrics,
    reach_TW_metrics_ballistic_phase, reach_TW_metrics_corrective_phase,
    reach_sparc_ballistic_phase, reach_sparc_corrective_phase,
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

def plot_metric_boxplots(updated_metrics_acorss_TWs, metrics=["TW3_LDLJ", "TW3_sparc", "durations", "distance"], use_median=False):
    import matplotlib.pyplot as plt

    # Set up the subplot grid.
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    axs = axs.flatten()

    # Loop through each metric to compute stats and plot the boxplots.
    for i, metric in enumerate(metrics):
        # Calculate global statistics for the metric across all subjects and hands.
        global_vals = []
        for subject, subject_data in updated_metrics_acorss_TWs.items():
            for hand, hand_data in subject_data.items():
                if metric in hand_data:
                    for trial, trial_vals in hand_data[metric].items():
                        global_vals.extend(trial_vals)
        global_vals = np.array(global_vals)
        if global_vals.size > 0:
            min_val = np.nanmin(global_vals)
            max_val = np.nanmax(global_vals)
            median_val = np.nanmedian(global_vals)
            q1 = np.nanpercentile(global_vals, 25)
            q3 = np.nanpercentile(global_vals, 75)
            iqr = q3 - q1
            # Print in thesis style with one sentence.
            print(f"For metric {metric}, the minimum value was {min_val:.2f}, the maximum value was {max_val:.2f}, the median was {median_val:.2f}, and the interquartile range was {iqr:.2f}.")
        
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

        # Add labels "Good" and "Bad" on the y-axis.
        ax.text(-0.15, 0, "Good", transform=ax.transAxes, color="green", fontsize=12, va="center")
        ax.text(-0.15, 1, "Bad", transform=ax.transAxes, color="red", fontsize=12, va="center")
        
        # Update title and axis labels depending on whether we're using median or mean.
        label = "Median" if use_median else "Mean"
        ax.set_title(f"Box Plot of {metric.lower()} {label} Values by Hand\n{test_title}")
        ax.set_xlabel("Hand")
        ax.set_ylabel(f"{label} {metric.lower()}")

        # Draw a dashed line connecting the two hands for each subject.
        for subject in pivot_df.index:
            nd_value = pivot_df.loc[subject, "non_dominant"]
            d_value = pivot_df.loc[subject, "dominant"]
            ax.plot([0, 1], [nd_value, d_value], color="gray", linestyle="--", linewidth=1, alpha=0.7)

        print(f"Plotted boxplot for metric: {metric}")
        print(f"Wilcoxon test result for {metric}: {test_title}")
    plt.tight_layout()
    plt.show()

plot_metric_boxplots(updated_metrics_acorss_phases, metrics=["ballistic_LDLJ", "ballistic_sparc", "corrective_LDLJ", "corrective_sparc","durations", "distance"], use_median=True)

# # -------------------------------------------------------------------------------------------------------------------
# Do reach types that are faster on average also tend to be less accurate on average?
result_Check_SAT_in_trials_mean_median_of_reach_indices = utils6.Check_SAT_in_trials_mean_median_of_reach_indices(updated_metrics_acorss_phases, '07/22/HW', 'durations', 'distance', stat_type="median")

# Within one reach location, is there still a speed–accuracy trade-off across repetitions?
utils6.scatter_plot_duration_distance_by_choice(updated_metrics_acorss_phases, overlay_hands=False, selected_subjects=['07/22/HW'], special_indices=[0], show_hyperbolic_fit=False, color_mode="uniform", show_median_overlay=False)
_, corr_results, result_Check_SAT_in_reach_indices_by_hand_by_subject, heatmap_medians = utils6.Check_SAT_in_reach_indices_by_index_or_subject(updated_metrics_acorss_phases, '07/22/HW', grouping="hand_by_subject", hyperbolic=False)
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

for var in ["ballistic_LDLJ", "ballistic_sparc", "corrective_LDLJ", "corrective_sparc"]:
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
