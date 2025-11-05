from scipy.stats import wilcoxon
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import numpy as np
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

# 3 BBT test
Traj_folder = "/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025"
Box_Traj_folder = "/Users/yilinwu/Desktop/Yilin-Honours/Box/Traj/2025"
DataProcess_folder = "/Users/yilinwu/Desktop/honours data/DataProcess"
Figure_folder = "/Users/yilinwu/Desktop/honours/Thesis/figure"
prominence_threshold_speed = 2.9
prominence_threshold_position = 2.9
import utils1 # Importing utils1 for data Pre-processing
import utils2 # Importing utils2 for reach metrics calculation and time window Specific calculation
import utils3 # Importing utils3 for plotting functions
import utils4 # Importing utils4 for image files
import utils5 # Importing utils5 for combining metrics
import utils6 # Importing utils6 for Data Analysis and Visualization
import utils7 # Importing utils7 for Motor Experiences
import utils8 # Importing utils8 for sBBTResult
import utils9 # Importing utils9 for thesis

import iBBT_utils # Importing iBBT_utils for iBBT specific functions


# --- GET ALL DATES ---
All_dates = sorted(utils1.get_subfolders_with_depth(Traj_folder, depth=3))
# --- PROCESS ALL DATE AND SAVE ALL MOVEMENT DATA AS pickle file ---
iBBT_utils.process_all_dates_separate(All_dates, Traj_folder, Box_Traj_folder, Figure_folder, DataProcess_folder, 
                      prominence_threshold_speed, prominence_threshold_position)

prominence_threshold_speed = 0
prominence_threshold_position = 0
indices_to_process = [14, 22, 25, 27]
for index in indices_to_process:
    iBBT_utils.process_all_dates_separate(All_dates[index:index+1], Traj_folder, Box_Traj_folder, Figure_folder, DataProcess_folder, 
                                          prominence_threshold_speed, prominence_threshold_position)

# --- LOAD ALL SUBJECTS' tBBT ERROR FROM IMAGE, SAVE AS pickle file---
iBBT_results = iBBT_utils.load_selected_subject_results(All_dates, DataProcess_folder)
iBBT_reach_speed_segments = utils2.get_reach_speed_segments(iBBT_results)

iBBT_reach_metrics = utils2.calculate_reach_metrics(iBBT_reach_speed_segments, iBBT_results, fs=200)
iBBT_test_windows_7 = utils9.compute_test_window_7(iBBT_results, iBBT_reach_speed_segments, iBBT_reach_metrics)


def calculate_total_time_per_trial(iBBT_reach_speed_segments):
    """
    Calculate the total time taken for each subject per trial for iBBT.
    Time is calculated as the difference between the start of the first reach and the end of the last reach.

    Args:
        iBBT_reach_speed_segments (dict): Dictionary containing reach speed segments for each subject and trial.

    Returns:
        dict: A dictionary with total time taken for each subject per trial.
    """
    total_time_per_trial = {}

    for subject, hands in iBBT_reach_speed_segments.items():
        total_time_per_trial[subject] = {}
        for hand, trials in hands.items():
            total_time_per_trial[subject][hand] = {}
            for trial, segments in trials.items():
                if segments:  # Ensure there are segments
                    start_time = segments[0][0]  # Start time of the first reach
                    end_time = segments[-1][1]  # End time of the last reach
                    total_time = end_time - start_time
                    total_time_per_trial[subject][hand][trial] = total_time/200  # Convert to seconds assuming fs=200 Hz

    return total_time_per_trial
iBBT_total_time_results = calculate_total_time_per_trial(iBBT_reach_speed_segments)

# Swap left/right total time results for specific subjects and rename keys as 'non_dominant' and 'dominant'
def swap_and_rename_total_time_results(total_time_results, all_dates):
    """
    Swap left/right total time results for specific subjects and rename keys as 'non_dominant' and 'dominant'.

    Args:
        total_time_results (dict): A dictionary with total time taken for each subject per trial.
        all_dates (list): List of all subject dates.

    Returns:
        dict: Modified total time results with swapped and renamed keys.
    """
    # Subjects for which left/right metrics should be swapped
    subjects_to_swap = {all_dates[20], all_dates[22]}

    # Create swapped and renamed copy
    modified_results = {}
    for subj, hands in total_time_results.items():
        if subj in subjects_to_swap:
            swapped_hands = {
                'non_dominant': hands.get('right', {}),
                'dominant': hands.get('left', {})
            }
        else:
            swapped_hands = {
                'non_dominant': hands.get('left', {}),
                'dominant': hands.get('right', {})
            }
        modified_results[subj] = swapped_hands

    return modified_results
iBBT_total_time_results = swap_and_rename_total_time_results(iBBT_total_time_results, All_dates)

import matplotlib.pyplot as plt 

def plot_total_time_box_chart_with_overlay_and_stats(total_time_per_trial):
    """
    Plot a box chart of average total time taken for each participant for each hand,
    with data points overlaid and connected between the same participants.
    Perform a Wilcoxon signed-rank test between the hands and annotate significance.
    Also, return the mean total time for each hand for each participant.

    Args:
        total_time_per_trial (dict): A dictionary with total time taken for each subject per trial.

    Returns:
        dict: A dictionary containing the mean total time for non-dominant and dominant hands for each participant.
    """
    non_dominant_times = []
    dominant_times = []
    participant_ids = []
    mean_times_per_participant = {}

    for subject, hands in total_time_per_trial.items():
        non_dominant_hand_times = []
        dominant_hand_times = []
        for hand, trials in hands.items():
            for trial, total_time in trials.items():
                if hand == 'non_dominant':
                    non_dominant_hand_times.append(total_time)
                elif hand == 'dominant':
                    dominant_hand_times.append(total_time)
        if non_dominant_hand_times or dominant_hand_times:
            participant_ids.append(subject)
        mean_non_dominant = sum(non_dominant_hand_times) / len(non_dominant_hand_times) if non_dominant_hand_times else None
        mean_dominant = sum(dominant_hand_times) / len(dominant_hand_times) if dominant_hand_times else None
        non_dominant_times.append(mean_non_dominant)
        dominant_times.append(mean_dominant)
        mean_times_per_participant[subject] = {
            "non_dominant": mean_non_dominant,
            "dominant": mean_dominant
        }

    # Remove None values for statistical test
    paired_non_dominant = []
    paired_dominant = []
    for nd, d in zip(non_dominant_times, dominant_times):
        if nd is not None and d is not None:
            paired_non_dominant.append(nd)
            paired_dominant.append(d)

    # Perform Wilcoxon signed-rank test
    if paired_non_dominant and paired_dominant:
        stat, p_value = wilcoxon(paired_non_dominant, paired_dominant)
        print(f"Wilcoxon signed-rank test: statistic={stat}, p-value={p_value}")
        print("Significant difference between hands." if p_value < 0.05 else "No significant difference between hands.")
    else:
        print("Not enough paired data for Wilcoxon test.")
        p_value = None

    # Calculate mean total time for each hand
    mean_non_dominant = sum(paired_non_dominant) / len(paired_non_dominant) if paired_non_dominant else None
    mean_dominant = sum(paired_dominant) / len(paired_dominant) if paired_dominant else None
    print(f"Mean total time - Non-Dominant Hand: {mean_non_dominant:.2f}s, Dominant Hand: {mean_dominant:.2f}s")

    data = [non_dominant_times, dominant_times]
    plt.boxplot(data, labels=['Non-Dominant Hand', 'Dominant Hand'])
    plt.ylabel('Average Total Time (s)')
    plt.title('Average Total Time Taken per Participant by Hand')

    # Overlay data points and connect them
    for i, (non_dominant, dominant) in enumerate(zip(non_dominant_times, dominant_times)):
        if non_dominant is not None and dominant is not None:
            plt.plot([1, 2], [non_dominant, dominant], 'k-', alpha=0.5)  # Connect points with a line
        if non_dominant is not None:
            plt.scatter(1, non_dominant, color='blue', alpha=0.7)  # Non-dominant hand data point
        if dominant is not None:
            plt.scatter(2, dominant, color='orange', alpha=0.7)  # Dominant hand data point

    # Add significance annotation
    if p_value is not None:
        sig_levels = [(0.001, "***"), (0.01, "**"), (0.05, "*"), (1.0, "ns")]
        significance = next((label for threshold, label in sig_levels if p_value <= threshold), "ns")
        y_max = max(max(non_dominant_times), max(dominant_times)) * 1.1
        plt.plot([1, 2], [y_max, y_max], 'k-', lw=1.5)  # Line for significance
        plt.text(1.5, y_max, significance, ha='center', va='bottom', fontsize=12)

    plt.show()

    return mean_times_per_participant

iBBT_mean_total_times = plot_total_time_box_chart_with_overlay_and_stats(iBBT_total_time_results)


# Load sBBTResult from CSV into a DataFrame and compute right and left hand scores
sBBTResult = utils8.load_and_compute_sbbt_result()
# Swap and rename sBBTResult scores for specific subjects
sBBTResult = utils8.swap_and_rename_sbbt_result(sBBTResult)


# print all non-dominant and dominant mean total time results
for subject, times in iBBT_mean_total_times.items():
    print(f"Subject: {subject}, Non-Dominant Mean Total Time: {times['non_dominant']}, Dominant Mean Total Time: {times['dominant']}")

# Print sBBTResult non-dominant and dominant scores
for index, row in sBBTResult.iterrows():
    print(f"Subject: {row['Subject']}, Non-Dominant Score: {row['non_dominant']}, Dominant Score: {row['dominant']}")

def plot_correlation_between_ibbt_and_sbbt(iBBT_mean_total_times, sBBTResult):
    """
    Plot correlation between iBBT mean total times and sBBT scores for non-dominant and dominant hands.
    Compute and display Spearman correlation coefficients and p-values.

    Args:
        iBBT_mean_total_times (dict): Dictionary containing mean total times for non-dominant and dominant hands.
        sBBTResult (DataFrame): DataFrame containing sBBT scores for non-dominant and dominant hands.
    """
    import matplotlib.pyplot as plt

    # Extract non-dominant and dominant data for correlation
    iBBT_non_dominant = [times['non_dominant'] for subject, times in iBBT_mean_total_times.items()]
    iBBT_dominant = [times['dominant'] for subject, times in iBBT_mean_total_times.items()]
    sBBT_non_dominant = sBBTResult['non_dominant'].tolist()
    sBBT_dominant = sBBTResult['dominant'].tolist()

    # Compute Spearman correlation for non-dominant hand
    corr_non_dominant, p_value_non_dominant = spearmanr(iBBT_non_dominant, sBBT_non_dominant)
    print(f"Non-Dominant Hand Spearman Correlation: {corr_non_dominant:.2f}, p-value: {p_value_non_dominant:.4f}")

    # Compute Spearman correlation for dominant hand
    corr_dominant, p_value_dominant = spearmanr(iBBT_dominant, sBBT_dominant)
    print(f"Dominant Hand Spearman Correlation: {corr_dominant:.2f}, p-value: {p_value_dominant:.4f}")

    # Plot scatter plot and linear regression for non-dominant hand
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(iBBT_non_dominant, sBBT_non_dominant, color='blue', alpha=0.7, label='Data Points')
    # Linear regression for visual purposes
    slope, intercept = np.polyfit(iBBT_non_dominant, sBBT_non_dominant, 1)
    regression_line = np.polyval([slope, intercept], iBBT_non_dominant)
    plt.plot(iBBT_non_dominant, regression_line, color='red', label='Linear Fit')
    plt.title(f"Non-Dominant Hand\nSpearman Correlation: {corr_non_dominant:.2f} (p={p_value_non_dominant:.4f})")
    plt.xlabel("iBBT Non-Dominant Mean Total Time (s)")
    plt.ylabel("sBBT Non-Dominant Score (num blocks transfer in 60s)")
    plt.legend()

    # Plot scatter plot and linear regression for dominant hand
    plt.subplot(1, 2, 2)
    plt.scatter(iBBT_dominant, sBBT_dominant, color='orange', alpha=0.7, label='Data Points')
    # Linear regression for visual purposes
    slope, intercept = np.polyfit(iBBT_dominant, sBBT_dominant, 1)
    regression_line = np.polyval([slope, intercept], iBBT_dominant)
    plt.plot(iBBT_dominant, regression_line, color='red', label='Linear Fit')
    plt.title(f"Dominant Hand\nSpearman Correlation: {corr_dominant:.2f} (p={p_value_dominant:.4f})")
    plt.xlabel("iBBT Dominant Mean Total Time (s)")
    plt.ylabel("sBBT Dominant Score (num blocks transfer in 60s)")
    plt.legend()

    plt.tight_layout()
    plt.show()
plot_correlation_between_ibbt_and_sbbt(iBBT_mean_total_times, sBBTResult)

# --- Compute medians for all subjects/dates and both hands ---
iBBT_medians = {}  # to store results

for date_subject, hands_data in iBBT_reach_metrics['reach_durations'].items():
    iBBT_medians[date_subject] = {}
    
    for hand, trials_data in hands_data.items():
        # trials_data: dict of 4 trials, each with 16 movement durations
        values = np.array(list(trials_data.values()))  # shape (n_trials, 16)
        
        # Calculate medians
        median_per_movement = np.median(values, axis=0)
        overall_median = np.median(median_per_movement)
        
        # Store results
        iBBT_medians[date_subject][hand] = {
            'median_time_per_location': median_per_movement.tolist(),
            'overall_locations': float(overall_median)
        }

def plot_correlation_between_ibbt_and_sbbt(iBBT_medians, sBBTResult):
    """
    Plot correlation between iBBT overall median times and sBBT scores for non-dominant and dominant hands.
    Compute and display Spearman correlation coefficients and p-values.

    Args:
        iBBT_medians (dict): Dictionary containing overall median times for non-dominant and dominant hands.
        sBBTResult (DataFrame): DataFrame containing sBBT scores for non-dominant and dominant hands.
    """
    import matplotlib.pyplot as plt

    # Extract non-dominant and dominant data for correlation
    iBBT_non_dominant = [hands['left']['overall_locations'] for subject, hands in iBBT_medians.items() if 'left' in hands]
    iBBT_dominant = [hands['right']['overall_locations'] for subject, hands in iBBT_medians.items() if 'right' in hands]
    sBBT_non_dominant = sBBTResult['non_dominant'].tolist()
    sBBT_dominant = sBBTResult['dominant'].tolist()

    # Compute Spearman correlation for non-dominant hand
    corr_non_dominant, p_value_non_dominant = spearmanr(iBBT_non_dominant, sBBT_non_dominant)
    print(f"Non-Dominant Hand Spearman Correlation: {corr_non_dominant:.2f}, p-value: {p_value_non_dominant:.4f}")

    # Compute Spearman correlation for dominant hand
    corr_dominant, p_value_dominant = spearmanr(iBBT_dominant, sBBT_dominant)
    print(f"Dominant Hand Spearman Correlation: {corr_dominant:.2f}, p-value: {p_value_dominant:.4f}")

    # Plot scatter plot and linear regression for non-dominant hand
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(iBBT_non_dominant, sBBT_non_dominant, color='blue', alpha=0.7, label='Data Points')
    # Linear regression for visual purposes
    slope, intercept = np.polyfit(iBBT_non_dominant, sBBT_non_dominant, 1)
    regression_line = np.polyval([slope, intercept], iBBT_non_dominant)
    plt.plot(iBBT_non_dominant, regression_line, color='red', label='Linear Fit')
    plt.title(f"Non-Dominant Hand\nSpearman Correlation: {corr_non_dominant:.2f} (p={p_value_non_dominant:.4f})")
    plt.xlabel("iBBT Non-Dominant Median time for single placement (s)")
    plt.ylabel("sBBT Non-Dominant Score (num blocks transfer in 60s)")
    plt.legend()

    # Plot scatter plot and linear regression for dominant hand
    plt.subplot(1, 2, 2)
    plt.scatter(iBBT_dominant, sBBT_dominant, color='orange', alpha=0.7, label='Data Points')
    # Linear regression for visual purposes
    slope, intercept = np.polyfit(iBBT_dominant, sBBT_dominant, 1)
    regression_line = np.polyval([slope, intercept], iBBT_dominant)
    plt.plot(iBBT_dominant, regression_line, color='red', label='Linear Fit')
    plt.title(f"Dominant Hand\nSpearman Correlation: {corr_dominant:.2f} (p={p_value_dominant:.4f})")
    plt.xlabel("iBBT Dominant Median time for single placement (s)")
    plt.ylabel("sBBT Dominant Score (num blocks transfer in 60s)")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_correlation_between_ibbt_and_sbbt(iBBT_medians, sBBTResult)

def plot_correlation_between_ibbt_and_sbbt(iBBT_medians, iBBT_mean_total_times):
    """
    Plot correlation between iBBT overall median times and iBBT mean total times for non-dominant and dominant hands.
    Compute and display Spearman correlation coefficients and p-values.

    Args:
        iBBT_medians (dict): Dictionary containing overall median times for non-dominant and dominant hands.
        iBBT_mean_total_times (dict): Dictionary containing mean total times for non-dominant and dominant hands.
    """
    import matplotlib.pyplot as plt

    # Extract non-dominant and dominant data for correlation
    iBBT_medians_non_dominant = [hands['left']['overall_locations'] for subject, hands in iBBT_medians.items() if 'left' in hands]
    iBBT_medians_dominant = [hands['right']['overall_locations'] for subject, hands in iBBT_medians.items() if 'right' in hands]
    iBBT_mean_non_dominant = [times['non_dominant'] for subject, times in iBBT_mean_total_times.items()]
    iBBT_mean_dominant = [times['dominant'] for subject, times in iBBT_mean_total_times.items()]

    # Compute Spearman correlation for non-dominant hand
    corr_non_dominant, p_value_non_dominant = spearmanr(iBBT_medians_non_dominant, iBBT_mean_non_dominant)
    print(f"Non-Dominant Hand Spearman Correlation: {corr_non_dominant:.2f}, p-value: {p_value_non_dominant:.4f}")

    # Compute Spearman correlation for dominant hand
    corr_dominant, p_value_dominant = spearmanr(iBBT_medians_dominant, iBBT_mean_dominant)
    print(f"Dominant Hand Spearman Correlation: {corr_dominant:.2f}, p-value: {p_value_dominant:.4f}")

    # Plot scatter plot and linear regression for non-dominant hand
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(iBBT_medians_non_dominant, iBBT_mean_non_dominant, color='blue', alpha=0.7, label='Data Points')
    # Linear regression for visual purposes
    slope, intercept = np.polyfit(iBBT_medians_non_dominant, iBBT_mean_non_dominant, 1)
    regression_line = np.polyval([slope, intercept], iBBT_medians_non_dominant)
    plt.plot(iBBT_medians_non_dominant, regression_line, color='red', label='Linear Fit')
    plt.title(f"Non-Dominant Hand\nSpearman Correlation: {corr_non_dominant:.2f} (p={p_value_non_dominant:.4f})")
    plt.xlabel("iBBT Non-Dominant Median time for single placement (s)")
    plt.ylabel("iBBT Non-Dominant Mean Total Time (s)")
    plt.legend()

    # Plot scatter plot and linear regression for dominant hand
    plt.subplot(1, 2, 2)
    plt.scatter(iBBT_medians_dominant, iBBT_mean_dominant, color='orange', alpha=0.7, label='Data Points')
    # Linear regression for visual purposes
    slope, intercept = np.polyfit(iBBT_medians_dominant, iBBT_mean_dominant, 1)
    regression_line = np.polyval([slope, intercept], iBBT_medians_dominant)
    plt.plot(iBBT_medians_dominant, regression_line, color='red', label='Linear Fit')
    plt.title(f"Dominant Hand\nSpearman Correlation: {corr_dominant:.2f} (p={p_value_dominant:.4f})")
    plt.xlabel("iBBT Dominant Median time for single placement (s)")
    plt.ylabel("iBBT Dominant Mean Total Time (s)")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_correlation_between_ibbt_and_sbbt(iBBT_medians, iBBT_mean_total_times)




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
iBBT_phase_indices = compute_phase_indices(All_dates, iBBT_results, iBBT_test_windows_7)

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
iBBT_nan_count, iBBT_nan_details = check_nan_phase_indices(iBBT_phase_indices)

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
ballistic_phase, correction_phase = compute_two_time_windows(iBBT_test_windows_7, iBBT_phase_indices)
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
plot_phase_duration_averages(ballistic_phase, correction_phase, iBBT_test_windows_7, iBBT_reach_speed_segments)

# -------------------------------------------------------------------------------------------------------------------
iBBT_reach_TW_metrics_ballistic_phase = utils2.calculate_reach_metrics_for_time_windows_Normalizing(ballistic_phase, iBBT_results)
iBBT_reach_TW_metrics_correction_phase = utils2.calculate_reach_metrics_for_time_windows_Normalizing(correction_phase, iBBT_results)
iBBT_reach_TW_metrics_TW = utils2.calculate_reach_metrics_for_time_windows_Normalizing(iBBT_test_windows_7, iBBT_results)

iBBT_reach_sparc_ballistic_phase = utils2.calculate_reach_sparc_Normalizing(ballistic_phase, iBBT_results)
iBBT_reach_sparc_correction_phase = utils2.calculate_reach_sparc_Normalizing(correction_phase, iBBT_results)
iBBT_reach_sparc_TW = utils2.calculate_reach_sparc_Normalizing(iBBT_test_windows_7, iBBT_results)

# -------------------------------------------------------------------------------------------------------------------
