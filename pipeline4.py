import numpy as np
from scipy.stats import zscore
from scipy.stats import ttest_rel
from scipy.stats import spearmanr
import seaborn as sns
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import utils1 # Importing utils1 for data Pre-processing
import utils2 # Importing utils2 for reach metrics calculation and time window Specific calculation
import utils3 # Importing utils3 for plotting functions
import utils4 # Importing utils4 for image files
import utils5 # Importing utils5 for combining metrics

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

# --- SELECT DATES TO PROCESS ---
# All_dates = All_dates[5:len(All_dates)] 

# -------------------------------------------------------------------------------------------------------------------

# PART 0: Data Pre-processing [!!! THINGS THAT NEED TO BE DONE ONCE !!!]

# --- PROCESS ALL DATE AND SAVE ALL MOVEMENT DATA AS pickle file ---
utils1.process_all_dates_separate(All_dates, Traj_folder, Box_Traj_folder, Figure_folder, DataProcess_folder, 
                      prominence_threshold_speed, prominence_threshold_position)

# --- RENAME IMAGE FILES ---
# run this only once to rename the files in the tBBT_Image_folder
# for date in All_dates[5:len(All_dates)]:
#     directory = f"{tBBT_Image_folder}{date}"
#     print(f"Renaming files in directory: {directory}")
#     utils4.rename_files(directory)

# --- FIND BEST CALIBRATION IMAGES COMBINATION FOR EACH SUBJECT ---
# subjects = [date for date in All_dates]
# utils4.run_test_for_each_subject(subjects, tBBT_Image_folder)

# --- PROCESS ALL SUBJECTS' IMAGES RETURN tBBT ERROR FROM IMAGE, SAVE AS pickle file---
utils4.process_all_subjects_images(All_dates, tBBT_Image_folder, DataProcess_folder)

# -------------------------------------------------------------------------------------------------------------------

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

# --- CALCULATE REACH METRICS ---
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
reach_TW_metrics = utils2.calculate_reach_metrics_for_time_windows(test_windows_3, results)

# --- Save ALL LDLJ VALUES BY SUBJECT, HAND, AND TRIAL ---
utils2.save_ldlj_values(reach_TW_metrics, DataProcess_folder)

# --- CALCULATE SPARC FOR EACH TEST WINDOW FOR ALL DATES, HANDS, AND TRIALS ---
reach_sparc_test_windows_1 = utils2.calculate_reach_sparc(test_windows_1, results)
reach_sparc_test_windows_2 = utils2.calculate_reach_sparc(test_windows_2, results)
reach_sparc_test_windows_3 = utils2.calculate_reach_sparc(test_windows_3, results)

# --- Save ALL SPARC VALUES BY SUBJECT, HAND, AND TRIAL ---
utils2.save_sparc_values(reach_sparc_test_windows_1, DataProcess_folder)

# -------------------------------------------------------------------------------------------------------------------

# PART 3: Combine Metrics and Save Results
# --- PROCESS AND SAVE COMBINED METRICS [DURATIONS, SPARC, LDLJ, AND DISTANCE, CALCULATED SPEED AND ACCURACY FOR ALL DATES]---
utils5.process_and_save_combined_metrics(Block_Distance, reach_metrics, reach_sparc_test_windows_1, reach_TW_metrics, All_dates, DataProcess_folder)

# --- LOAD ALL COMBINED METRICS PER SUBJECT FROM PICKLE FILE ---
all_combined_metrics = utils5.load_combined_metrics_per_subject(DataProcess_folder)

# --- LOCATE NaN INDICES (UNDETECTED BLOCK) FOR ALL SUBJECTS ---
nan_indices_all = utils5.find_nan_indices_all_subjects(all_combined_metrics)

# -------------------------------------------------------------------------------------------------------------------

















# Scatter plot for durations vs distance for all trials, overlaying them with different colors
def plot_durations_vs_distance(all_combined_metrics, subject, hand):
    trials = all_combined_metrics[subject][hand]['durations'].keys()
    colors = sns.color_palette("Blues", len(trials))  # Generate a color palette from light to dark

    plt.figure(figsize=(10, 8))
    for i, trial_path in enumerate(trials):
        durations = all_combined_metrics[subject][hand]['durations'][trial_path]
        distances = all_combined_metrics[subject][hand]['distance'][trial_path]

        # Ensure the lengths match
        if len(durations) != len(distances):
            print(f"Mismatch in the number of durations and distances data points for trial {trial_path}.")
            continue

        # Scatter plot for each trial
        plt.scatter(durations, distances, color=colors[i], edgecolor='k', alpha=0.7, label=f'Trial {i+1}')

    # Calculate x and y limits based on data
    valid_durations = [d for trial_path in trials for d in all_combined_metrics[subject][hand]['durations'][trial_path] if not np.isnan(d) and not np.isinf(d)]
    valid_distances = [d for trial_path in trials for d in all_combined_metrics[subject][hand]['distance'][trial_path] if not np.isnan(d) and not np.isinf(d)]

    if valid_durations and valid_distances:
        x_min, x_max = np.percentile(valid_durations, [0.5, 99.5])
        y_min, y_max = np.percentile(valid_distances, [0.5, 99.5])
    else:
        print("Error: No valid data points for axis limits.")
        return

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.title(f"Durations vs Distance for {subject} ({hand.capitalize()} Hand)", fontsize=14)
    plt.xlabel("Durations (s)", fontsize=12)
    plt.ylabel("Distance (mm)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Trials", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()

# Example usage
plot_durations_vs_distance(all_combined_metrics, '07/22/HW', 'right')

# Scatter plot for durations vs distance for all trials, each hand as a subplot
def plot_durations_vs_distance(all_combined_metrics, subject):
    hands = ['left', 'right']
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    for ax, hand in zip(axes, hands):
        trials = all_combined_metrics[subject][hand]['durations'].keys()
        colors = sns.color_palette("Blues", len(trials))  # Generate a color palette from light to dark

        for i, trial_path in enumerate(trials):
            durations = all_combined_metrics[subject][hand]['durations'][trial_path]
            distances = all_combined_metrics[subject][hand]['distance'][trial_path]

            # Ensure the lengths match
            if len(durations) != len(distances):
                print(f"Mismatch in the number of durations and distances data points for trial {trial_path}.")
                continue

            # Scatter plot for each trial
            ax.scatter(durations, distances, color=colors[i], edgecolor='k', alpha=0.7, label=f'Trial {i+1}')

        # Calculate x and y limits based on data
        valid_durations = [d for trial_path in trials for d in all_combined_metrics[subject][hand]['durations'][trial_path] if not np.isnan(d) and not np.isinf(d)]
        valid_distances = [d for trial_path in trials for d in all_combined_metrics[subject][hand]['distance'][trial_path] if not np.isnan(d) and not np.isinf(d)]

        if valid_durations and valid_distances:
            x_min, x_max = np.percentile(valid_durations, [0.5, 99.5])
            y_min, y_max = np.percentile(valid_distances, [0.5, 99.5])
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        else:
            print(f"Error: No valid data points for axis limits for {hand} hand.")
            continue

        ax.set_title(f"Durations vs Distance for {subject} ({hand.capitalize()} Hand)", fontsize=14)
        ax.set_xlabel("Durations (s)", fontsize=12)
        ax.set_ylabel("Distance (mm)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(title="Trials", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.show()

# Example usage for both hands
plot_durations_vs_distance(all_combined_metrics, '07/22/HW')


# Scatter plot for speed vs accuracy for all trials, each hand as a subplot
def plot_speed_vs_accuracy(all_combined_metrics, subject):
    hands = ['left', 'right']
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    for ax, hand in zip(axes, hands):
        trials = all_combined_metrics[subject][hand]['speed'].keys()
        colors = sns.color_palette("Blues", len(trials))  # Generate a color palette from light to dark

        for i, trial_path in enumerate(trials):
            speeds = list(all_combined_metrics[subject][hand]['speed'][trial_path])
            accuracies = list(all_combined_metrics[subject][hand]['accuracy'][trial_path])
            if any(np.isnan(accuracies)):
                motor_acuity = np.nan
                continue

            # Ensure the lengths match
            if len(speeds) != len(accuracies):
                print(f"Mismatch in the number of speeds and accuracies data points for trial {trial_path}.")
                continue

            # Scatter plot for each trial
            ax.scatter(speeds, accuracies, color=colors[i], edgecolor='k', alpha=0.7, label=f'Trial {i+1}')

        # Calculate x and y limits based on data
        valid_speeds = [s for trial_path in trials for s in all_combined_metrics[subject][hand]['speed'][trial_path] if not np.isnan(s) and not np.isinf(s)]
        valid_accuracies = [a for trial_path in trials for a in all_combined_metrics[subject][hand]['accuracy'][trial_path] if not np.isnan(a) and not np.isinf(a)]

        if valid_speeds and valid_accuracies:
            x_min, x_max = np.percentile(valid_speeds, [0.5, 99.5])
            y_min, y_max = np.percentile(valid_accuracies, [0.5, 99.5])
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        else:
            print(f"Error: No valid data points for axis limits for {hand} hand.")
            continue

        ax.set_title(f"Speed vs Accuracy for {subject} ({hand.capitalize()} Hand)", fontsize=14)
        ax.set_xlabel("Speed (1/Duration)", fontsize=12)
        ax.set_ylabel("Accuracy (1/Distance)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(title="Trials", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.show()

# Example usage for both hands
plot_speed_vs_accuracy(all_combined_metrics, '07/22/HW')


# Scatter plot for speed vs accuracy for a single reach, each hand as a subplot
def plot_speed_vs_accuracy_single_reach(all_combined_metrics, subject, reach_index):
    hands = ['left', 'right']
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    for ax, hand in zip(axes, hands):
        trials = all_combined_metrics[subject][hand]['speed'].keys()
        colors = sns.color_palette("Blues", len(trials))  # Generate a color palette from light to dark

        for i, trial_path in enumerate(trials):
            speeds = list(all_combined_metrics[subject][hand]['speed'][trial_path])
            accuracies = list(all_combined_metrics[subject][hand]['accuracy'][trial_path])

            # Ensure the reach index is valid
            if reach_index >= len(speeds) or reach_index >= len(accuracies):
                print(f"Invalid reach index {reach_index} for trial {trial_path}.")
                continue

            # Scatter plot for the specified reach
            ax.scatter(speeds[reach_index], accuracies[reach_index], color=colors[i], edgecolor='k', alpha=0.7, label=f'Trial {i+1}')

        # Calculate x and y limits based on data
        valid_speeds = [s[reach_index] for trial_path in trials for s in [all_combined_metrics[subject][hand]['speed'][trial_path]] if not np.isnan(s[reach_index]) and not np.isinf(s[reach_index])]
        valid_accuracies = [a[reach_index] for trial_path in trials for a in [all_combined_metrics[subject][hand]['accuracy'][trial_path]] if not np.isnan(a[reach_index]) and not np.isinf(a[reach_index])]

        if valid_speeds and valid_accuracies:
            x_min, x_max = np.percentile(valid_speeds, [0.5, 99.5])
            y_min, y_max = np.percentile(valid_accuracies, [0.5, 99.5])
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        else:
            print(f"Error: No valid data points for axis limits for {hand} hand.")
            continue

        ax.set_title(f"Speed vs Accuracy for {subject} ({hand.capitalize()} Hand, Reach {reach_index + 1})", fontsize=14)
        ax.set_xlabel("Speed (1/Duration)", fontsize=12)
        ax.set_ylabel("Accuracy (1/Distance)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(title="Trials", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.show()
# Example usage for a single reach (e.g., reach index 0)
plot_speed_vs_accuracy_single_reach(all_combined_metrics, '07/22/HW', 12)



# Scatter plot for speed vs accuracy for all reaches, each hand as a separate figure, 4x4 layout for each reach
def plot_speed_vs_accuracy_all_reaches(all_combined_metrics, subject):
    hands = ['left', 'right']
    num_reaches = max(len(all_combined_metrics[subject][hand]['speed'][list(all_combined_metrics[subject][hand]['speed'].keys())[0]]) for hand in hands)
    grid_size = 4  # 4x4 layout

    for hand in hands:
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16), sharey=True)
        axes = axes.flatten()

        for reach_index in range(num_reaches):
            ax = axes[reach_index]
            trials = all_combined_metrics[subject][hand]['speed'].keys()
            colors = sns.color_palette("Blues", len(trials))  # Generate a color palette from light to dark

            for i, trial_path in enumerate(trials):
                speeds = list(all_combined_metrics[subject][hand]['speed'][trial_path])
                accuracies = list(all_combined_metrics[subject][hand]['accuracy'][trial_path])

                # Ensure the reach index is valid
                if reach_index >= len(speeds) or reach_index >= len(accuracies):
                    print(f"Invalid reach index {reach_index} for trial {trial_path}.")
                    continue

                # Scatter plot for the specified reach
                ax.scatter(speeds[reach_index], accuracies[reach_index], color=colors[i], edgecolor='k', alpha=0.7, label=f'Trial {i+1}')

            # Calculate x and y limits based on data
            valid_speeds = [s[reach_index] for trial_path in trials for s in [all_combined_metrics[subject][hand]['speed'][trial_path]] if not np.isnan(s[reach_index]) and not np.isinf(s[reach_index])]
            valid_accuracies = [a[reach_index] for trial_path in trials for a in [all_combined_metrics[subject][hand]['accuracy'][trial_path]] if not np.isnan(a[reach_index]) and not np.isinf(a[reach_index])]

            if valid_speeds and valid_accuracies:
                # x_min, x_max = np.percentile(valid_speeds, [0.5, 99.5])
                # y_min, y_max = np.percentile(valid_accuracies, [0.5, 99.5])
                x_min, x_max = min(valid_speeds), max(valid_speeds)
                y_min, y_max = min(valid_accuracies), max(valid_accuracies)
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
            # else:
            #     print(f"Error: No valid data points for axis limits for {hand} hand.")
            #     continue

            ax.set_title(f"Reach {reach_index + 1} ({len(valid_accuracies)} points)", fontsize=10)
            ax.set_xlabel("Speed (1/Duration)\n(Slow → Fast)", fontsize=8)
            ax.set_ylabel("Accuracy (1/Distance)\n(Bad → Good)", fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.6)

        # Hide unused subplots
        for unused_ax in axes[num_reaches:]:
            unused_ax.axis('off')

        fig.suptitle(f"Speed vs Accuracy for {hand.capitalize()} Hand ({subject})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

# Example usage for all reaches
plot_speed_vs_accuracy_all_reaches(all_combined_metrics, '07/22/HW')


# Calculate motor acuity for all reaches, each hand
def calculate_motor_acuity_for_all(all_combined_metrics):
    for subject in all_combined_metrics:
        hands = ['left', 'right']

        for hand in hands:
            trials = all_combined_metrics[subject][hand]['speed'].keys()

            for trial_path in trials:
                speeds = list(all_combined_metrics[subject][hand]['speed'][trial_path])
                accuracies = list(all_combined_metrics[subject][hand]['accuracy'][trial_path])

                # Calculate motor acuity for all reaches
                motor_acuity_list = []
                for reach_index in range(len(speeds)):
                    if np.isnan(accuracies[reach_index]):
                        motor_acuity = np.nan
                    else:
                        motor_acuity = np.sqrt(speeds[reach_index]**2 + accuracies[reach_index]**2)
                    motor_acuity_list.append(motor_acuity)

                if 'motor_acuity' not in all_combined_metrics[subject][hand]:
                    all_combined_metrics[subject][hand]['motor_acuity'] = {}
                all_combined_metrics[subject][hand]['motor_acuity'][trial_path] = motor_acuity_list

    return all_combined_metrics

# Example usage for calculating motor acuity for all reaches
all_combined_metrics = calculate_motor_acuity_for_all(all_combined_metrics)

HW = all_combined_metrics['07/22/HW']['right']['motor_acuity']

# Scatter plot for motor_acuity vs sparc for all reaches, each hand as a separate figure, 4x4 layout for each reach
def plot_motor_acuity_vs_sparc_all_reaches(all_combined_metrics, reach_sparc_test_windows_1, subject):
    hands = ['left', 'right']
    num_reaches = max(len(all_combined_metrics[subject][hand]['motor_acuity'][list(all_combined_metrics[subject][hand]['motor_acuity'].keys())[0]]) for hand in hands)
    grid_size = 4  # 4x4 layout

    for hand in hands:
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16), sharey=True)
        axes = axes.flatten()

        for reach_index in range(num_reaches):
            ax = axes[reach_index]
            trials = all_combined_metrics[subject][hand]['motor_acuity'].keys()
            colors = sns.color_palette("Blues", len(trials))  # Generate a color palette from light to dark

            for i, trial_path in enumerate(trials):
                motor_acuity = list(all_combined_metrics[subject][hand]['motor_acuity'][trial_path])
                sparc_values = list(reach_sparc_test_windows_1[subject][hand][trial_path])

                # Ensure the reach index is valid
                if reach_index >= len(motor_acuity) or reach_index >= len(sparc_values):
                    print(f"Invalid reach index {reach_index} for trial {trial_path}.")
                    continue

                # Scatter plot for the specified reach
                ax.scatter(motor_acuity[reach_index], sparc_values[reach_index], color=colors[i], edgecolor='k', alpha=0.7, label=f'Trial {i+1}')

            # # Calculate x and y limits based on data
            # valid_motor_acuity = [m[reach_index] for trial_path in trials for m in [all_combined_metrics[subject][hand]['motor_acuity'][trial_path]] if not np.isnan(m[reach_index]) and not np.isinf(m[reach_index])]
            # valid_sparc = [s[reach_index] for trial_path in trials for s in [reach_sparc_test_windows_1[subject][hand][trial_path]] if not np.isnan(s[reach_index]) and not np.isinf(s[reach_index])]

            # if valid_motor_acuity and valid_sparc:
            #     x_min, x_max = min(valid_motor_acuity), max(valid_motor_acuity)
            #     y_min, y_max = min(valid_sparc), max(valid_sparc)
            #     ax.set_xlim(x_min, x_max)
            #     ax.set_ylim(y_min, y_max)

            ax.set_title(f"Reach {reach_index + 1}", fontsize=10)
            ax.set_xlabel("Motor Acuity", fontsize=8)
            ax.set_ylabel("SPARC", fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.6)

        # Hide unused subplots
        for unused_ax in axes[num_reaches:]:
            unused_ax.axis('off')

        fig.suptitle(f"Motor Acuity vs SPARC for {hand.capitalize()} Hand ({subject})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

# Example usage for all reaches
plot_motor_acuity_vs_sparc_all_reaches(all_combined_metrics, reach_sparc_test_windows_1, '07/22/HW')















# Plot scatter plot of Speed vs Accuracy for a specific subject and hand with correlation
def plot_speed_vs_accuracy_scatter_with_correlation(reach_metrics, Block_Distance, subject, hand):

    # Access duration from reach_metrics['reach_durations']
    if 'reach_durations' in reach_metrics and subject in reach_metrics['reach_durations'] and hand in reach_metrics['reach_durations'][subject]:
        durations = list(reach_metrics['reach_durations'][subject][hand].values())
    else:
        print(f"No reach durations available for subject {subject} and hand {hand}.")
        return

    # Ensure subject and hand are strings
    subject = str(subject)
    hand = str(hand)
    accuracies = list(Block_Distance[subject][hand].values())

    # Ensure the lengths match
    if len(durations) != len(accuracies):
        print("Mismatch in the number of speed and accuracy data points.")
        return

    # Flatten durations and accuracies for correlation calculation
    flat_accuracies = [item for sublist in accuracies for item in sublist if not pd.isna(item)]
    flat_durations = [item for sublist, acc_sublist in zip(durations, accuracies) for item, acc in zip(sublist, acc_sublist) if not pd.isna(acc)]

    # Calculate correlation
    if len(flat_durations) == len(flat_accuracies):
        correlation, p_value = spearmanr(flat_durations, flat_accuracies)
        print(f"Spearman Correlation: {correlation:.2f}, p-value: {p_value:.3e}")
    else:
        print("Mismatch in flattened durations and accuracies for correlation calculation.")
        return

    # Calculate 0.5th and 99.5th percentiles for x and y axis limits
    x_min, x_max = np.percentile(flat_durations, [0.5, 99.5])
    y_min, y_max = np.percentile(flat_accuracies, [0.5, 99.5])

    # Create a unique color for each trial
    colors = sns.color_palette("husl", len(durations))

    # Create scatter plot
    plt.figure(figsize=(8, 8))
    for i in range(len(durations)):
        points_count = len([val for val in accuracies[i] if not pd.isna(val)])
        plt.scatter(durations[i], accuracies[i], color=colors[i], edgecolor='k', alpha=0.7, label=f'Trial {i+1} ({points_count} points)')
    plt.title(f"Durations vs Accuracy for {subject} ({hand.capitalize()} Hand)\nSpearman Correlation: {correlation:.2f}, p={p_value:.3e}", fontsize=14)
    plt.xlabel("Durations (s)", fontsize=12)
    plt.ylabel("Accuracy (distance (mm) away from centre)", fontsize=12)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Trials", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()























# Plot scatter plot of Speed vs Accuracy for a specific subject and hand with correlation
def plot_speed_vs_accuracy_scatter_with_correlation(reach_metrics, Block_Distance, subject, hand):

    # Access duration from reach_metrics['reach_durations']
    if 'reach_durations' in reach_metrics and subject in reach_metrics['reach_durations'] and hand in reach_metrics['reach_durations'][subject]:
        durations = list(reach_metrics['reach_durations'][subject][hand].values())
    else:
        print(f"No reach durations available for subject {subject} and hand {hand}.")
        return

    # Ensure subject and hand are strings
    subject = str(subject)
    hand = str(hand)
    accuracies = list(Block_Distance[subject][hand].values())

    # Ensure the lengths match
    if len(durations) != len(accuracies):
        print("Mismatch in the number of speed and accuracy data points.")
        return

    # Flatten durations and accuracies for correlation calculation
    flat_accuracies = [item for sublist in accuracies for item in sublist if not pd.isna(item)]
    flat_durations = [item for sublist, acc_sublist in zip(durations, accuracies) for item, acc in zip(sublist, acc_sublist) if not pd.isna(acc)]

    # Calculate correlation
    if len(flat_durations) == len(flat_accuracies):
        correlation, p_value = spearmanr(flat_durations, flat_accuracies)
        print(f"Spearman Correlation: {correlation:.2f}, p-value: {p_value:.3e}")
    else:
        print("Mismatch in flattened durations and accuracies for correlation calculation.")
        return

    # Calculate 0.5th and 99.5th percentiles for x and y axis limits
    x_min, x_max = np.percentile(flat_durations, [0.5, 99.5])
    y_min, y_max = np.percentile(flat_accuracies, [0.5, 99.5])

    # Create a unique color for each trial
    colors = sns.color_palette("husl", len(durations))

    # Create scatter plot
    plt.figure(figsize=(8, 8))
    for i in range(len(durations)):
        points_count = len([val for val in accuracies[i] if not pd.isna(val)])
        plt.scatter(durations[i], accuracies[i], color=colors[i], edgecolor='k', alpha=0.7, label=f'Trial {i+1} ({points_count} points)')
    plt.title(f"Durations vs Accuracy for {subject} ({hand.capitalize()} Hand)\nSpearman Correlation: {correlation:.2f}, p={p_value:.3e}", fontsize=14)
    plt.xlabel("Durations (s)", fontsize=12)
    plt.ylabel("Accuracy (distance (mm) away from centre)", fontsize=12)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Trials", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()

# Example usage
plot_speed_vs_accuracy_scatter_with_correlation(reach_metrics, Block_Distance, '07/22/HW', 'left')





# Plot scatter plot of Speed vs Accuracy for a specific subject and hand with correlation
def plot_speed_vs_accuracy_scatter_with_correlation(reach_metrics, Block_Distance, subject, hand):

    # Access duration from reach_metrics['reach_durations']
    if 'reach_durations' in reach_metrics and subject in reach_metrics['reach_durations'] and hand in reach_metrics['reach_durations'][subject]:
        durations = list(reach_metrics['reach_durations'][subject][hand].values())
    else:
        print(f"No reach durations available for subject {subject} and hand {hand}.")
        return

    # Ensure subject and hand are strings
    subject = str(subject)
    hand = str(hand)
    accuracies = list(Block_Distance[subject][hand].values())

    # Ensure the lengths match
    if len(durations) != len(accuracies):
        print("Mismatch in the number of speed and accuracy data points.")
        return

    # Flatten durations and accuracies for correlation calculation, excluding NaN values
    paired_data = [
        (1 / duration, 1 / accuracy)
        for sublist, acc_sublist in zip(durations, accuracies)
        for duration, accuracy in zip(sublist, acc_sublist)
        if not pd.isna(duration) and not pd.isna(accuracy)
    ]

    if not paired_data:
        print("No valid paired data available for correlation calculation.")
        return

    speeds, inverse_Distance = zip(*paired_data)

    # Calculate correlation
    correlation, p_value = spearmanr(speeds, inverse_Distance)
    print(f"Spearman Correlation: {correlation:.2f}, p-value: {p_value:.3e}")

    # Calculate 0.5th and 99.5th percentiles for x and y axis limits
    x_min, x_max = np.percentile(speeds, [0.5, 99.5])
    y_min, y_max = np.percentile(inverse_Distance, [0.5, 99.5])

    # Create scatter plot
    plt.figure(figsize=(8, 8))
    for i, (speed, accuracy) in enumerate(zip(speeds, inverse_Distance)):
        color = plt.cm.Blues(i / len(speeds))  # Gradually darken the color based on the index
        plt.scatter(speed, accuracy, color=color, edgecolor='k', alpha=0.7)
    plt.title(f"Speed vs Accuracy for {subject} ({hand.capitalize()} Hand)\nSpearman Correlation: {correlation:.2f}, p={p_value:.3e}", fontsize=14)
    plt.xlabel("Speed (1/Duration)", fontsize=12)
    plt.ylabel("Accuracy (1/Distance)", fontsize=12)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Example usage
for date in All_dates:
    for hand in ['left', 'right']:
        plot_speed_vs_accuracy_scatter_with_correlation(reach_metrics, Block_Distance, date, hand)





# Plot scatter plot of Speed vs Accuracy for both hands as subplots
def plot_speed_vs_accuracy_both_hands(reach_metrics, Block_Distance, subject):
    hands = ['left', 'right']
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    for ax, hand in zip(axes, hands):
        # Access duration from reach_metrics['reach_durations']
        if 'reach_durations' in reach_metrics and subject in reach_metrics['reach_durations'] and hand in reach_metrics['reach_durations'][subject]:
            durations = list(reach_metrics['reach_durations'][subject][hand].values())
        else:
            print(f"No reach durations available for subject {subject} and hand {hand}.")
            continue

        # Ensure subject and hand are strings
        subject = str(subject)
        hand = str(hand)
        accuracies = list(Block_Distance[subject][hand].values())

        # Ensure the lengths match
        if len(durations) != len(accuracies):
            print(f"Mismatch in the number of speed and accuracy data points for {hand} hand.")
            continue

        # Flatten durations and accuracies for correlation calculation, excluding NaN values
        paired_data = [
            (1 / duration, 1 / accuracy)
            for sublist, acc_sublist in zip(durations, accuracies)
            for duration, accuracy in zip(sublist, acc_sublist)
            if not pd.isna(duration) and not pd.isna(accuracy)
        ]

        if not paired_data:
            print(f"No valid paired data available for correlation calculation for {hand} hand.")
            continue

        speeds, inverse_Distance = zip(*paired_data)

        # Calculate correlation
        correlation, p_value = spearmanr(speeds, inverse_Distance)
        print(f"Spearman Correlation ({hand.capitalize()} Hand): {correlation:.2f}, p-value: {p_value:.3e}")

        # Calculate 0.5th and 99.5th percentiles for x and y axis limits
        x_min, x_max = np.percentile(speeds, [0.5, 99.5])
        y_min, y_max = np.percentile(inverse_Distance, [0.5, 99.5])

        # Create scatter plot
        for i, (speed, accuracy) in enumerate(zip(speeds, inverse_Distance)):
            color = plt.cm.Blues(i / len(speeds))  # Gradually darken the color based on the index
            ax.scatter(speed, accuracy, color=color, edgecolor='k', alpha=0.7)

        ax.set_title(f"Speed vs Accuracy ({hand.capitalize()} Hand)\nSpearman Correlation: {correlation:.2f}, p={p_value:.3e}", fontsize=14)
        ax.set_xlabel("Speed (1/Duration)", fontsize=12)
        ax.set_ylabel("Accuracy (1/Distance)", fontsize=12)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

# Example usage
for date in All_dates:
    print(f"Processing date: {date}")
    plot_speed_vs_accuracy_both_hands(reach_metrics, Block_Distance, date)






# Plot scatter plot of Speed vs Accuracy for all dates with accuracy data, each subject as a subplot
def plot_speed_vs_accuracy_all_subjects(reach_metrics, Block_Distance, All_dates):
    valid_dates = [date for date in All_dates if date in Block_Distance]
    num_subjects = len(valid_dates)
    fig, axes = plt.subplots(num_subjects, 2, figsize=(16, 4 * num_subjects), sharey=True)

    for idx, date in enumerate(valid_dates):
        for ax, hand in zip(axes[idx], ['left', 'right']):
            # Access duration from reach_metrics['reach_durations']
            if 'reach_durations' in reach_metrics and date in reach_metrics['reach_durations'] and hand in reach_metrics['reach_durations'][date]:
                durations = list(reach_metrics['reach_durations'][date][hand].values())
            else:
                print(f"No reach durations available for subject {date} and hand {hand}.")
                continue

            # Ensure subject and hand are strings
            date = str(date)
            hand = str(hand)
            accuracies = list(Block_Distance[date][hand].values())

            # Ensure the lengths match
            if len(durations) != len(accuracies):
                print(f"Mismatch in the number of speed and accuracy data points for {hand} hand.")
                continue

            # Flatten durations and accuracies for correlation calculation, excluding NaN values
            paired_data = [
                (1 / duration, 1 / accuracy)
                for sublist, acc_sublist in zip(durations, accuracies)
                for duration, accuracy in zip(sublist, acc_sublist)
                if not pd.isna(duration) and not pd.isna(accuracy)
            ]

            if not paired_data:
                print(f"No valid paired data available for correlation calculation for {hand} hand.")
                continue

            speeds, inverse_Distance = zip(*paired_data)

            # Calculate correlation
            correlation, p_value = spearmanr(speeds, inverse_Distance)
            print(f"Spearman Correlation ({hand.capitalize()} Hand, {date}): {correlation:.2f}, p-value: {p_value:.3e}")

            # Calculate 0.5th and 99.5th percentiles for x and y axis limits
            x_min, x_max = np.percentile(speeds, [0.5, 99.5])
            y_min, y_max = np.percentile(inverse_Distance, [0.5, 99.5])

            # Create scatter plot
            for i, (speed, accuracy) in enumerate(zip(speeds, inverse_Distance)):
                color = plt.cm.Blues(i / len(speeds))  # Gradually darken the color based on the index
                ax.scatter(speed, accuracy, color=color, edgecolor='k', alpha=0.7)

            ax.set_title(f"{date} ({hand.capitalize()} Hand)\nSpearman Correlation: {correlation:.2f}, p={p_value:.3e}", fontsize=12)
            ax.set_xlabel("Speed (1/Duration)", fontsize=10)
            ax.set_ylabel("Accuracy (1/Distance)", fontsize=10)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

# Example usage
plot_speed_vs_accuracy_all_subjects(reach_metrics, Block_Distance, [date for i, date in enumerate(All_dates) if i not in [2, 16]])



# Plot scatter plot of Speed vs Accuracy for all dates with accuracy data, each subject as a subplot
def plot_speed_vs_accuracy_all_subjects(reach_metrics, Block_Distance, All_dates):
    valid_dates = [date for date in Block_Distance if date in All_dates]
    num_subjects = len(valid_dates)
    fig, axes = plt.subplots(num_subjects, 2, figsize=(16, 4 * num_subjects))

    for idx, date in enumerate(valid_dates):
        for ax, hand in zip(axes[idx], ['left', 'right']):
            # Access duration from reach_metrics['reach_durations']
            if 'reach_durations' in reach_metrics and date in reach_metrics['reach_durations'] and hand in reach_metrics['reach_durations'][date]:
                durations = list(reach_metrics['reach_durations'][date][hand].values())
            else:
                print(f"No reach durations available for subject {date} and hand {hand}.")
                continue

            # Ensure subject and hand are strings
            date = str(date)
            hand = str(hand)
            accuracies = list(Block_Distance[date][hand].values())

            # Ensure the lengths match
            if len(durations) != len(accuracies):
                print(f"Mismatch in the number of speed and accuracy data points for {hand} hand.")
                continue

            # Flatten durations and accuracies for correlation calculation, excluding NaN values
            paired_data = [
                (1 / duration, 1 / accuracy)
                for sublist, acc_sublist in zip(durations, accuracies)
                for duration, accuracy in zip(sublist, acc_sublist)
                if not pd.isna(duration) and not pd.isna(accuracy)
            ]

            if not paired_data:
                print(f"No valid paired data available for correlation calculation for {hand} hand.")
                continue

            speeds, inverse_Distance = zip(*paired_data)

            # Calculate correlation
            correlation, p_value = np.corrcoef(speeds, inverse_Distance)[0, 1], spearmanr(speeds, inverse_Distance).pvalue
            print(f"Pearson Correlation ({hand.capitalize()} Hand, {date}): {correlation:.2f}, p-value: {p_value:.3e}")

            # Create scatter plot
            ax.scatter(speeds, inverse_Distance, color='blue', edgecolor='k', alpha=0.7)
            ax.set_title(f"{date} ({hand.capitalize()} Hand)\nPearson Correlation: {correlation:.2f}, p={p_value:.3e}", fontsize=12)
            ax.set_xlabel("Speed (1/Duration)", fontsize=10)
            ax.set_ylabel("Accuracy (1/Distance)", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

# Example usage
plot_speed_vs_accuracy_all_subjects(reach_metrics, Block_Distance, [date for i, date in enumerate(All_dates) if i not in [2, 16]])












# This function plots a heatmap for each subject and hand based on the Block_Distance data.
def plot_heatmap_by_subject(Block_Distance):
    for subject, hands_data in Block_Distance.items():
        for hand, distances_data in hands_data.items():
            num_images = len(distances_data)
            heatmap_data = np.full((num_images, 16), np.nan)  # Initialize with NaN for missing values

            for img_idx, (image_number, distances) in enumerate(sorted(distances_data.items())):
                heatmap_data[img_idx, :] = distances  # Fill the row with distances for each image

            # Calculate outlier thresholds
            lower_thresh = np.percentile(heatmap_data[~np.isnan(heatmap_data)], 1)
            upper_thresh = np.percentile(heatmap_data[~np.isnan(heatmap_data)], 99)

            # Mask outliers in the heatmap data
            heatmap_data_masked = np.clip(heatmap_data, lower_thresh, upper_thresh)

            # Plot the heatmap for the current subject and hand
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(heatmap_data_masked, aspect='auto', cmap='viridis', interpolation='nearest')

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Distance (Clipped to Outlier Thresholds)')

            # Set axis labels and ticks
            ax.set_xlabel("Block Membership (1 to 16)")
            ax.set_ylabel("Image Index")
            ax.set_xticks(np.arange(16))
            ax.set_xticklabels(np.arange(1, 17))
            ax.set_yticks(np.arange(num_images))
            ax.set_yticklabels([f"Image {img}" for img in sorted(distances_data.keys())])

            # Add title
            ax.set_title(f"Subject: {subject}, Hand: {hand}")

            plt.tight_layout()
            plt.show()

# Example usage
plot_heatmap_by_subject(Block_Distance)





























# Calculate total data points for reach durations and block accuracy
total_reach_durations = sum(len(reach_metrics['reach_durations'][subject][hand]) 
                            for subject in reach_metrics['reach_durations'] 
                            for hand in reach_metrics['reach_durations'][subject])

total_block_accuracy = sum(len(Block_Distance[subject][hand]) 
                           for subject in Block_Distance 
                           for hand in Block_Distance[subject])

print(f"Total data points in reach durations: {total_reach_durations}")
print(f"Total data points in block accuracy: {total_block_accuracy}")



























# PART 3: 

# SPARC 
# Plot a heatmap of SPARC values for a specific subject and hand, with outliers annotated.
utils3.plot_sparc_heatmap_with_outliers(reach_sparc_test_windows_1, '07/23/AK', 'right')

# Plot a heatmap of actual average SPARC values across trials for all subjects and a specific hand,
# with an additional subplot for the average across all subjects.
utils3.plot_average_sparc_value_heatmap_with_subject_average(reach_sparc_test_windows_1, 'right')

# Plot a heatmap of ranked SPARC values for a specific subject and hand.
utils3.plot_ranked_sparc_heatmap(reach_sparc_test_windows_1, '07/23/AK', 'right')

# Plot a heatmap of average ranked SPARC values across trials across all subjects for a specific hand.
utils3.plot_average_sparc_ranking_heatmap_across_all_dates(reach_sparc_test_windows_1, 'left')

# LDLJ
# Plot a heatmap of actual average LDLJ values across trials for all subjects and a specific hand,
# with an additional subplot for the average across all subjects.
utils3.plot_average_ldlj_value_heatmap_with_subject_average(reach_TW_metrics, 'left')

# Plot a heatmap of average ranked LDLJ values across trials across all subjects for a specific hand.
utils3.plot_average_ldlj_ranking_heatmap_across_all_dates(reach_TW_metrics, 'right')


# Plot a heatmap of overall average SPARC values for a specific hand, rearranged into a 4x4 grid.
utils3.plot_average_sparc_value_heatmap_with_subject_average_4x4(reach_sparc_test_windows_1, 'left')

# Plot a heatmap of overall average LDLJ values for a specific hand, rearranged into a 4x4 grid.
utils3.plot_average_ldlj_value_heatmap_with_subject_average_4x4(reach_TW_metrics, 'left')

# Plot a violin plot of SPARC values for all participants, each participant in one color, excluding outliers.
def plot_sparc_violin_all_participants_no_outliers(reach_sparc_test_windows, hand):
    # Prepare data for all participants
    all_data = []
    all_labels = []
    all_colors = []
    participants = list(reach_sparc_test_windows.keys())
    color_palette = sns.color_palette("husl", len(participants))

    for idx, participant in enumerate(participants):
        sparc_data = reach_sparc_test_windows[participant][hand]
        sparc_matrix = np.array([values for values in sparc_data.values()])
        trials, reaches = sparc_matrix.shape

        # Remove outliers using z-score
        sparc_matrix = sparc_matrix[(np.abs(zscore(sparc_matrix, axis=None)) < 3).all(axis=1)]

        # Append data and labels
        for reach_idx in range(reaches):
            reach_values = sparc_matrix[:, reach_idx]
            all_data.extend(reach_values)
            all_labels.extend([reach_idx + 1] * len(reach_values))
            all_colors.extend([color_palette[idx]] * len(reach_values))

    # Create DataFrame for plotting
    data = pd.DataFrame({
        'SPARC Values': all_data,
        'Reach': all_labels,
        'Participant': all_colors
    })

    # Plot violin plot
    plt.figure(figsize=(12, 8))
    sns.violinplot(
        x='Reach', 
        y='SPARC Values', 
        data=data, 
        inner=None, 
        scale='width', 
        palette='light:#d3d3d3'  # Light grey fill for violins
    )

    # Overlay value dots for each participant
    for idx, participant in enumerate(participants):
        sparc_data = reach_sparc_test_windows[participant][hand]
        sparc_matrix = np.array([values for values in sparc_data.values()])
        trials, reaches = sparc_matrix.shape

        # Remove outliers using z-score
        sparc_matrix = sparc_matrix[(np.abs(zscore(sparc_matrix, axis=None)) < 3).all(axis=1)]

        for reach_idx in range(reaches):
            reach_values = sparc_matrix[:, reach_idx]
            plt.scatter(
                [reach_idx] * len(reach_values),
                reach_values,
                c=[color_palette[idx]] * len(reach_values),
                alpha=0.6,
                label=participant if reach_idx == 0 else ""
            )

    plt.xlabel('Reach')
    plt.ylabel('SPARC Values')
    plt.title(f'Violin Plot of SPARC Values for {hand.capitalize()} Hand (All Participants, No Outliers)')
    plt.legend(title='Participants', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

plot_sparc_violin_all_participants_no_outliers(reach_sparc_test_windows_1, 'left')


# Plot a violin plot of LDLJ values for all participants, each participant in one color, excluding outliers.
def plot_ldlj_violin_all_participants_no_outliers(reach_TW_metrics, hand):
    # Prepare data for all participants
    all_data = []
    all_labels = []
    all_colors = []
    participants = list(reach_TW_metrics['reach_LDLJ'].keys())
    color_palette = sns.color_palette("husl", len(participants))

    for idx, participant in enumerate(participants):
        ldlj_data = reach_TW_metrics['reach_LDLJ'][participant][hand]
        ldlj_matrix = np.array([values for values in ldlj_data.values()])
        trials, reaches = ldlj_matrix.shape

        # Remove outliers using z-score
        ldlj_matrix = ldlj_matrix[(np.abs(zscore(ldlj_matrix, axis=None)) < 3).all(axis=1)]

        # Append data and labels
        for reach_idx in range(reaches):
            reach_values = ldlj_matrix[:, reach_idx]
            all_data.extend(reach_values)
            all_labels.extend([reach_idx + 1] * len(reach_values))
            all_colors.extend([color_palette[idx]] * len(reach_values))

    # Create DataFrame for plotting
    data = pd.DataFrame({
        'LDLJ Values': all_data,
        'Reach': all_labels,
        'Participant': all_colors
    })

    # Plot violin plot
    plt.figure(figsize=(12, 8))
    sns.violinplot(
        x='Reach', 
        y='LDLJ Values', 
        data=data, 
        inner=None, 
        scale='width', 
        palette='light:#d3d3d3'  # Light grey fill for violins
    )

    # Overlay value dots for each participant
    for idx, participant in enumerate(participants):
        ldlj_data = reach_TW_metrics['reach_LDLJ'][participant][hand]
        ldlj_matrix = np.array([values for values in ldlj_data.values()])
        trials, reaches = ldlj_matrix.shape

        # Remove outliers using z-score
        ldlj_matrix = ldlj_matrix[(np.abs(zscore(ldlj_matrix, axis=None)) < 3).all(axis=1)]

        for reach_idx in range(reaches):
            reach_values = ldlj_matrix[:, reach_idx]
            plt.scatter(
                [reach_idx] * len(reach_values),
                reach_values,
                c=[color_palette[idx]] * len(reach_values),
                alpha=0.6,
                label=participant if reach_idx == 0 else ""
            )

    plt.xlabel('Reach')
    plt.ylabel('LDLJ Values')
    plt.title(f'Violin Plot of LDLJ Values for {hand.capitalize()} Hand (All Participants, No Outliers)')
    plt.legend(title='Participants', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

plot_ldlj_violin_all_participants_no_outliers(reach_TW_metrics, 'right')

# Plot mean and standard deviation of LDLJ values for each subject and hand as separate box plots for each hand
def plot_subject_hand_ldlj_boxplot_with_data_points(reach_TW_metrics):
    hands = ['right', 'left']
    data = []

    for hand in hands:
        all_subjects = list(reach_TW_metrics['reach_LDLJ'].keys())
        for subject in all_subjects:
            if hand not in reach_TW_metrics['reach_LDLJ'][subject]:
                continue
            ldlj_matrix = np.array([values for values in reach_TW_metrics['reach_LDLJ'][subject][hand].values()])
            
            # Collect all data points, mean, and standard deviation for each subject and hand
            mean_value = ldlj_matrix.mean()
            sd_value = ldlj_matrix.std()
            for value in ldlj_matrix.flatten():
                data.append({'Hand': hand.capitalize(), 'LDLJ Value': value, 'Mean LDLJ': mean_value, 'SD': sd_value, 'Subject': subject})

    df = pd.DataFrame(data)

    # Create subplots for left and right hands
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    for ax, hand in zip(axes, hands):
        sns.boxplot(
            ax=ax, x='Subject', y='LDLJ Value', 
            data=df[df['Hand'] == hand.capitalize()], 
            palette='Set2', showfliers=False
        )
        sns.stripplot(
            ax=ax, x='Subject', y='LDLJ Value', 
            data=df[df['Hand'] == hand.capitalize()], 
            dodge=True, alpha=0.6, marker='o', size=8, palette='husl'
        )
        
        # Add error bars for standard deviation and annotate mean values
        for i, subject in enumerate(df[df['Hand'] == hand.capitalize()]['Subject'].unique()):
            mean = df[(df['Hand'] == hand.capitalize()) & (df['Subject'] == subject)]['Mean LDLJ'].values[0]
            sd = df[(df['Hand'] == hand.capitalize()) & (df['Subject'] == subject)]['SD'].values[0]
            ax.errorbar(
                x=i, y=mean, yerr=sd, fmt='none', c='black', capsize=5, label='SD' if i == 0 else ""
            )
            ax.text(
                x=i, y=mean, s=f'{mean:.2f}', color='black', ha='center', va='bottom', fontsize=10
            )

        ax.set_title(f'LDLJ Values for {hand.capitalize()} Hand (Including Data Points)', fontsize=14)
        ax.set_xlabel('Subject', fontsize=12)
        ax.set_ylabel('LDLJ Value', fontsize=12)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

plot_subject_hand_ldlj_boxplot_with_data_points(reach_TW_metrics)

# Plot mean and standard deviation of LDLJ values for each subject and hand as separate box plots for each hand
# Averaging across specific reach groups: (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15), (4, 8, 12, 16)
def plot_subject_hand_ldlj_boxplot_with_grouped_means(reach_TW_metrics):
    hands = ['right', 'left']
    data = []

    for hand in hands:
        all_subjects = list(reach_TW_metrics['reach_LDLJ'].keys())
        for subject in all_subjects:
            if hand not in reach_TW_metrics['reach_LDLJ'][subject]:
                continue
            ldlj_matrix = np.array([values for values in reach_TW_metrics['reach_LDLJ'][subject][hand].values()])
            
            # Group reaches into (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15), (4, 8, 12, 16)
            grouped_means = []
            for group in range(4):
                group_indices = range(group, ldlj_matrix.shape[1], 4)
                group_values = ldlj_matrix[:, group_indices].flatten()
                grouped_means.append(group_values.mean())

            # Collect grouped means and standard deviations for each subject and hand
            for group_idx, mean_value in enumerate(grouped_means, start=1):
                data.append({
                    'Hand': hand.capitalize(),
                    'Group': f'Group {group_idx}',
                    'Mean LDLJ': mean_value,
                    'Subject': subject
                })

    df = pd.DataFrame(data)

    # Create subplots for left and right hands
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    for ax, hand in zip(axes, hands):
        sns.boxplot(
            ax=ax, x='Group', y='Mean LDLJ', 
            data=df[df['Hand'] == hand.capitalize()], 
            palette='Set2', showfliers=False
        )
        sns.stripplot(
            ax=ax, x='Group', y='Mean LDLJ', 
            data=df[df['Hand'] == hand.capitalize()], 
            dodge=True, alpha=0.6, marker='o', size=8, hue='Subject', palette='husl'
        )
        
        ax.set_title(f'LDLJ Grouped Means for {hand.capitalize()} Hand', fontsize=14)
        ax.set_xlabel('Reach Group', fontsize=12)
        ax.set_ylabel('Mean LDLJ Value', fontsize=12)
        ax.legend(title='Subject', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

plot_subject_hand_ldlj_boxplot_with_grouped_means(reach_TW_metrics)

# Plot a heatmap of overall average LDLJ values for a specific hand, averaging across columns and showing as a subplot.
def plot_average_ldlj_value_heatmap_with_column_average(reach_TW_metrics, hand):
    all_subjects = list(reach_TW_metrics['reach_LDLJ'].keys())
    all_average_values = []

    for subject in all_subjects:
        if hand not in reach_TW_metrics['reach_LDLJ'][subject]:
            continue
        ldlj_matrix = np.array([values for values in reach_TW_metrics['reach_LDLJ'][subject][hand].values()])
        all_average_values.append(ldlj_matrix.mean(axis=0))

    all_average_values = np.array(all_average_values)
    overall_average = all_average_values.mean(axis=0)

    # Rearrange the overall average into the specified order
    if hand.lower() == 'right':
        rearranged_indices = [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]
    elif hand.lower() == 'left':
        rearranged_indices = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    else:
        raise ValueError("Invalid hand specified. Use 'right' or 'left'.")

    rearranged_average = overall_average[rearranged_indices]

    # Reshape the rearranged average into a 4x4 grid
    grid_size = 4
    rearranged_average_reshaped = rearranged_average.reshape(grid_size, grid_size)

    # Calculate column averages
    column_averages = rearranged_average_reshaped.mean(axis=0)
    print(column_averages)

    # Create subplots for the heatmap and column averages
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [4, 1]})

    # Heatmap subplot
    ax1 = axes[0]
    im = ax1.imshow(rearranged_average_reshaped, aspect='auto', cmap='viridis', interpolation='nearest')
    ax1.set_title(f"Overall Average LDLJ Values for Hand: {hand.capitalize()}", fontsize=14, fontweight='bold')
    ax1.set_xticks(range(grid_size))
    ax1.set_xticklabels(range(1, grid_size + 1))
    ax1.set_yticks(range(grid_size))
    ax1.set_yticklabels(range(1, grid_size + 1))
    ax1.set_xlabel("Reach Index (Columns)")
    ax1.set_ylabel("Reach Index (Rows)")
    plt.colorbar(im, ax=ax1, orientation='vertical', label='LDLJ Values')

    # Annotate the heatmap with LDLJ values
    for i in range(grid_size):
        for j in range(grid_size):
            ax1.text(j, i, f'{rearranged_average_reshaped[i, j]:.2f}', ha='center', va='center', color='white', fontsize=8)

    # Column averages subplot
    ax2 = axes[1]
    ax2.barh(range(grid_size), column_averages, color='skyblue')
    ax2.set_yticks(range(grid_size))
    ax2.set_yticklabels(range(1, grid_size + 1))
    ax2.set_xlabel("Average LDLJ Value")
    ax2.set_title("Column Averages", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

plot_average_ldlj_value_heatmap_with_column_average(reach_TW_metrics, 'left')

# Calculate mean, median, IQR, and SD for both hands and return it as sparc_parameters
def calculate_sparc_statistics(reach_sparc_test_windows):
    sparc_parameters = {}
    participants = list(reach_sparc_test_windows.keys())
    hands = ['right', 'left']

    for participant in participants:
        sparc_parameters[participant] = {}
        for hand in hands:
            sparc_data = reach_sparc_test_windows[participant][hand]
            sparc_matrix = np.array([values for values in sparc_data.values()])
            
            # Remove outliers using z-score
            sparc_matrix = sparc_matrix[(np.abs(zscore(sparc_matrix, axis=None)) < 3).all(axis=1)]
            
            # Flatten the matrix to calculate statistics
            sparc_values = sparc_matrix.flatten()
            
            # Calculate statistics
            mean = np.mean(sparc_values)
            median = np.median(sparc_values)
            iqr = np.percentile(sparc_values, 75) - np.percentile(sparc_values, 25)
            sd = np.std(sparc_values)
            q1 = np.percentile(sparc_values, 25)
            q3 = np.percentile(sparc_values, 75)
            
            # Store statistics for the participant and hand
            sparc_parameters[participant][hand] = {
                'mean': mean,
                'median': median,
                'iqr': iqr,
                'sd': sd,
                'q1': q1,
                'q3': q3
            }
    
    return sparc_parameters

sparc_parameters = calculate_sparc_statistics(reach_sparc_test_windows_1)

# Calculate mean, median, IQR, and SD for both hands and return it as ldlj_parameters
def calculate_ldlj_statistics(reach_TW_metrics):
    ldlj_parameters = {}
    participants = list(reach_TW_metrics['reach_LDLJ'].keys())
    hands = ['right', 'left']

    for participant in participants:
        ldlj_parameters[participant] = {}
        for hand in hands:
            ldlj_matrix = np.array([values for values in reach_TW_metrics['reach_LDLJ'][participant][hand].values()])
            
            # Remove outliers using z-score
            ldlj_matrix = ldlj_matrix[(np.abs(zscore(ldlj_matrix, axis=None)) < 3).all(axis=1)]
            
            # Flatten the matrix to calculate statistics
            ldlj_values = ldlj_matrix.flatten()
            
            # Calculate statistics
            mean = np.mean(ldlj_values)
            median = np.median(ldlj_values)
            iqr = np.percentile(ldlj_values, 75) - np.percentile(ldlj_values, 25)
            sd = np.std(ldlj_values)
            q1 = np.percentile(ldlj_values, 25)
            q3 = np.percentile(ldlj_values, 75)
            
            # Store statistics for the participant and hand
            ldlj_parameters[participant][hand] = {
                'mean': mean,
                'median': median,
                'iqr': iqr,
                'sd': sd,
                'q1': q1,
                'q3': q3
            }
    
    return ldlj_parameters

ldlj_parameters = calculate_ldlj_statistics(reach_TW_metrics)

# Plot mean and standard deviation for SPARC and LDLJ for both hands as box plots with paired t-test
def plot_mean_sd_boxplot_with_ttest(sparc_parameters, ldlj_parameters):
    data = []
    for metric, parameters in [('SPARC', sparc_parameters), ('LDLJ', ldlj_parameters)]:
        for participant, hand_data in parameters.items():
            for hand, stats in hand_data.items():
                data.append({
                    'Metric': metric,
                    'Hand': hand.capitalize(),
                    'Mean': stats['mean'],
                    'SD': stats['sd'],
                    'Participant': participant
                })

    df = pd.DataFrame(data)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    palette = sns.color_palette("husl", len(df['Participant'].unique()))
    participant_colors = {p: palette[i] for i, p in enumerate(df['Participant'].unique())}

    for ax, metric in zip(axes, ['SPARC', 'LDLJ']):
        sns.boxplot(ax=ax, x='Hand', y='Mean', data=df[df['Metric'] == metric], palette='Set2', showfliers=False)
        sns.stripplot(
            ax=ax, x='Hand', y='Mean', data=df[df['Metric'] == metric], dodge=True, 
            palette=participant_colors, alpha=0.6, marker='o', hue='Participant', size=8
        )
        ax.set_title(f'Mean Values of {metric} for Both Hands')
        ax.set_ylabel(f'{metric} Mean Value')
        ax.set_ylim(df[df['Metric'] == metric]['Mean'].min() - 0.1, df[df['Metric'] == metric]['Mean'].max() + 0.1)
        ax.set_xlabel('Hand')
        ax.legend(title='Participant', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Perform paired t-test between hands
        right_hand_means = df[(df['Metric'] == metric) & (df['Hand'] == 'Right')]['Mean']
        left_hand_means = df[(df['Metric'] == metric) & (df['Hand'] == 'Left')]['Mean']
        t_stat, p_value = ttest_rel(right_hand_means, left_hand_means)

        # Annotate t-test result on the plot title
        ax.set_title(f'Mean Values of {metric} for Both Hands\nPaired t-test: t={t_stat:.2f}, p={p_value:.3e}')

    plt.tight_layout()
    plt.show()

plot_mean_sd_boxplot_with_ttest(sparc_parameters, ldlj_parameters)


# SPARC vs LDLJ Correlation
# Plot scatter plot of LDLJ vs. SPARC values for a specific subject and hand
utils3.plot_ldlj_sparc_correlation_by_trial(
    reach_TW_metrics=reach_TW_metrics,
    reach_sparc_test_windows_1=reach_sparc_test_windows_1,
    subject='07/23/AK',
    hand='right'
)

# Plot scatter plot of LDLJ vs. SPARC values for a specific subject and hand
utils3.plot_ldlj_sparc_scatter_by_trial(
    reach_TW_metrics=reach_TW_metrics,
    reach_sparc_test_windows_1=reach_sparc_test_windows_1,
    subject='07/23/AK',
    hand='right'
)

# --- PLOT EACH SEGMENT SPEED AS SUBPLOT WITH LDLJ AND SPARC VALUES ---
utils3.plot_all_segments_with_ldlj_and_sparc(Figure_folder, test_windows_6, results, reach_TW_metrics, reach_sparc_test_windows_1)

# Plot scatter plot of LDLJ vs. SPARC values for a specific subject and hand
utils3.plot_ldlj_sparc_correlation_by_trial(
    reach_TW_metrics=reach_TW_metrics,
    reach_sparc_test_windows_1=reach_sparc_test_windows_1,
    subject='07/23/AK',
    hand='right'
)

# Plot scatter plot of LDLJ vs. SPARC values for a specific subject and hand
utils3.plot_ldlj_sparc_scatter_by_trial(
    reach_TW_metrics=reach_TW_metrics,
    reach_sparc_test_windows_1=reach_sparc_test_windows_1,
    subject='07/23/AK',
    hand='right'
)





