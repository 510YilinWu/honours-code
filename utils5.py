import pickle
import os
import numpy as np


# --- UPDATE BLOCK DISTANCE KEYS TO MATCH FILENAMES IN REACH METRICS ---
def update_block_distance_keys(Block_Distance, reach_metrics, reach_sparc_test_windows_1, reach_TW_metrics):
    """
    Updates the keys of Block_Distance to match the filenames in reach_metrics for each subject and hand.

    Args:
        Block_Distance (dict): Dictionary containing block distance data.
        reach_metrics (dict): Dictionary containing reach metrics data.
        reach_sparc_test_windows_1 (dict): Dictionary containing SPARC test window data.
        reach_TW_metrics (dict): Dictionary containing time window metrics data.

    Returns:
        None: Updates Block_Distance in place.
    """
    for subject in Block_Distance:
        for hand in Block_Distance[subject]:
            if subject in reach_metrics['reach_durations'] and hand in reach_metrics['reach_durations'][subject] and \
               subject in reach_sparc_test_windows_1 and hand in reach_sparc_test_windows_1[subject] and \
               subject in reach_TW_metrics['reach_LDLJ'] and hand in reach_TW_metrics['reach_LDLJ'][subject] and \
               len(Block_Distance[subject][hand]) == len(reach_metrics['reach_durations'][subject][hand]) == \
               len(reach_sparc_test_windows_1[subject][hand]) == len(reach_TW_metrics['reach_LDLJ'][subject][hand]):

                filenames = list(reach_metrics['reach_durations'][subject][hand].keys())  # Get filenames in a list

                if len(filenames) != len(Block_Distance[subject][hand]):
                    print(f"Error: Mismatch in lengths for subject {subject}, hand {hand}.")
                    print(f"Filenames length: {len(filenames)}, Block_Distance length: {len(Block_Distance[subject][hand])}")
                else:
                    # Update Block_Distance keys to match filenames
                    updated_distance = {filenames[i]: v for i, v in enumerate(Block_Distance[subject][hand].values())}
                    Block_Distance[subject][hand] = updated_distance

                    # # Example usage
                    # print(f"Subject: {subject}, Hand: {hand}")
                    # print(Block_Distance[subject][hand])

# --- COMBINE DURATIONS, SPARC, LDLJ, AND DISTANCE, CALCULATED SPEED AND ACCURACY FOR ALL DATES ---
def combine_metrics_for_all_dates(reach_metrics, reach_sparc_test_windows_1, reach_TW_metrics, Block_Distance, all_dates):
    """
    Combines reach durations, SPARC, LDLJ, and distance metrics into a single dictionary for all dates and hands.

    Args:
        reach_metrics (dict): Dictionary containing reach durations.
        reach_sparc_test_windows_1 (dict): Dictionary containing SPARC metrics.
        reach_TW_metrics (dict): Dictionary containing LDLJ metrics.
        Block_Distance (dict): Dictionary containing distance metrics.
        all_dates (list): List of all dates to process.

    Returns:
        dict: Combined metrics for all dates and hands.
    """
    combined_metrics = {}

    for date in all_dates:
        combined_metrics[date] = {}
        for hand in ['left', 'right']:
            if (
                date in reach_metrics['reach_durations'] and hand in reach_metrics['reach_durations'][date] and
                date in reach_sparc_test_windows_1 and hand in reach_sparc_test_windows_1[date] and
                date in reach_TW_metrics['reach_LDLJ'] and hand in reach_TW_metrics['reach_LDLJ'][date] and
                date in Block_Distance and hand in Block_Distance[date]
            ):
                combined_metrics[date][hand] = {
                    "durations": {k: np.float64(v) for k, v in reach_metrics['reach_durations'][date][hand].items()},
                    "sparc": {k: np.float64(v) for k, v in reach_sparc_test_windows_1[date][hand].items()},
                    "ldlj": {k: np.float64(v) for k, v in reach_TW_metrics['reach_LDLJ'][date][hand].items()},
                    "distance": {k: np.float64(v) for k, v in Block_Distance[date][hand].items()}
                }
                    # "speed": {k: 1 / np.float64(v) if v != 0 else np.nan for k, v in reach_metrics['reach_durations'][date][hand].items()},
                    # "accuracy": {k: 1 / np.float64(v) if v != 0 else np.nan for k, v in Block_Distance[date][hand].items()}

    return combined_metrics

# --- CALCULATE MOTOR ACUITY FOR ALL REACHES, EACH HAND ---
# def calculate_motor_acuity_for_all(all_combined_metrics):
#     for subject in all_combined_metrics:
#         hands = ['left', 'right']

#         for hand in hands:
#             trials = all_combined_metrics[subject][hand]['speed'].keys()

#             for trial_path in trials:
#                 speeds = list(all_combined_metrics[subject][hand]['speed'][trial_path])
#                 accuracies = list(all_combined_metrics[subject][hand]['accuracy'][trial_path])

#                 # Calculate motor acuity for all reaches
#                 motor_acuity_list = []
#                 for reach_index in range(len(speeds)):
#                     if np.isnan(accuracies[reach_index]):
#                         motor_acuity = np.nan
#                     else:
#                         motor_acuity = np.sqrt(speeds[reach_index]**2 + accuracies[reach_index]**2)
#                     motor_acuity_list.append(motor_acuity)

#                 if 'motor_acuity' not in all_combined_metrics[subject][hand]:
#                     all_combined_metrics[subject][hand]['motor_acuity'] = {}
#                 all_combined_metrics[subject][hand]['motor_acuity'][trial_path] = motor_acuity_list

#     return all_combined_metrics


# def calculate_motor_acuity_for_all(all_combined_metrics, use_group_stats=True):
#     """
#     Computes motor acuity per subject and hand using the 'SAT intersection' method.
#     Your dataset stores 'accuracy' as PRECISION (1/distance to center, higher=better).
    
#     Steps:
#       1) Fit SAT line: S = a + bE  (speed vs. error).
#       2) Convert to precision: E = 1/P  -> S(P) = a + b/P.
#       3) Build diagonal S = mP using mean+3σ across group (or per subject if use_group_stats=False).
#       4) Solve intersection (P*, S*).
    
#     Returns:
#       Adds all_combined_metrics[subject][hand]['motor_acuity'] dict with:
#         a, b, m, P_star, S_star, diag_stats, and per-trial mean table.
#     """

#     # --------- 1) Collect trial-level stats across all subjects/hands ----------
#     group_S, group_P = [], []
#     per_sh = {}

#     for subject, subj_data in all_combined_metrics.items():
#         for hand in ['left', 'right']:
#             if hand not in subj_data:
#                 continue
#             speed_dict = subj_data[hand].get('speed', {})
#             acc_dict   = subj_data[hand].get('accuracy', {})  # already precision

#             rows = []
#             for trial_path, speeds in speed_dict.items():
#                 if trial_path not in acc_dict:
#                     continue
#                 accs = acc_dict[trial_path]

#                 s = np.asarray(speeds, dtype=float)
#                 P = np.asarray(accs,   dtype=float)   # your accuracy = precision
#                 E = 1.0 / P                            # error = 1/precision

#                 # filter out bad values
#                 mask = np.isfinite(s) & np.isfinite(P) & (P > 0)
#                 if not np.any(mask):
#                     continue

#                 mean_S = float(np.nanmean(s[mask]))
#                 mean_P = float(np.nanmean(P[mask]))
#                 mean_E = float(np.nanmean(1.0 / P[mask]))

#                 rows.append((trial_path, mean_S, mean_E, mean_P))
#                 group_S.append(mean_S)
#                 group_P.append(mean_P)

#             per_sh[(subject, hand)] = rows

#     group_S = np.array(group_S)
#     group_P = np.array(group_P)

#     # --------- 2) Group diagonal stats (if requested) ----------
#     group_diag = None
#     if use_group_stats and group_S.size > 0 and group_P.size > 0:
#         mu_S, sigma_S = float(np.mean(group_S)), float(np.std(group_S, ddof=1))
#         mu_P, sigma_P = float(np.mean(group_P)), float(np.std(group_P, ddof=1))
#         denom = (mu_P + 3*sigma_P) if (mu_P + 3*sigma_P) > 0 else 1e-12
#         m_group = (mu_S + 3*sigma_S) / denom
#         group_diag = dict(mu_S=mu_S, sigma_S=sigma_S, mu_P=mu_P, sigma_P=sigma_P, m=m_group)

#     # --------- 3) Per subject/hand computation ----------
#     for (subject, hand), rows in per_sh.items():
#         if len(rows) < 2:  # not enough trials
#             all_combined_metrics[subject][hand]['motor_acuity'] = {
#                 'a': np.nan, 'b': np.nan, 'm': np.nan,
#                 'P_star': np.nan, 'S_star': np.nan,
#                 'diag_stats': group_diag, 'trial_table': rows
#             }
#             continue

#         trial_paths, S_arr, E_arr, P_arr = zip(*rows)
#         S_arr, E_arr, P_arr = map(np.array, (S_arr, E_arr, P_arr))

#         # Fit SAT in error-space: S = a + bE
#         good = np.isfinite(S_arr) & np.isfinite(E_arr)
#         if np.sum(good) >= 2:
#             b, a = np.polyfit(E_arr[good], S_arr[good], 1)  # np.polyfit returns slope, intercept
#         else:
#             a = b = np.nan

#         # Choose diagonal slope m
#         if group_diag is not None:
#             mu_S, sigma_S, mu_P, sigma_P, m = (
#                 group_diag['mu_S'], group_diag['sigma_S'],
#                 group_diag['mu_P'], group_diag['sigma_P'],
#                 group_diag['m']
#             )
#             diag_stats = dict(mu_S=mu_S, sigma_S=sigma_S, mu_P=mu_P, sigma_P=sigma_P)
#         else:  # per subject/hand
#             mu_S, sigma_S = float(np.mean(S_arr)), float(np.std(S_arr, ddof=1))
#             mu_P, sigma_P = float(np.mean(P_arr)), float(np.std(P_arr, ddof=1))
#             denom = (mu_P + 3*sigma_P) if (mu_P + 3*sigma_P) > 0 else 1e-12
#             m = (mu_S + 3*sigma_S) / denom
#             diag_stats = dict(mu_S=mu_S, sigma_S=sigma_S, mu_P=mu_P, sigma_P=sigma_P)

#         # Intersect: mP^2 - aP - b = 0
#         if np.any(np.isnan([a, b, m])):
#             P_star = S_star = np.nan
#         else:
#             roots = np.roots([m, -a, -b])
#             roots = roots[np.isreal(roots)].real
#             roots = roots[roots > 0]
#             if roots.size == 0:
#                 P_star = S_star = np.nan
#             else:
#                 P_star = float(np.max(roots))
#                 S_star = float(m * P_star)

#         # Save result
#         trial_table = {
#             tp: {'mean_speed': ms, 'mean_error': me, 'mean_precision': mp}
#             for tp, ms, me, mp in rows
#         }
#         all_combined_metrics[subject][hand]['motor_acuity'] = {
#             'a': float(a) if np.isfinite(a) else np.nan,
#             'b': float(b) if np.isfinite(b) else np.nan,
#             'm': float(m) if np.isfinite(m) else np.nan,
#             'P_star': P_star, 'S_star': S_star,
#             'diag_stats': diag_stats,
#             'trial_table': trial_table
#         }

#     return all_combined_metrics

# --- LOCATE NaN INDICES (UNDETECTED BLOCK) FOR ALL SUBJECTS ---
def find_nan_indices_all_subjects(all_combined_metrics):
    nan_reach_indices = {}

    for subject in all_combined_metrics:
        nan_reach_indices[subject] = {}
        for hand in ['left', 'right']:
            if hand in all_combined_metrics[subject]:
                distance = all_combined_metrics[subject][hand]['distance']
                accuracy = all_combined_metrics[subject][hand]['accuracy']
                motor_acuity = all_combined_metrics[subject][hand]['motor_acuity']

                nan_indices = [
                    (trial_idx, value_idx)
                    for trial_idx, trial in enumerate(distance.values())
                    for value_idx, value in enumerate(trial)
                    if np.isnan(value)
                ]

                if not (
                    nan_indices == [
                        (trial_idx, value_idx)
                        for trial_idx, trial in enumerate(accuracy.values())
                        for value_idx, value in enumerate(trial)
                        if np.isnan(value)
                    ] == [
                        (trial_idx, value_idx)
                        for trial_idx, trial in enumerate(motor_acuity.values())
                        for value_idx, value in enumerate(trial)
                        if np.isnan(value)
                    ]
                ):
                    print(
                        f"Subject: {subject}, Hand: {hand} - NaN indices do not match."
                    )
                nan_reach_indices[subject][hand] = nan_indices

    return nan_reach_indices

# --- SAVE ALL COMBINED METRICS PER SUBJECT AS PICKLE FILE ---
def save_combined_metrics_per_subject(all_combined_metrics, output_folder):
    """
    Saves the combined metrics dictionary as separate pickle files for each subject.

    Args:
        all_combined_metrics (dict): The combined metrics to save.
        output_folder (str): The folder where the pickle files will be saved.

    Returns:
        None
    """

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for subject, metrics in all_combined_metrics.items():
        # Replace slashes in subject names to create valid file paths
        sanitized_subject = subject.replace("/", "_")
        output_file = f"{output_folder}/{sanitized_subject}_combined_metrics.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(metrics, f)
        print(f"Combined metrics for subject {subject} saved to {output_file}")

# --- LOAD SELECTED SUBJECT RESULTS ---
def load_selected_subject_results(selected_subjects, DataProcess_folder):
    """
    Loads the processed results for selected subjects and aggregates them into a single dictionary.

    Args:
        selected_subjects (list): List of subjects to load results for.
        DataProcess_folder (str): Path to the data processing folder.

    Returns:
        dict: A dictionary containing the aggregated results for all selected subjects.
    """
    results = {}
    for subject in selected_subjects:
        subject_filename = f"{subject.replace('/', '_')}_combined_metrics.pkl"
        try:
            file_path = os.path.join(DataProcess_folder, subject_filename)
            with open(file_path, 'rb') as f:
                results[subject] = pickle.load(f)
            print(f"Results for subject {subject} loaded from {file_path}")
        except FileNotFoundError:
            print(f"Warning: Results file for {subject} not found. Skipping.")
    return results

# --- PROCESS AND SAVE COMBINED METRICS FOR ALL SUBJECTS ---
def process_and_save_combined_metrics(Block_Distance, reach_metrics, reach_sparc_test_windows_1, reach_TW_metrics, All_dates, DataProcess_folder):
    """
    Combines multiple processing steps into one function:
    1. Updates Block_Distance keys to match filenames in reach_metrics.
    2. Combines durations, SPARC, LDLJ, and distance, and calculates speed and accuracy for all dates.
    3. Calculates motor acuity for all reaches for each hand.
    4. Saves all combined metrics per subject as pickle files.

    Args:
        Block_Distance (dict): Dictionary containing block distance data.
        reach_metrics (dict): Dictionary containing reach metrics data.
        reach_sparc_test_windows_1 (dict): Dictionary containing SPARC test window data.
        reach_TW_metrics (dict): Dictionary containing time window metrics data.
        All_dates (list): List of all dates to process.
        DataProcess_folder (str): The folder where the pickle files will be saved.

    Returns:
        None
    """
    # Step 1: Update Block_Distance keys
    update_block_distance_keys(Block_Distance, reach_metrics, reach_sparc_test_windows_1, reach_TW_metrics)

    # Step 2: Combine metrics for all dates
    all_combined_metrics = combine_metrics_for_all_dates(
        reach_metrics, reach_sparc_test_windows_1, reach_TW_metrics, Block_Distance, All_dates
    )

    # Step 3: Calculate motor acuity for all reaches
    # all_combined_metrics = calculate_motor_acuity_for_all(all_combined_metrics)

    # Step 4: Save combined metrics per subject
    save_combined_metrics_per_subject(all_combined_metrics, DataProcess_folder)

# -------------------------------------------------------------------------------------------------------------------

# # --- PLOT ---
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from scipy.stats import spearmanr
# from statsmodels.nonparametric.smoothers_lowess import lowess


# # Scatter plot for durations vs distance for all trials, overlaying them with different colors
# def plot_durations_vs_distance_hand(all_combined_metrics, subject, hand):
#     trials = all_combined_metrics[subject][hand]['durations'].keys()
#     colors = sns.color_palette("Blues", len(trials))  # Generate a color palette from light to dark

#     plt.figure(figsize=(10, 8))
#     for i, trial_path in enumerate(trials):
#         durations = all_combined_metrics[subject][hand]['durations'][trial_path]
#         distances = all_combined_metrics[subject][hand]['distance'][trial_path]

#         # Ensure the lengths match
#         if len(durations) != len(distances):
#             print(f"Mismatch in the number of durations and distances data points for trial {trial_path}.")
#             continue

#         # Scatter plot for each trial
#         plt.scatter(durations, distances, color=colors[i], edgecolor='k', alpha=0.7, label=f'Trial {i+1}')

#     # Calculate x and y limits based on data
#     valid_durations = [d for trial_path in trials for d in all_combined_metrics[subject][hand]['durations'][trial_path] if not np.isnan(d) and not np.isinf(d)]
#     valid_distances = [d for trial_path in trials for d in all_combined_metrics[subject][hand]['distance'][trial_path] if not np.isnan(d) and not np.isinf(d)]

#     if valid_durations and valid_distances:
#         x_min, x_max = np.percentile(valid_durations, [0.5, 99.5])
#         y_min, y_max = np.percentile(valid_distances, [0.5, 99.5])
#     else:
#         print("Error: No valid data points for axis limits.")
#         return

#     plt.xlim(x_min, x_max)
#     plt.ylim(y_min, y_max)

#     plt.title(f"Durations vs Distance for {subject} ({hand.capitalize()} Hand)", fontsize=14)
#     plt.xlabel("Durations (s)", fontsize=12)
#     plt.ylabel("Distance (mm)", fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.legend(title="Trials", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
#     plt.tight_layout()
#     plt.show()

# # Scatter plot for durations vs distance for all trials, each hand as a subplot
# def plot_durations_vs_distance_hands(all_combined_metrics, subject):
#     hands = ['left', 'right']
#     fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

#     for ax, hand in zip(axes, hands):
#         trials = all_combined_metrics[subject][hand]['durations'].keys()
#         colors = sns.color_palette("Blues", len(trials))  # Generate a color palette from light to dark

#         for i, trial_path in enumerate(trials):
#             durations = all_combined_metrics[subject][hand]['durations'][trial_path]
#             distances = all_combined_metrics[subject][hand]['distance'][trial_path]

#             # Ensure the lengths match
#             if len(durations) != len(distances):
#                 print(f"Mismatch in the number of durations and distances data points for trial {trial_path}.")
#                 continue

#             # Scatter plot for each trial
#             ax.scatter(durations, distances, color=colors[i], edgecolor='k', alpha=0.7, label=f'Trial {i+1}')

#         # Calculate x and y limits based on data
#         valid_durations = [d for trial_path in trials for d in all_combined_metrics[subject][hand]['durations'][trial_path] if not np.isnan(d) and not np.isinf(d)]
#         valid_distances = [d for trial_path in trials for d in all_combined_metrics[subject][hand]['distance'][trial_path] if not np.isnan(d) and not np.isinf(d)]

#         if valid_durations and valid_distances:
#             x_min, x_max = np.percentile(valid_durations, [0.5, 99.5])
#             y_min, y_max = np.percentile(valid_distances, [0.5, 99.5])
#             ax.set_xlim(x_min, x_max)
#             ax.set_ylim(y_min, y_max)
#         else:
#             print(f"Error: No valid data points for axis limits for {hand} hand.")
#             continue

#         ax.set_title(f"Durations vs Distance for {subject} ({hand.capitalize()} Hand)", fontsize=14)
#         ax.set_xlabel("Durations (s)", fontsize=12)
#         ax.set_ylabel("Distance (mm)", fontsize=12)
#         ax.grid(True, linestyle='--', alpha=0.6)
#         ax.legend(title="Trials", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

#     plt.tight_layout()
#     plt.show()

# # Scatter plot for speed vs accuracy for all trials, each hand as a subplot
# def plot_speed_vs_accuracy(all_combined_metrics, subject):
#     hands = ['left', 'right']
#     fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

#     for ax, hand in zip(axes, hands):
#         trials = all_combined_metrics[subject][hand]['speed'].keys()
#         colors = sns.color_palette("Blues", len(trials))  # Generate a color palette from light to dark

#         for i, trial_path in enumerate(trials):
#             speeds = list(all_combined_metrics[subject][hand]['speed'][trial_path])
#             accuracies = list(all_combined_metrics[subject][hand]['accuracy'][trial_path])
#             if any(np.isnan(accuracies)):
#                 motor_acuity = np.nan
#                 continue

#             # Ensure the lengths match
#             if len(speeds) != len(accuracies):
#                 print(f"Mismatch in the number of speeds and accuracies data points for trial {trial_path}.")
#                 continue

#             # Scatter plot for each trial
#             ax.scatter(speeds, accuracies, color=colors[i], edgecolor='k', alpha=0.7, label=f'Trial {i+1}')

#         # Calculate x and y limits based on data
#         valid_speeds = [s for trial_path in trials for s in all_combined_metrics[subject][hand]['speed'][trial_path] if not np.isnan(s) and not np.isinf(s)]
#         valid_accuracies = [a for trial_path in trials for a in all_combined_metrics[subject][hand]['accuracy'][trial_path] if not np.isnan(a) and not np.isinf(a)]

#         if valid_speeds and valid_accuracies:
#             x_min, x_max = np.percentile(valid_speeds, [0.5, 99.5])
#             y_min, y_max = np.percentile(valid_accuracies, [0.5, 99.5])
#             ax.set_xlim(x_min, x_max)
#             ax.set_ylim(y_min, y_max)
#         else:
#             print(f"Error: No valid data points for axis limits for {hand} hand.")
#             continue

#         ax.set_title(f"Speed vs Accuracy for {subject} ({hand.capitalize()} Hand)", fontsize=14)
#         ax.set_xlabel("Speed (1/Duration)\n(Slow → Fast)", fontsize=8)
#         ax.set_ylabel("Accuracy (1/Distance)\n(Bad → Good)", fontsize=8)
#         ax.grid(True, linestyle='--', alpha=0.6)
#         ax.legend(title="Trials", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

#     plt.tight_layout()
#     plt.show()

# # Scatter plot for speed vs accuracy for a single reach, each hand as a subplot
# # def plot_speed_vs_accuracy_single_reach(all_combined_metrics, subject, reach_index):
# #     hands = ['left', 'right']
# #     fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

# #     for ax, hand in zip(axes, hands):
# #         trials = all_combined_metrics[subject][hand]['speed'].keys()
# #         colors = sns.color_palette("Blues", len(trials))  # Generate a color palette from light to dark

# #         for i, trial_path in enumerate(trials):
# #             speeds = list(all_combined_metrics[subject][hand]['speed'][trial_path])
# #             accuracies = list(all_combined_metrics[subject][hand]['accuracy'][trial_path])

# #             # Ensure the reach index is valid
# #             if reach_index >= len(speeds) or reach_index >= len(accuracies):
# #                 print(f"Invalid reach index {reach_index} for trial {trial_path}.")
# #                 continue

# #             # Scatter plot for the specified reach
# #             ax.scatter(speeds[reach_index], accuracies[reach_index], color=colors[i], edgecolor='k', alpha=0.7, label=f'Trial {i+1}')

# #         # Calculate x and y limits based on data
# #         valid_speeds = [s[reach_index] for trial_path in trials for s in [all_combined_metrics[subject][hand]['speed'][trial_path]] if not np.isnan(s[reach_index]) and not np.isinf(s[reach_index])]
# #         valid_accuracies = [a[reach_index] for trial_path in trials for a in [all_combined_metrics[subject][hand]['accuracy'][trial_path]] if not np.isnan(a[reach_index]) and not np.isinf(a[reach_index])]

# #         if valid_speeds and valid_accuracies:
# #             x_min, x_max = np.percentile(valid_speeds, [1, 90])
# #             y_min, y_max = np.percentile(valid_accuracies, [0.5, 90])
# #             # x_min, x_max = min(valid_speeds), max(valid_speeds)
# #             # y_min, y_max = min(valid_accuracies), max(valid_accuracies)
# #             ax.set_xlim(x_min, x_max)
# #             ax.set_ylim(y_min, y_max)
# #         else:
# #             print(f"Error: No valid data points for axis limits for {hand} hand.")
# #             continue

# #         ax.set_title(f"Speed vs Accuracy for {subject} ({hand.capitalize()} Hand, Reach {reach_index + 1})", fontsize=14)
# #         ax.set_xlabel("Speed (1/Duration)\n(Slow → Fast)", fontsize=8)
# #         ax.set_ylabel("Accuracy (1/Distance)\n(Bad → Good)", fontsize=8)
# #         ax.grid(True, linestyle='--', alpha=0.6)
# #         ax.legend(title="Trials", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

# #     plt.tight_layout()
# #     plt.show()


# # Scatter plot for speed vs accuracy for a single reach, each hand as a subplot
# def plot_speed_vs_accuracy_single_reach(all_combined_metrics, subject, reach_index):
#     hands = ['left', 'right']
#     fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

#     for ax, hand in zip(axes, hands):
#         trials = all_combined_metrics[subject][hand]['speed'].keys()
#         colors = sns.color_palette("Blues", len(trials))  # Generate a color palette from light to dark

#         all_speeds = []
#         all_accuracies = []

#         for i, trial_path in enumerate(trials):
#             speeds = list(all_combined_metrics[subject][hand]['speed'][trial_path])
#             accuracies = list(all_combined_metrics[subject][hand]['accuracy'][trial_path])

#             # Ensure the reach index is valid
#             if reach_index >= len(speeds) or reach_index >= len(accuracies):
#                 print(f"Invalid reach index {reach_index} for trial {trial_path}.")
#                 continue

#             # Scatter plot for the specified reach
#             ax.scatter(speeds[reach_index], accuracies[reach_index], color=colors[i], edgecolor='k', alpha=0.7, label=f'Trial {i+1}')
#             all_speeds.append(speeds[reach_index])
#             all_accuracies.append(accuracies[reach_index])

#         # Calculate x and y limits based on data
#         valid_data = [(s, a) for s, a in zip(all_speeds, all_accuracies) if not np.isnan(s) and not np.isnan(a)]
#         if valid_data:
#             valid_speeds, valid_accuracies = zip(*valid_data)
#             x_min, x_max = np.percentile(valid_speeds, [0.5, 90])
#             y_min, y_max = np.percentile(valid_accuracies, [0.5, 90])
#             ax.set_xlim(x_min, x_max)
#             ax.set_ylim(y_min, y_max)

#             # Calculate and display Spearman correlation
#             spearman_corr, _ = spearmanr(valid_speeds, valid_accuracies)
#             ax.text(0.05, 0.95, f"Spearman r: {spearman_corr:.2f}", transform=ax.transAxes, fontsize=12, verticalalignment='top')
#         else:
#             print(f"Error: No valid data points for axis limits for {hand} hand.")
#             continue

#         ax.set_title(f"Speed vs Accuracy for {subject} ({hand.capitalize()} Hand, Reach {reach_index + 1})", fontsize=14)
#         ax.set_xlabel("Speed (1/Duration)\n(Slow → Fast)", fontsize=14)
#         ax.set_ylabel("Accuracy (1/Distance)\n(Bad → Good)", fontsize=14)
#         ax.grid(True, linestyle='--', alpha=0.6)
#         ax.legend(title="Trials", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

#     plt.tight_layout()
#     plt.show()

# # # Scatter plot for speed vs accuracy for all reaches, each hand as a separate figure, 4x4 layout for each reach
# # def plot_speed_vs_accuracy_all_reaches(all_combined_metrics, subject):
# #     hands = ['left', 'right']
# #     num_reaches = max(len(all_combined_metrics[subject][hand]['speed'][list(all_combined_metrics[subject][hand]['speed'].keys())[0]]) for hand in hands)
# #     grid_size = 4  # 4x4 layout

# #     for hand in hands:
# #         fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16), sharey=True)
# #         axes = axes.flatten()

# #         for reach_index in range(num_reaches):
# #             ax = axes[reach_index]
# #             trials = all_combined_metrics[subject][hand]['speed'].keys()
# #             colors = sns.color_palette("Blues", len(trials))  # Generate a color palette from light to dark

# #             for i, trial_path in enumerate(trials):
# #                 speeds = list(all_combined_metrics[subject][hand]['speed'][trial_path])
# #                 accuracies = list(all_combined_metrics[subject][hand]['accuracy'][trial_path])

# #                 # Ensure the reach index is valid
# #                 if reach_index >= len(speeds) or reach_index >= len(accuracies):
# #                     print(f"Invalid reach index {reach_index} for trial {trial_path}.")
# #                     continue

# #                 # Scatter plot for the specified reach
# #                 ax.scatter(speeds[reach_index], accuracies[reach_index], color=colors[i], edgecolor='k', alpha=0.7, label=f'Trial {i+1}')

# #             # Calculate x and y limits based on data
# #             valid_speeds = [s[reach_index] for trial_path in trials for s in [all_combined_metrics[subject][hand]['speed'][trial_path]] if not np.isnan(s[reach_index]) and not np.isinf(s[reach_index])]
# #             valid_accuracies = [a[reach_index] for trial_path in trials for a in [all_combined_metrics[subject][hand]['accuracy'][trial_path]] if not np.isnan(a[reach_index]) and not np.isinf(a[reach_index])]

# #             if valid_speeds and valid_accuracies:
# #                 x_min, x_max = np.percentile(valid_speeds, [0.5, 99.5])
# #                 y_min, y_max = np.percentile(valid_accuracies, [0.5, 99.5])
# #                 # x_min, x_max = min(valid_speeds), max(valid_speeds)
# #                 # y_min, y_max = min(valid_accuracies), max(valid_accuracies)
# #                 ax.set_xlim(x_min, x_max)
# #                 ax.set_ylim(y_min-0.5, y_max)
# #             # else:
# #             #     print(f"Error: No valid data points for axis limits for {hand} hand.")
# #             #     continue

# #             ax.set_title(f"Reach {reach_index + 1} ({len(valid_accuracies)} points)", fontsize=10)
# #             ax.set_xlabel("Speed (1/Duration)\n(Slow → Fast)", fontsize=8)
# #             ax.set_ylabel("Accuracy (1/Distance)\n(Bad → Good)", fontsize=8)
# #             ax.grid(True, linestyle='--', alpha=0.6)

# #         # Hide unused subplots
# #         for unused_ax in axes[num_reaches:]:
# #             unused_ax.axis('off')

# #         fig.suptitle(f"Speed vs Accuracy for {hand.capitalize()} Hand ({subject})", fontsize=16)
# #         plt.tight_layout(rect=[0, 0, 1, 0.95])
# #         plt.show()

# # Scatter plot for speed vs accuracy for all reaches, each hand as a separate figure, 4x4 layout for each reach
# def plot_speed_vs_accuracy_all_reaches(all_combined_metrics, subject):
#     hands = ['left', 'right']
#     num_reaches = max(len(all_combined_metrics[subject][hand]['speed'][list(all_combined_metrics[subject][hand]['speed'].keys())[0]]) for hand in hands)
#     grid_size = 4  # 4x4 layout

#     for hand in hands:
#         # Determine global x and y ranges for all subplots
#         all_speeds = []
#         all_accuracies = []
#         for trial_path in all_combined_metrics[subject][hand]['speed'].keys():
#             speeds = all_combined_metrics[subject][hand]['speed'][trial_path]
#             accuracies = all_combined_metrics[subject][hand]['accuracy'][trial_path]
#             all_speeds.extend([value for value in speeds if not np.isnan(value)])
#             all_accuracies.extend([value for value in accuracies if not np.isnan(value)])
#         x_min, x_max = np.percentile(all_speeds, [0.5, 99.5])
#         y_min, y_max = np.percentile(all_accuracies, [0.5, 99.5])

#         fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16), sharey=True)
#         axes = axes.flatten()

#         for reach_index in range(num_reaches):
#             ax = axes[reach_index]
#             trials = all_combined_metrics[subject][hand]['speed'].keys()
#             colors = sns.color_palette("Blues", len(trials))

#             reach_speeds = []
#             reach_accuracies = []

#             for i, trial_path in enumerate(trials):
#                 speeds = list(all_combined_metrics[subject][hand]['speed'][trial_path])
#                 accuracies = list(all_combined_metrics[subject][hand]['accuracy'][trial_path])
#                 if reach_index >= len(speeds) or reach_index >= len(accuracies):
#                     continue
#                 ax.scatter(speeds[reach_index], accuracies[reach_index], color=colors[i], edgecolor='k', alpha=0.7)
#                 reach_speeds.append(speeds[reach_index])
#                 reach_accuracies.append(accuracies[reach_index])

#             # Calculate and display Spearman correlation
#             valid_data = [(s, a) for s, a in zip(reach_speeds, reach_accuracies) if not np.isnan(s) and not np.isnan(a)]
#             if valid_data:
#                 valid_speeds, valid_accuracies = zip(*valid_data)
#                 spearman_corr, _ = spearmanr(valid_speeds, valid_accuracies)
#                 ax.text(0.05, 0.95, f"Spearman r: {spearman_corr:.2f}", transform=ax.transAxes, fontsize=8, verticalalignment='top')

#             ax.set_xlim(0, x_max)
#             ax.set_ylim(0, y_max)
#             ax.set_xlabel("Speed (1/Duration)", fontsize=12)
#             ax.set_ylabel("Accuracy (1/Distance)", fontsize=12)
#             ax.grid(True, linestyle='--', alpha=0.6)

#         for unused_ax in axes[num_reaches:]:
#             unused_ax.axis('off')

#         fig.suptitle(f"Speed vs Accuracy for {hand.capitalize()} Hand ({subject})", fontsize=16)
#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         plt.show()

# # Scatter plot for motor_acuity vs sparc for all reaches, each hand as a separate figure, 4x4 layout for each reach
# def plot_motor_acuity_vs_sparc_all_reaches(all_combined_metrics, subject):
#     hands = ['left', 'right']
#     num_reaches = max(len(all_combined_metrics[subject][hand]['motor_acuity'][list(all_combined_metrics[subject][hand]['motor_acuity'].keys())[0]]) for hand in hands)
#     grid_size = 4  # 4x4 layout

#     for hand in hands:
#         # Determine global x and y ranges for all subplots
#         all_motor_acuity = []
#         all_sparc = []
#         for trial_path in all_combined_metrics[subject][hand]['motor_acuity'].keys():
#             motor_acuity_values = all_combined_metrics[subject][hand]['motor_acuity'][trial_path]
#             sparc_values = all_combined_metrics[subject][hand]['sparc'][trial_path]
#             all_motor_acuity.extend([value for value in motor_acuity_values if not pd.isna(value)])
#             all_sparc.extend([value for value in sparc_values if not pd.isna(value)])
#         x_min, x_max = np.percentile(all_motor_acuity, [0.5, 99.5])
#         y_min, y_max = np.percentile(all_sparc, [0.5, 99.5])

#         fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16), sharey=True)
#         axes = axes.flatten()

#         for reach_index in range(num_reaches):
#             ax = axes[reach_index]
#             trials = all_combined_metrics[subject][hand]['motor_acuity'].keys()
#             colors = sns.color_palette("Blues", len(trials))  # Generate a color palette from light to dark

#             for i, trial_path in enumerate(trials):
#                 motor_acuity = list(all_combined_metrics[subject][hand]['motor_acuity'][trial_path])
#                 sparc_values = list(all_combined_metrics[subject][hand]['sparc'][trial_path])

#                 # Ensure the reach index is valid
#                 if reach_index >= len(motor_acuity) or reach_index >= len(sparc_values):
#                     print(f"Invalid reach index {reach_index} for trial {trial_path}.")
#                     continue

#                 # Scatter plot for the specified reach
#                 ax.scatter(motor_acuity[reach_index], sparc_values[reach_index], color=colors[i], edgecolor='k', alpha=0.7, label=f'Trial {i+1}')

#             ax.set_title(f"Reach {reach_index + 1}", fontsize=10)
#             ax.set_xlabel("Motor Acuity\n(Bad → Good)", fontsize=8)
#             ax.set_ylabel("SPARC\n(Unsmooth → Smooth)", fontsize=8)
#             ax.set_xlim(x_min, x_max)
#             ax.set_ylim(y_min, y_max)
#             ax.grid(True, linestyle='--', alpha=0.6)

#         # Hide unused subplots
#         for unused_ax in axes[num_reaches:]:
#             unused_ax.axis('off')

#         fig.suptitle(f"Motor Acuity vs SPARC for {hand.capitalize()} Hand ({subject})", fontsize=16)
#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         plt.show()
