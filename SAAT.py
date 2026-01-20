import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
from scipy.stats import chisquare
from scipy.stats import circmean, rayleigh
import numpy as np
import seaborn as sns
from scipy.stats import ttest_rel
from scipy.spatial.distance import mahalanobis
from scipy.stats import f_oneway, ttest_ind
from itertools import combinations
from scipy.stats import ttest_1samp
from pingouin import intraclass_corr


import utils1 # Importing utils1 for data Pre-processing
import utils2 # Importing utils2 for iBBT reach speed segments and metrics
import utils4 # Importing utils4 for image files

import iBBT_utils # Importing iBBT_utils for iBBT specific functions

import utils8 # Importing utils8 for sBBTResult
import utils9 # Importing utils9 for thesis


Traj_folder = "/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025"
Box_Traj_folder = "/Users/yilinwu/Desktop/Yilin-Honours/Box/Traj/2025"
Figure_folder = "/Users/yilinwu/Desktop/honours/Thesis/figure"
DataProcess_folder = "/Users/yilinwu/Desktop/honours data/DataProcess"
tBBT_Image_folder = "/Users/yilinwu/Desktop/Yilin-Honours/tBBT_Image/2025/"



# Get subject
All_dates = sorted(utils1.get_subfolders_with_depth(Traj_folder, depth=3))

## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## sBBT data
# Load sBBTResult from CSV into a DataFrame and compute right and left hand scores
sBBTResult = utils8.load_and_compute_sbbt_result()

# Swap and rename sBBTResult scores for specific subjects
sBBTResult = utils8.swap_and_rename_sbbt_result(sBBTResult)

# Intraclass correlations between the scores between first and second sBBT tests for each hand


# # Calculate sBBTResult score statistics for dominant and non-dominant columns
sBBTResult_stats = utils8.compute_sbbt_result_stats(sBBTResult)

# # Analyze correlations in two repeated measures of sBBT
# sBBT_combine_stat = utils8.analyze_sbbt_results(sBBTResult)



# Apply Pearson correlation for dominant and non-dominant scores between Test1 and Test2
# Adjust for subjects with left-hand dominance (PY and MC)
import numpy as np
from scipy.stats import pearsonr

# Subjects that are left-handed
left_handed = ['PY', 'MC']

# Ensure Subject is a column
df = sBBTResult.copy()

# Create dominant hand column
df['dominant_hand'] = np.where(df['Subject'].isin(left_handed), 'left', 'right')
# Dominant hand scores
df['dom_test1'] = np.where(df['dominant_hand'] == 'left', df['Left'],  df['Right'])
df['dom_test2'] = np.where(df['dominant_hand'] == 'left', df['Left.1'], df['Right.1'])

# Non-dominant hand scores
df['nondom_test1'] = np.where(df['dominant_hand'] == 'left', df['Right'],  df['Left'])
df['nondom_test2'] = np.where(df['dominant_hand'] == 'left', df['Right.1'], df['Left.1'])

dominant_corr, dominant_p = pearsonr(df['dom_test1'], df['dom_test2'])
print("Pearson correlation for dominant hand:", dominant_corr, "p-value:", dominant_p)

nondom_corr, nondom_p = pearsonr(df['nondom_test1'], df['nondom_test2'])
print("Pearson correlation for non-dominant hand:", nondom_corr, "p-value:", nondom_p)


import numpy as np
import pandas as pd
import pingouin as pg

# Subjects that are left-handed
left_handed = ['PY', 'MC']

# Ensure Subject is a column
df = sBBTResult.copy()

# Create dominant hand column
df['dominant_hand'] = np.where(df['Subject'].isin(left_handed), 'left', 'right')

# Dominant hand scores
df['dom_test1'] = np.where(df['dominant_hand'] == 'left', df['Left'], df['Right'])
df['dom_test2'] = np.where(df['dominant_hand'] == 'left', df['Left.1'], df['Right.1'])

# Non-dominant hand scores
df['nondom_test1'] = np.where(df['dominant_hand'] == 'left', df['Right'], df['Left'])
df['nondom_test2'] = np.where(df['dominant_hand'] == 'left', df['Right.1'], df['Left.1'])

# Prepare data for ICC (long format)
dominant_long = df.melt(
    id_vars=['Subject'],
    value_vars=['dom_test1', 'dom_test2'],
    var_name='rater',
    value_name='score'
)

nondom_long = df.melt(
    id_vars=['Subject'],
    value_vars=['nondom_test1', 'nondom_test2'],
    var_name='rater',
    value_name='score'
)

# Compute ICC(3,1) for dominant hand
icc_dom = pg.intraclass_corr(data=dominant_long, targets='Subject', raters='rater', ratings='score')
icc_dom3_1 = icc_dom[icc_dom['Type'] == 'ICC3']
icc_dom3_1_value = icc_dom3_1['ICC'].values[0]
icc_dom3_1_ci = icc_dom3_1['CI95%'].values[0]
print(f"ICC(3,1) for dominant hand: {icc_dom3_1_value:.3f}, 95% CI: {icc_dom3_1_ci}")

# Compute ICC(3,1) for non-dominant hand
icc_nondom = pg.intraclass_corr(data=nondom_long, targets='Subject', raters='rater', ratings='score')
icc_nondom3_1 = icc_nondom[icc_nondom['Type'] == 'ICC3']
icc_nondom3_1_value = icc_nondom3_1['ICC'].values[0]
icc_nondom3_1_ci = icc_nondom3_1['CI95%'].values[0]
print(f"ICC(3,1) for non-dominant hand: {icc_nondom3_1_value:.3f}, 95% CI: {icc_nondom3_1_ci}")





## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# iBBT data
# --- iBBT - PROCESS ALL DATE AND SAVE ALL MOVEMENT DATA AS pickle file ---
# def process_ibbt_data(all_dates, traj_folder, box_traj_folder, figure_folder, data_process_folder):
#     # Process all dates with default prominence thresholds
#     prominence_threshold_speed = 2.9
#     prominence_threshold_position = 2.9
#     iBBT_utils.process_all_dates_separate(all_dates, traj_folder, box_traj_folder, figure_folder, data_process_folder, 
#                                           prominence_threshold_speed, prominence_threshold_position)

#     # Process specific indices with zero prominence thresholds
#     prominence_threshold_speed = 0
#     prominence_threshold_position = 0
#     indices_to_process = [14, 22, 25, 27]
#     for index in indices_to_process:
#         iBBT_utils.process_all_dates_separate(all_dates[index:index+1], traj_folder, box_traj_folder, figure_folder, data_process_folder, 
#                                               prominence_threshold_speed, prominence_threshold_position)

# process_ibbt_data(All_dates, Traj_folder, Box_Traj_folder, Figure_folder, DataProcess_folder)

# --- LOAD ALL SUBJECTS' tBBT ERROR FROM IMAGE, SAVE AS pickle file---
iBBT_results = iBBT_utils.load_selected_subject_results(All_dates, DataProcess_folder)
iBBT_reach_speed_segments = utils2.get_reach_speed_segments(iBBT_results)
iBBT_reach_metrics = utils2.calculate_reach_metrics(iBBT_reach_speed_segments, iBBT_results, fs=200)
iBBT_test_windows_7 = utils9.compute_test_window_7(iBBT_results, iBBT_reach_speed_segments, iBBT_reach_metrics)

# Count the number of data points in iBBT_reach_speed_segments
def count_data_points(reach_speed_segments):
    """
    Count the total number of data points in the reach speed segments.

    Args:
        reach_speed_segments (dict): Dictionary containing reach speed segments for each subject and trial.

    Returns:
        int: Total number of data points.
    """
    total_count = 0
    for subject, hands in reach_speed_segments.items():
        for hand, trials in hands.items():
            for trial, segments in (trials.items() if isinstance(trials, dict) else enumerate(trials)):
                total_count += len(segments)
    return total_count

# Count the data points in iBBT_reach_speed_segments
total_data_points = count_data_points(iBBT_reach_speed_segments)
print(f"Total number of data points in iBBT_reach_speed_segments: {total_data_points}")

def find_missing_data(reach_speed_segments, expected_trials=4, expected_hands=2, expected_blocks=16):
    """
    Identify subjects with missing data based on the expected number of trials, hands, and blocks.

    Args:
        reach_speed_segments (dict): Dictionary containing reach speed segments for each subject and trial.
        expected_trials (int): Expected number of trials per hand.
        expected_hands (int): Expected number of hands per subject.
        expected_blocks (int): Expected number of blocks per trial.

    Returns:
        list: List of subjects with missing data.
    """
    missing_data_subjects = []

    for subject, hands in reach_speed_segments.items():
        total_trials = 0
        for hand, trials in hands.items():
            total_trials += len(trials)
            for trial, segments in trials.items():
                if len(segments) != expected_blocks:
                    print(f"Subject {subject}, Hand {hand}, Trial {trial} has {len(segments)} blocks instead of {expected_blocks}.")
        
        if total_trials != expected_trials * expected_hands:
            print(f"Subject {subject} has {total_trials} trials instead of {expected_trials * expected_hands}.")
            missing_data_subjects.append(subject)

    return missing_data_subjects

# Find subjects with missing data
missing_subjects = find_missing_data(iBBT_reach_speed_segments)
print(f"Subjects with missing data: {missing_subjects}")



def calculate_and_swap_total_time_per_trial(iBBT_reach_speed_segments, all_dates):
    """
    Calculate the total time taken for each subject per trial for iBBT and swap left/right total time results 
    for specific subjects, renaming keys as 'non_dominant' and 'dominant'.

    Args:
        iBBT_reach_speed_segments (dict): Dictionary containing reach speed segments for each subject and trial.
        all_dates (list): List of all subject dates.

    Returns:
        dict: Modified total time results with swapped and renamed keys.
    """
    total_time_per_trial = {}

    # Calculate total time per trial
    for subject, hands in iBBT_reach_speed_segments.items():
        total_time_per_trial[subject] = {}
        for hand, trials in hands.items():
            total_time_per_trial[subject][hand] = {}
            for trial, segments in trials.items():
                if segments:  # Ensure there are segments
                    start_time = segments[0][0]  # Start time of the first reach
                    end_time = segments[-1][1]  # End time of the last reach
                    total_time = end_time - start_time
                    total_time_per_trial[subject][hand][trial] = total_time / 200  # Convert to seconds assuming fs=200 Hz

    # Subjects for which left/right metrics should be swapped
    subjects_to_swap = {all_dates[20], all_dates[22]}

    # Swap and rename keys
    modified_results = {}
    for subj, hands in total_time_per_trial.items():
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
iBBT_total_time_results = calculate_and_swap_total_time_per_trial(iBBT_reach_speed_segments, All_dates)

def calculate_average_total_time(total_time_results):
    """
    Calculate the average total time across all trials for dominant and non-dominant hands.

    Args:
        total_time_results (dict): Dictionary containing total time results for each participant and hand.

    Returns:
        dict: Dictionary with average total time for each participant and hand.
    """
    average_total_time_results = {}

    for participant, hands in total_time_results.items():
        average_total_time_results[participant] = {}
        for hand, trials in hands.items():
            if trials:  # Ensure there are trials
                average_total_time = sum(trials.values()) / len(trials)
                average_total_time_results[participant][hand] = average_total_time

    return average_total_time_results

# Calculate average total time for iBBT
iBBT_average_total_time_results = calculate_average_total_time(iBBT_total_time_results)



def calculate_icc_for_hands(total_time_results):
    """
    Calculate ICC(3,1) for each hand (dominant and non-dominant) across 4 repeat tests.

    Args:
        total_time_results (dict): Dictionary containing total time results for each participant and hand.

    Returns:
        dict: Dictionary with ICC results for each hand.
    """
    icc_results = {}

    for hand in ['dominant', 'non_dominant']:
        # Prepare data for ICC calculation
        data = []
        for participant, hands in total_time_results.items():
            if hand in hands:
                for trial_idx, (trial, value) in enumerate(hands[hand].items(), start=1):
                    data.append({'Participant': participant, 'Trial': trial_idx, 'Value': value})

        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(df)
        # Calculate ICC(3,1)
        icc = intraclass_corr(data=df, targets='Participant', raters='Trial', ratings='Value')
        icc_results[hand] = icc[icc['Type'] == 'ICC3'].iloc[0]['ICC']

    return icc_results

# Calculate ICC for iBBT total time results
icc_results = calculate_icc_for_hands(iBBT_total_time_results)
print("ICC results for iBBT total time results:", icc_results)


import pandas as pd
from pingouin import intraclass_corr


def calculate_icc_and_pairwise_correlations(total_time_results):
    """
    Calculate ICC(3,1) and pairwise Pearson correlations across 4 repeated tests
    for dominant and non-dominant hands.

    Args:
        total_time_results (dict): Dictionary containing total time results
                                   for each participant and hand.

    Returns:
        dict: ICC and pairwise correlation results for each hand.
    """

    results = {}

    for hand in ['dominant', 'non_dominant']:

        # -------------------------------
        # Prepare long-format data
        # -------------------------------
        data = []

        for participant, hands in total_time_results.items():
            if hand in hands:
                # Ensure trials are ordered
                for trial_idx, trial in enumerate(sorted(hands[hand]), start=1):
                    data.append({
                        'Participant': participant,
                        'Trial': trial_idx,
                        'Value': hands[hand][trial]
                    })

        df_long = pd.DataFrame(data)

        # -------------------------------
        # Calculate ICC(3,1)
        # -------------------------------
        icc_table = intraclass_corr(
            data=df_long,
            targets='Participant',
            raters='Trial',
            ratings='Value'
        )

        icc_row = icc_table[icc_table['Type'] == 'ICC3'].iloc[0]

        icc_value = icc_row['ICC']
        icc_ci_low, icc_ci_high = icc_row['CI95%']

        # -------------------------------
        # Pairwise Pearson correlations
        # -------------------------------
        df_wide = df_long.pivot(
            index='Participant',
            columns='Trial',
            values='Value'
        )

        pairwise_corr = df_wide.corr(method='pearson')

        # -------------------------------
        # Store results
        # -------------------------------
        results[hand] = {
            'ICC_3_1': icc_value,
            'ICC_95CI': (icc_ci_low, icc_ci_high),
            'Pairwise_Correlations': pairwise_corr
        }

    return results


# -----------------------------------
# Run analysis
# -----------------------------------
icc_results = calculate_icc_and_pairwise_correlations(iBBT_total_time_results)

# Print results
for hand, res in icc_results.items():
    print(f"\n{hand.upper()} HAND")
    print(f"ICC(3,1): {res['ICC_3_1']:.3f}")
    print(f"95% CI: [{res['ICC_95CI'][0]:.3f}, {res['ICC_95CI'][1]:.3f}]")
    print("Pairwise correlations:")
    print(res['Pairwise_Correlations'])


# Calculate the mean of pairwise correlations for each hand, excluding self-correlations
mean_pairwise_correlations = {
    hand: res['Pairwise_Correlations'].where(~np.eye(res['Pairwise_Correlations'].shape[0], dtype=bool)).mean().mean()
    for hand, res in icc_results.items()
}

# Print the mean pairwise correlations
for hand, mean_corr in mean_pairwise_correlations.items():
    print(f"Mean pairwise correlation for {hand} hand (excluding self-correlations): {mean_corr:.3f}")

# 0.811761+0.678525+0.592318+0.778856+0.783951+0.92261 = 4.567021 /6 = 0.7611701666666667
# 0.849556+0.787001+0.803959+0.922532+0.914835+0.935707 = 5.21359 /6 = 0.8689316666666667



# Extract first test and second test values for each participant and each hand
def extract_first_second_test_values(total_time_results):
    """
    Extract the first test and second test values for each participant and each hand.

    Args:
        total_time_results (dict): Dictionary containing total time results for each participant and hand.

    Returns:
        dict: Dictionary with first and second test values for each participant and hand.
    """
    first_second_test_values = {}

    for participant, hands in total_time_results.items():
        first_second_test_values[participant] = {}
        for hand, trials in hands.items():
            sorted_trials = sorted(trials.items())  # Ensure trials are sorted by trial number
            if len(sorted_trials) >= 2:  # Ensure there are at least two trials
                first_test = sorted_trials[0][1]
                second_test = sorted_trials[1][1]
                first_second_test_values[participant][hand] = {
                    'first_test': first_test,
                    'second_test': second_test
                }

    return first_second_test_values

# Extract first and second test values for iBBT
iBBT_first_second_test_values = extract_first_second_test_values(iBBT_total_time_results)

# Apply ICC(3,1) on iBBT_first_second_test_values
def calculate_icc_for_iBBT_first_second_test(iBBT_first_second_test_values):
    """
    Calculate ICC(3,1) for the first and second test values for each hand.

    Args:
        iBBT_first_second_test_values (dict): Dictionary containing first and second test values for each participant and hand.

    Returns:
        dict: Dictionary with ICC results for each hand.
    """
    icc_results = {}

    for hand in ['dominant', 'non_dominant']:
        # Prepare data for ICC calculation
        data = []
        for participant, hands in iBBT_first_second_test_values.items():
            if hand in hands:
                data.append({'Participant': participant, 'Test': 1, 'Value': hands[hand]['first_test']})
                data.append({'Participant': participant, 'Test': 2, 'Value': hands[hand]['second_test']})

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Calculate ICC(3,1)
        icc = intraclass_corr(data=df, targets='Participant', raters='Test', ratings='Value')
        icc_results[hand] = icc[icc['Type'] == 'ICC3'].iloc[0]['ICC']

    return icc_results

# Calculate ICC for iBBT first and second test values
icc_iBBT_results = calculate_icc_for_iBBT_first_second_test(iBBT_first_second_test_values)
print("ICC results for iBBT first and second test values:", icc_iBBT_results)

## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## tBBT data
# --- tBBT - PROCESS ALL DATE AND SAVE ALL MOVEMENT DATA AS pickle file ---
prominence_threshold_speed = 400
prominence_threshold_position = 80
# utils1.process_all_dates_separate(All_dates, Traj_folder, Box_Traj_folder, Figure_folder, DataProcess_folder, 
#                       prominence_threshold_speed, prominence_threshold_position)

tBBT_results = utils1.load_selected_subject_results(All_dates, DataProcess_folder)
tBBT_reach_speed_segments = utils2.get_reach_speed_segments(tBBT_results)
tBBT_reach_metrics = utils2.calculate_reach_metrics(tBBT_reach_speed_segments, tBBT_results, fs=200)
tBBT_test_windows_7 = utils9.compute_test_window_7(tBBT_results, tBBT_reach_speed_segments, tBBT_reach_metrics)

tBBT_total_time_results = calculate_and_swap_total_time_per_trial(tBBT_reach_speed_segments, All_dates)






# Extract first test and second test values for each participant and each hand
def extract_first_second_test_values(total_time_results):
    """
    Extract the first test and second test values for each participant and each hand.

    Args:
        total_time_results (dict): Dictionary containing total time results for each participant and hand.

    Returns:
        dict: Dictionary with first and second test values for each participant and hand.
    """
    first_second_test_values = {}

    for participant, hands in total_time_results.items():
        first_second_test_values[participant] = {}
        for hand, trials in hands.items():
            sorted_trials = sorted(trials.items())  # Ensure trials are sorted by trial number
            if len(sorted_trials) >= 2:  # Ensure there are at least two trials
                first_test = sorted_trials[0][1]
                second_test = sorted_trials[1][1]
                first_second_test_values[participant][hand] = {
                    'first_test': first_test,
                    'second_test': second_test
                }

    return first_second_test_values

# Extract first and second test values for tBBT
tBBT_first_second_test_values = extract_first_second_test_values(tBBT_total_time_results)

# Apply ICC(3,1) on tBBT_first_second_test_values
def calculate_icc_for_tBBT_first_second_test(tBBT_first_second_test_values):
    """
    Calculate ICC(3,1) for the first and second test values for each hand.

    Args:
        tBBT_first_second_test_values (dict): Dictionary containing first and second test values for each participant and hand.

    Returns:
        dict: Dictionary with ICC results for each hand.
    """
    icc_results = {}

    for hand in ['dominant', 'non_dominant']:
        # Prepare data for ICC calculation
        data = []
        for participant, hands in tBBT_first_second_test_values.items():
            if hand in hands:
                data.append({'Participant': participant, 'Test': 1, 'Value': hands[hand]['first_test']})
                data.append({'Participant': participant, 'Test': 2, 'Value': hands[hand]['second_test']})

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Calculate ICC(3,1)
        icc = intraclass_corr(data=df, targets='Participant', raters='Test', ratings='Value')
        icc_results[hand] = icc[icc['Type'] == 'ICC3'].iloc[0]['ICC']

    return icc_results

# Calculate ICC for tBBT first and second test values
icc_tBBT_results = calculate_icc_for_tBBT_first_second_test(tBBT_first_second_test_values)
print("ICC results for tBBT first and second test values:", icc_tBBT_results)




# Calculate average total time across all trials for dominant and non-dominant hands
tBBT_average_total_time_results = calculate_average_total_time(tBBT_total_time_results)

# Total number of valid (non-NaN) datapoints in Block_Distance: 29809
Block_Distance = utils4.load_selected_subject_errors(All_dates, DataProcess_folder)

# Count the total and valid (non-NaN) data points in tBBT_reach_metrics['reach_durations']
def count_total_and_valid_data_points(reach_metrics):
    """
    Count the total and valid (non-NaN) data points in reach_metrics['reach_durations'].

    Args:
        reach_metrics (dict): Dictionary containing reach metrics data.

    Returns:
        tuple: Total data points, valid data points.
    """
    total_count = 0
    valid_count = 0

    for subject, hands in reach_metrics['reach_durations'].items():
        for hand, trials in hands.items():
            for trial, durations in trials.items():
                total_count += len(durations)
                valid_count += sum(1 for d in durations if not np.isnan(d))

    return total_count, valid_count

# Call the function to count total and valid data points in tBBT_reach_metrics['reach_durations']
total_data_points, valid_data_points = count_total_and_valid_data_points(tBBT_reach_metrics)
print(f"Total data points in tBBT_reach_metrics['reach_durations']: {total_data_points}")
print(f"Valid data points in tBBT_reach_metrics['reach_durations']: {valid_data_points}")

# Find None values in Block_Distance and analyze their distribution
def find_none_in_block_distance(Block_Distance):
    """
    Find None values in Block_Distance and analyze their distribution.

    Args:
        Block_Distance (dict): Dictionary containing block distance data.

    Returns:
        dict: Summary of None values for each subject and hand.
    """
    none_summary = {}
    total_none_count = 0

    for subject, hands in Block_Distance.items():
        none_summary[subject] = {}
        for hand, trials in hands.items():
            none_count = 0
            for trial, blocks in trials.items():
                none_count += sum(1 for block in blocks if block is None)
            if none_count > 0:
                none_summary[subject][hand] = none_count
                total_none_count += none_count

    return none_summary, total_none_count

# Analyze None values in Block_Distance
none_summary, total_none_count = find_none_in_block_distance(Block_Distance)

# Print the results
print(f"Total None values in Block_Distance: {total_none_count}")
for subject, hands in none_summary.items():
    for hand, count in hands.items():
        print(f"Subject {subject}, Hand {hand}: {count} None values")



# Check if each subject, each hand, and each trial in tBBT_reach_metrics['reach_durations'] has 16 valid (non-NaN) data points
def check_reach_durations_data_points(reach_metrics):
    """
    Check if each subject, each hand, and each trial in reach_metrics['reach_durations'] has 16 valid (non-NaN) data points.

    Args:
        reach_metrics (dict): Dictionary containing reach metrics data.

    Returns:
        None
    """
    for subject, hands in reach_metrics['reach_durations'].items():
        for hand, trials in hands.items():
            for trial, durations in trials.items():
                valid_durations = [d for d in durations if not np.isnan(d)]
                if len(valid_durations) != 16:
                    print(f"Subject {subject}, Hand {hand}, Trial {trial} has {len(valid_durations)} valid data points instead of 16.")

# Call the function to check tBBT_reach_metrics['reach_durations']
check_reach_durations_data_points(tBBT_reach_metrics)

# Update and swap Block_Distance
def update_and_swap_block_distance(Block_Distance, reach_metrics, all_dates):
    """
    Updates the keys of Block_Distance to match the filenames in reach_metrics for each subject and hand,
    and swaps left/right metrics for specific subjects.

    Args:
        Block_Distance (dict): Dictionary containing block distance data.
        reach_metrics (dict): Dictionary containing reach metrics data.
        all_dates (list): List of all subject dates.

    Returns:
        dict: Updated and swapped Block_Distance.
    """
    # Update Block_Distance keys to match filenames in reach_metrics
    for subject in Block_Distance:
        for hand in Block_Distance[subject]:
            if subject in reach_metrics['reach_durations'] and hand in reach_metrics['reach_durations'][subject] and \
               len(Block_Distance[subject][hand]) == len(reach_metrics['reach_durations'][subject][hand]):

                filenames = list(reach_metrics['reach_durations'][subject][hand].keys())  # Get filenames in a list

                if len(filenames) != len(Block_Distance[subject][hand]):
                    print(f"Error: Mismatch in lengths for subject {subject}, hand {hand}.")
                    print(f"Filenames length: {len(filenames)}, Block_Distance length: {len(Block_Distance[subject][hand])}")
                else:
                    # Update Block_Distance keys to match filenames
                    updated_distance = {filenames[i]: v for i, v in enumerate(Block_Distance[subject][hand].values())}
                    Block_Distance[subject][hand] = updated_distance

    # Subjects for which left/right metrics should be swapped
    subjects_to_swap = {all_dates[20], all_dates[22]}

    # Swap and rename keys for Block_Distance
    modified_block_distance = {}
    for subj, hands in Block_Distance.items():
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
        modified_block_distance[subj] = swapped_hands

    return modified_block_distance

Block_Distance = update_and_swap_block_distance(Block_Distance, tBBT_reach_metrics, All_dates)

# # Plot histogram for Block_Distance data overlaid for both hands
# def plot_histogram_block_distance_overlay(Block_Distance):
#     """
#     Plot a single histogram overlaying block distances for both hands (dominant and non-dominant) across all subjects and trials.
#     Calculate and display the count, average, and standard deviation for each hand on the plot.

#     Args:
#         Block_Distance (dict): Dictionary containing block distance data.
#     """
#     distances_by_hand = {'dominant': [], 'non_dominant': []}

#     # Collect block distances separated by hand
#     for subject, hands in Block_Distance.items():
#         for hand, trials in hands.items():
#             for trial, blocks in trials.items():
#                 for distance in blocks:
#                     if isinstance(distance, (int, float)) and not np.isnan(distance):
#                         distances_by_hand[hand].append(distance)

#     # Calculate statistics for each hand
#     stats = {}
#     for hand in distances_by_hand:
#         stats[hand] = {
#             'count': len(distances_by_hand[hand]),
#             'mean': np.nanmean(distances_by_hand[hand]),  # Use np.nanmean to ignore NaN values
#             'std': np.nanstd(distances_by_hand[hand])    # Use np.nanstd to ignore NaN values
#         }

#     # Plot overlaid histograms for both hands
#     plt.figure(figsize=(10, 6))
#     plt.hist(
#         distances_by_hand['dominant'], bins=50, color='blue', alpha=0.7, edgecolor='black', label='Dominant Hand'
#     )
#     plt.hist(
#         distances_by_hand['non_dominant'], bins=50, color='green', alpha=0.7, edgecolor='black', label='Non-Dominant Hand'
#     )
#     plt.xlabel('Block Distance', fontsize=14)
#     plt.ylabel('Frequency', fontsize=14)

#     # Add statistics to the plot
#     for i, hand in enumerate(['dominant', 'non_dominant']):
#         plt.text(
#             0.95, 0.85 - i * 0.3,
#             f"{hand.capitalize()}:\nCount: {stats[hand]['count']}\nMean: {stats[hand]['mean']:.2f}\nSD: {stats[hand]['std']:.2f}",
#             transform=plt.gca().transAxes,
#             fontsize=12,
#             color='blue' if hand == 'dominant' else 'green',
#             ha='right'
#         )

#     plt.legend(fontsize=12)
#     plt.grid(False)
#     sns.despine()
#     plt.show()

# # Call the function to plot the overlaid histogram
# plot_histogram_block_distance_overlay(Block_Distance)

def calculate_average_block_distance_per_location(Block_Distance):
    """
    Calculate the average block distance per subject, per hand, and per location.

    Args:
        Block_Distance (dict): Dictionary containing block distance data.

    Returns:
        dict: Dictionary with average block distance per subject, hand, and location.
    """
    average_block_distance_per_location = {}

    for subject, hands in Block_Distance.items():
        average_block_distance_per_location[subject] = {}
        for hand, trials in hands.items():
            location_distances = {loc: [] for loc in range(16)}  # Initialize distances for 16 locations
            for trial, blocks in trials.items():
                for loc, distance in enumerate(blocks):
                    if isinstance(distance, (int, float)) and not np.isnan(distance):
                        location_distances[loc].append(distance)

            # Calculate average distance for each location
            average_block_distance_per_location[subject][hand] = {
                loc: np.mean(distances) if distances else np.nan
                for loc, distances in location_distances.items()
            }

    return average_block_distance_per_location

# Calculate the average block distance per subject, per hand, and per location
average_block_distance_per_location = calculate_average_block_distance_per_location(Block_Distance)



def plot_average_block_distance_heatmap(average_block_distance_per_location):
    """
    Plot heatmaps for average block distance per location for paired hands side by side in a single figure,
    averaged across all participants, with a more distinct color difference.
    """
    metric = 'average_block_distance'
    title = 'Average Block Distance'

    # Define custom layout mappings
    non_dominant_layout = [
        [12, 13, 14, 15],  # top row
        [8, 9, 10, 11],
        [4, 5, 6, 7],
        [0, 1, 2, 3]       # bottom row
    ]

    dominant_layout = [
        [15, 14, 13, 12],      # mirrored top row
        [11, 10, 9, 8],
        [7, 6, 5, 4],
        [3, 2, 1, 0]   # mirrored bottom row
    ]

    # Compute average block distance across participants for each location
    averaged_data = {hand: {loc: [] for loc in range(16)} for hand in ['non_dominant', 'dominant']}
    for subject, hands in average_block_distance_per_location.items():
        for hand, locations in hands.items():
            for loc, distance in locations.items():
                if not np.isnan(distance):
                    averaged_data[hand][loc].append(distance)

    for hand in averaged_data:
        for loc in averaged_data[hand]:
            distances = averaged_data[hand][loc]
            averaged_data[hand][loc] = np.mean(distances) if distances else np.nan

    # Collect data for both hands to determine the shared color bar range
    all_data = []
    for hand in ['dominant', 'non_dominant']:
        layout = non_dominant_layout if hand == 'non_dominant' else dominant_layout
        for row in range(4):
            for col in range(4):
                loc = layout[row][col]
                if loc in averaged_data[hand]:
                    all_data.append(averaged_data[hand][loc])

    vmin, vmax = np.nanmin(all_data), np.nanmax(all_data)

    # Use a more distinct colormap
    cmap = sns.color_palette("coolwarm", as_cmap=True)

    # Heatmap for 16 locations
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for j, hand in enumerate(['dominant', 'non_dominant']):  # Dominant hand in column 0, non-dominant hand in column 1
        data = np.full((4, 4), np.nan)

        layout = non_dominant_layout if hand == 'non_dominant' else dominant_layout

        for row in range(4):
            for col in range(4):
                loc = layout[row][col]
                if loc in averaged_data[hand]:
                    data[row, col] = averaged_data[hand][loc]

        sns.heatmap(
            data, annot=True, fmt=".2f",
            cmap=cmap, 
            ax=axes[j], cbar=(j == 1), cbar_ax=None if j == 0 else fig.add_axes([0.92, 0.3, 0.02, 0.4]),
            vmin=vmin, vmax=vmax, annot_kws={"fontsize": 18}, square=True
        )

        axes[j].set_title(f"{hand.replace('_', ' ').capitalize()} Hand", fontsize=18)
        axes[j].set_xticks([])
        axes[j].set_yticks([])

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()

    # Heatmap for averaged block distance across 16 locations
    avg_data = []
    for hand in ['dominant', 'non_dominant']:
        avg_distance = np.nanmean([averaged_data[hand][loc] for loc in range(16)])
        avg_data.append(avg_distance)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        np.array(avg_data).reshape(1, 2), annot=True, fmt=".2f",
        cmap=cmap, cbar=True, xticklabels=['Dominant Hand', 'Non-Dominant Hand'], yticklabels=['Average'],
        vmin=vmin, vmax=vmax, annot_kws={"fontsize": 18}, square=True, ax=ax
    )
    ax.set_title(f"Averaged {title} Across 16 Locations", fontsize=18)
    plt.tight_layout()
    plt.show()

plot_average_block_distance_heatmap(average_block_distance_per_location)








def calculate_average_block_distance_per_person_per_hand(average_block_distance_per_location):
    """
    Calculate the average block distance per person and per hand across all locations.

    Args:
        average_block_distance_per_location (dict): Dictionary containing average block distance per subject, hand, and location.

    Returns:
        dict: Dictionary with average block distance per person and per hand.
    """
    average_per_person_per_hand = {}

    for subject, hands in average_block_distance_per_location.items():
        average_per_person_per_hand[subject] = {}
        for hand, locations in hands.items():
            valid_distances = [distance for distance in locations.values() if not np.isnan(distance)]
            if valid_distances:
                average_per_person_per_hand[subject][hand] = np.mean(valid_distances)
            else:
                average_per_person_per_hand[subject][hand] = np.nan

    return average_per_person_per_hand

# Calculate the average block distance per person and per hand
average_block_distance_per_person_per_hand = calculate_average_block_distance_per_person_per_hand(average_block_distance_per_location)



cmap_choice = LinearSegmentedColormap.from_list("WhiteGreenBlue", ["white", "green", "blue"], N=256)

# # Plot heatmap for average block distance per location, overlaying participants
# def plot_combined_average_block_distance_heatmap(average_block_distance_per_location, cmap_choice):
#     """
#     Plot a single heatmap for average block distance per location for both hands, overlaying participants' data.
#     Annotate the average block distance for each hand on the plot.

#     Args:
#         average_block_distance_per_location (dict): Dictionary containing average block distance per subject, hand, and location.
#         cmap_choice: Colormap to use for the heatmap.
#     """
#     # Initialize data storage for each location
#     location_data = {hand: {loc: [] for loc in range(16)} for hand in ['non_dominant', 'dominant']}

#     # Collect data for each location across all participants
#     for subject, hands in average_block_distance_per_location.items():
#         for hand, locations in hands.items():
#             for loc, distance in locations.items():
#                 if not np.isnan(distance):
#                     location_data[hand][loc].append(distance)

#     # Define custom layout mappings
#     left_layout = [
#         [12, 13, 14, 15],  # top row
#         [8, 9, 10, 11],
#         [4, 5, 6, 7],
#         [0, 1, 2, 3]       # bottom row
#     ]
    
#     right_layout = [
#         [15, 14, 13, 12],      # mirrored top row
#         [11, 10, 9, 8],
#         [7, 6, 5, 4],
#         [3, 2, 1, 0]   # mirrored bottom row
#     ]

#     # Prepare data for combined heatmap
#     combined_data = np.full((4, 8), np.nan)  # 4 rows, 8 columns (left + right)
#     for row in range(4):
#         for col in range(4):
#             left_loc = left_layout[row][col]
#             right_loc = right_layout[row][col]
#             if left_loc in location_data['non_dominant']:
#                 combined_data[row, col + 4] = np.mean(location_data['non_dominant'][left_loc]) if location_data['non_dominant'][left_loc] else np.nan
#             if right_loc in location_data['dominant']:
#                 combined_data[row, col] = np.mean(location_data['dominant'][right_loc]) if location_data['dominant'][right_loc] else np.nan

#     # Calculate overall averages for each hand
#     avg_non_dominant = np.nanmean([np.mean(location_data['non_dominant'][loc]) for loc in range(16) if location_data['non_dominant'][loc]])
#     avg_dominant = np.nanmean([np.mean(location_data['dominant'][loc]) for loc in range(16) if location_data['dominant'][loc]])

#     # Plot combined heatmap
#     fig, ax = plt.subplots(figsize=(12, 6))
#     sns.heatmap(
#         combined_data, annot=True, fmt=".2f", cmap="viridis_r", square=True, ax=ax,
#         cbar=True, cbar_kws={'label': 'Average Block Distance'}
#     )
#     ax.set_xticks([])
#     ax.set_yticks([])

#     # Annotate overall averages for each hand
#     ax.text(2, -0.5, f"Dominant Avg: {avg_dominant:.4f}", fontsize=16, color='blue', ha='center')
#     ax.text(6, -0.5, f"Non-Dominant Avg: {avg_non_dominant:.4f}", fontsize=16, color='green', ha='center')

#     plt.tight_layout()
#     plt.show()

# # Call the function to plot the combined heatmap
# plot_combined_average_block_distance_heatmap(average_block_distance_per_location, cmap_choice)



# Calculate the avergae block distance per trial for each subject, each hand, and each trial by adding up the distances of all 16 blocks and deviding by 16, if nah exists, devide by the number of existing blocks
def calculate_average_block_distance_per_trial(Block_Distance):
    """
    Calculate the average block distance per trial for each subject, each hand, and each trial.

    Args:
        Block_Distance (dict): Dictionary containing block distance data.

    Returns:
        dict: Dictionary with average block distance per trial for each subject and hand.
    """
    average_block_distance = {}

    for subject, hands in Block_Distance.items():
        average_block_distance[subject] = {}
        for hand, trials in hands.items():
            average_block_distance[subject][hand] = {}
            for trial, blocks in trials.items():
                if blocks:  # Ensure there are blocks
                    valid_blocks = [distance for distance in blocks if isinstance(distance, (int, float)) and not np.isnan(distance)]
                    if valid_blocks:  # Ensure there are valid blocks
                        average_distance = sum(valid_blocks) / len(valid_blocks)
                        average_block_distance[subject][hand][trial] = average_distance
                    else:
                        average_block_distance[subject][hand][trial] = np.nan
                else:
                    average_block_distance[subject][hand][trial] = np.nan

    return average_block_distance

# Calculate the average block distance per trial
tBBT_average_block_distance_per_trial = calculate_average_block_distance_per_trial(Block_Distance)
tBBT_average_block_distance = calculate_average_total_time(tBBT_average_block_distance_per_trial)


def process_and_modify_tBBTs_errors(pickle_file, all_dates):
    """
    Load and process tBBTs errors from a pickle file, swap left/right metrics for specific subjects,
    and apply transformations to the data.

    Args:
        pickle_file (str): Path to the pickle file containing tBBTs errors.
        all_dates (list): List of all subject dates.

    Returns:
        dict: Modified tBBTs errors.
    """
    # Load the pickle file
    with open(pickle_file, "rb") as file:
        all_subject_tBBTs_errors = pickle.load(file)

    # Subjects for which left/right metrics should be swapped
    subjects_to_swap = {all_dates[20], all_dates[22]}

    # Swap and rename keys for all_subject_tBBTs_errors
    modified_tBBTs_errors = {}
    for (subj, hand), trials in all_subject_tBBTs_errors.items():
        if subj in subjects_to_swap:
            new_hand = 'non_dominant' if hand == 'right' else 'dominant'
        else:
            new_hand = 'dominant' if hand == 'right' else 'non_dominant'

        if (subj, new_hand) not in modified_tBBTs_errors:
            modified_tBBTs_errors[(subj, new_hand)] = {}

        modified_tBBTs_errors[(subj, new_hand)].update(trials)

    # Apply adjustments to x values in modified_tBBTs_errors for specific subjects based on membership and hand
    for (subj, hand), trials in modified_tBBTs_errors.items():
        if subj in subjects_to_swap:
            for trial, trial_data in trials.items():
                if 'block_errors' in trial_data:
                    trial_data['block_errors'] = [
                        {
                            'point': np.array([
                                p['point'][0] - 95.71428571 if p['membership'] in [0, 4, 8, 12] and hand == 'dominant' else
                                p['point'][0] + 95.71428571 if p['membership'] in [0, 4, 8, 12] and hand == 'non_dominant' else
                                p['point'][0] - 204.57142857 if p['membership'] in [1, 5, 9, 13] and hand == 'dominant' else
                                p['point'][0] + 204.57142857 if p['membership'] in [1, 5, 9, 13] and hand == 'non_dominant' else
                                p['point'][0] - 313.42857143 if p['membership'] in [2, 6, 10, 14] and hand == 'dominant' else
                                p['point'][0] + 313.42857143 if p['membership'] in [2, 6, 10, 14] and hand == 'non_dominant' else
                                p['point'][0] - 422.28571429 if p['membership'] in [3, 7, 11, 15] and hand == 'dominant' else
                                p['point'][0] + 422.28571429 if p['membership'] in [3, 7, 11, 15] and hand == 'non_dominant' else
                                p['point'][0],  # Default case if no condition matches
                                p['point'][1],
                                p['point'][2]
                            ]),
                            'membership': p['membership'],
                            'distance': p['distance']
                        }
                        for p in trial_data['block_errors']
                    ]

    return modified_tBBTs_errors

pickle_file = "/Users/yilinwu/Desktop/honours data/DataProcess/All_Subject_tBBTs_errors.pkl"
All_Subject_tBBTs_errors = process_and_modify_tBBTs_errors(pickle_file, All_dates)

def plot_points_for_trial(errors, subject, hand, trial):
    """
    Plot the points from the specified trial for a given subject and hand.

    Args:
        errors (dict): Dictionary containing error data.
        subject (str): Subject identifier.
        hand (str): Hand identifier ('dominant' or 'non_dominant').
        trial (int): Trial number.

    Returns:
        None
    """
    # Extract points for the specified trial
    points = errors[(subject, hand)][trial]['block_errors']

    # Extract x, y coordinates and membership labels
    x_coords = [point['point'][0] for point in points]
    y_coords = [point['point'][1] for point in points]
    memberships = [point['membership'] for point in points]

    # Plot the points
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_coords, y_coords, c=memberships, cmap='viridis', s=100, edgecolor='black')

    # Add labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f"2D Plot of Points for ({subject}, {hand}) Trial {trial}")
    plt.colorbar(scatter, label='Membership')
    plt.grid(True)

    # Annotate each point with its membership label
    for i, (x, y, membership) in enumerate(zip(x_coords, y_coords, memberships)):
        plt.text(x, y, str(membership), fontsize=9, ha='right', va='bottom')

    plt.show()

# Example usage
plot_points_for_trial(All_Subject_tBBTs_errors, '08/02/PY', 'non_dominant', 1)
plot_points_for_trial(All_Subject_tBBTs_errors, '07/22/HW', 'non_dominant', 1)









## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Figure 1 - Correlation between dominant and non-dominant hand performance for sBBT, iBBT, and tBBT
def plot_correlation(data_x, data_y, x_label, y_label, title, plot=True, x_range=None, y_range=None):
    """
    Calculate correlation and plot the relationship between two datasets.

    Args:
        data_x (list or array): Data for the x-axis.
        data_y (list or array): Data for the y-axis.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.
        plot (bool): Whether to display the plot. Default is True.
        x_range (tuple): Tuple specifying the x-axis range as (min, max). Default is None.
        y_range (tuple): Tuple specifying the y-axis range as (min, max). Default is None.

    Returns:
        tuple: Correlation coefficient and p-value.
    """
    # Calculate correlation and p-value
    corr, p_value = pearsonr(data_x, data_y)

    if plot:
        # Plot correlation
        plt.figure(figsize=(6, 6))
        sns.regplot(x=data_x, y=data_y, scatter_kws={'s': 50}, line_kws={'color': 'red'})
        plt.xlabel(x_label, fontsize=18)
        plt.ylabel(y_label, fontsize=18)

        plt.xticks(fontsize=16)

        # Determine x-axis integer ticks
        if x_range:
            plt.xlim(x_range)
            x_min, x_max = int(np.floor(x_range[0])), int(np.ceil(x_range[1]))
        else:
            x_min, x_max = int(np.floor(min(data_x))), int(np.ceil(max(data_x)))
        ax = plt.gca()  # Get the current axis
        ax.set_xticks(range(x_min, x_max + 1))  # only integer ticks

        plt.yticks(fontsize=16)
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))

        # Set x and y axis ranges if provided
        if x_range:
            plt.xlim(x_range)
        if y_range:
            plt.ylim(y_range)

        # Plot mean points as black
        mean_x = np.mean(data_x)
        mean_y = np.mean(data_y)
        # plt.scatter(mean_x, mean_y, color='black', s=100, label='Mean Point', zorder=5)
        plt.scatter(mean_x, mean_y, color='red',edgecolors='black', s=150, marker='s', label='Mean Point', zorder=5)

        # # Plot line of unity
        # min_val = min(min(data_x), min(data_y))
        # max_val = max(max(data_x), max(data_y))
        # plt.plot([min_val, max_val], [min_val, max_val], color='blue', linestyle='--', label='Line of Unity')

        sns.despine()

        # Annotate correlation and p-value at the top left corner
        # plt.annotate(f'r = {corr:.2f}, p = {p_value:.3f} \n n = {len(data_x)}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=18, ha='left', va='top')
        # plt.annotate(f'r = {corr:.2f}, p = {p_value:.3e} \n n = {len(data_x)}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=18, ha='left', va='top')
        plt.annotate(f'r = {corr:.2f}\n n = {len(data_x)}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=18, ha='left', va='top')

        plt.show()

    return corr, p_value

# sBBT dominant vs non-dominant scores
plot_correlation(
    sBBTResult['dominant'], 
    sBBTResult['non_dominant'], 
    'Dominant hand \n sBBT scores (no. of blocks)',   
    'Non-dominant hand \n sBBT scores (no. of blocks)', 
    'sBBT Dominant vs Non-dominant',
    x_range=(40, 90),  # Example x-axis range
    y_range=(40, 90)   # Example y-axis range
)

# iBBT dominant vs non-dominant average total times
plot_correlation(
    [hands['dominant'] for hands in iBBT_average_total_time_results.values() if 'dominant' in hands],
    [hands['non_dominant'] for hands in iBBT_average_total_time_results.values() if 'non_dominant' in hands],
    'Dominant hand \n iBBT average total time (s)',
    'Non-dominant hand \n iBBT average total time (s)',
    'iBBT Dominant vs Non-dominant',
    x_range=(20, 40),  # Example x-axis range
    y_range=(20, 40)   # Example y-axis range
)

# tBBT dominant vs non-dominant average total times
plot_correlation(
    [hands['dominant'] for hands in tBBT_average_total_time_results.values() if 'dominant' in hands],
    [hands['non_dominant'] for hands in tBBT_average_total_time_results.values() if 'non_dominant' in hands],
    'Dominant hand \n tBBT average total time (s)',
    'Non-dominant hand \n tBBT average total time (s)',
    'tBBT Dominant vs Non-dominant',
    x_range=(15, 35),  # Example x-axis range
    y_range=(15, 35)   # Example y-axis range
)




# sBBT: t-statistic = 1.33, p-value = 0.194
# iBBT: t-statistic = -7.36, p-value = 0.000
# tBBT: t-statistic = -8.28, p-value = 0.000










# Figure 2 - Correlation analysis between sBBT scores, iBBT total time, and tBBT total time
def perform_correlation_analysis(sBBT_dominant, sBBT_non_dominant, iBBT_dominant, iBBT_non_dominant, tBBT_dominant, tBBT_non_dominant):
    """
    Perform correlation analysis between sBBT scores, iBBT total time, and tBBT total time.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    titles = [
        ('sBBT vs iBBT', sBBT_dominant, iBBT_dominant, sBBT_non_dominant, iBBT_non_dominant),
        ('sBBT vs tBBT', sBBT_dominant, tBBT_dominant, sBBT_non_dominant, tBBT_non_dominant),
        ('iBBT vs tBBT', iBBT_dominant, tBBT_dominant, iBBT_non_dominant, tBBT_non_dominant)
    ]

    results = []

    for title, x_dom, y_dom, x_non, y_non in titles:
        for x, y, hand in [(x_dom, y_dom, 'Dominant'), (x_non, y_non, 'Non-Dominant')]:
            corr, p_value = pearsonr(x, y)
            results.append({
                'Comparison': title,
                'Hand': hand,
                'Correlation (r)': f'{corr:.2f}',
                'p-value': f'{p_value:.4f}',
                'n': len(x)
            })

    # Convert results to a DataFrame and print as a table
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    for i, (title, x_dom, y_dom, x_non, y_non) in enumerate(titles):
        for j, (x, y, hand) in enumerate([(x_dom, y_dom, 'Dominant'), (x_non, y_non, 'Non-Dominant')]):
            sns.regplot(x=x, y=y, scatter_kws={'s': 50}, line_kws={'color': 'red'}, ax=axes[j, i])
            axes[j, i].set_xlabel(f'{title.split(" vs ")[0]} Scores', fontsize=18)
            axes[j, i].set_ylabel(f'{title.split(" vs ")[1]} Avg Total Time (s)', fontsize=18)
            # axes[j, i].set_title(f'{title} ({hand})', fontsize=14)
            corr, p_value = pearsonr(x, y)
            axes[j, i].annotate(f'r = {corr:.2f}, p = {p_value:.4f} \n n = {len(x_dom)}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=18, ha='left', va='top')
            axes[j, i].grid(False)

    plt.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    sns.despine()

    plt.show()

perform_correlation_analysis(
    sBBTResult['dominant'].tolist(),
    sBBTResult['non_dominant'].tolist(),
    [hands['dominant'] for hands in iBBT_average_total_time_results.values() if 'dominant' in hands],
    [hands['non_dominant'] for hands in iBBT_average_total_time_results.values() if 'non_dominant' in hands],
    [hands['dominant'] for hands in tBBT_average_total_time_results.values() if 'dominant' in hands],
    [hands['non_dominant'] for hands in tBBT_average_total_time_results.values() if 'non_dominant' in hands]
)


def plot_correlation(data_x, data_y, x_label, y_label, title, plot=True, x_range=None, y_range=None):
    corr, p_value = pearsonr(data_x, data_y)

    if plot:
        plt.figure(figsize=(6, 6))
        sns.regplot(x=data_x, y=data_y, scatter_kws={'s': 50}, line_kws={'color': 'red'})
        plt.xlabel(x_label, fontsize=18)
        plt.ylabel(y_label, fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        # Set integer ticks
        ax = plt.gca()
        
        # Determine x-axis integer ticks
        if x_range:
            plt.xlim(x_range)
            x_min, x_max = int(np.floor(x_range[0])), int(np.ceil(x_range[1]))
        else:
            x_min, x_max = int(np.floor(min(data_x))), int(np.ceil(max(data_x)))
        ax.set_xticks(range(x_min, x_max + 1))  # only integer ticks

        # Determine y-axis integer ticks
        if y_range:
            plt.ylim(y_range)
            y_min, y_max = int(np.floor(y_range[0])), int(np.ceil(y_range[1]))
        else:
            y_min, y_max = int(np.floor(min(data_y))), int(np.ceil(max(data_y)))
        ax.set_yticks(range(y_min, y_max + 1))  # only integer ticks

        # Plot mean point
        mean_x, mean_y = np.mean(data_x), np.mean(data_y)
        # plt.scatter(mean_x, mean_y, color='black', s=100, label='Mean Point', zorder=5)
        plt.scatter(mean_x, mean_y, color='red',edgecolors='black', s=150, marker='s', label='Mean Point', zorder=5)

        # # Line of unity
        # min_val = min(min(data_x), min(data_y))
        # max_val = max(max(data_x), max(data_y))
        # plt.plot([min_val, max_val], [min_val, max_val], color='blue', linestyle='--', label='Line of Unity')

        sns.despine()
        # plt.annotate(f'r = {corr:.2f}, p = {p_value:.3e} \n n = {len(data_x)}', 
        #              xy=(0.05, 0.95), xycoords='axes fraction', fontsize=18, ha='left', va='top')
        plt.annotate(f'r = {corr:.2f} \n n = {len(data_x)}', 
                     xy=(0.05, 0.95), xycoords='axes fraction', fontsize=18, ha='left', va='top')

        plt.show()

    return corr, p_value

# Figure 3 - Correlation between dominant and non-dominant hands tBBT average block distance
plot_correlation(
    [hands['dominant'] for hands in tBBT_average_block_distance.values() if 'dominant' in hands],
    [hands['non_dominant'] for hands in tBBT_average_block_distance.values() if 'non_dominant' in hands],
    'Dominant hand \n tBBT average error (mm)',
    'Non-dominant hand \n tBBT average error (mm)',
    'tBBT Dominant vs Non-dominant Block Distance'
)

# Figure 4 - SAT: correlation for each hand between tBBT_average_block_distance and tBBT_average_total_time_results
for hand in ['dominant', 'non_dominant']:
    plot_correlation(
        [tBBT_average_block_distance[participant][hand] for participant in tBBT_average_block_distance if hand in tBBT_average_block_distance[participant]],
        [tBBT_average_total_time_results[participant][hand] for participant in tBBT_average_total_time_results if hand in tBBT_average_total_time_results[participant]],
        f'{hand.capitalize().replace("Non_dominant", "Non-dominant")} hand \n tBBT average error (mm)',
        f'{hand.capitalize().replace("Non_dominant", "Non-dominant")} hand \n tBBT average total time (s)',
        f'{hand.capitalize().replace("Non_dominant", "Non-dominant")} hand: Block Distance vs Total Time'
    )



###----------------------------------------------------------
# Figure 2
### Combine 16 blocks data into one for each subject and hand across all trials
def Combine_16_blocks(All_Subject_tBBTs_errors):
    """
    For each subject and hand in All_Subject_tBBTs_errors, extract the 'p3_block2' data
    and 'blocks_without_points' from all trials, and compute new coordinates based on
    blockMembership and grid adjustments.
    
    For left-hand entries, grid values from grid_xxR and grid_yyR are used.
    For right-hand entries, grid values from grid_xxL and grid_yyL are used.
    
    Returns:
        dict: Mapping from (subject, hand) key to a dictionary of trials, where each trial maps to a list of tuples (new_x, new_y, block)
    """

    # Define grid markers for right hand (for left-hand data adjustment)
    grid_xxR = np.array([[ 12.5      ,  66.92857143, 121.35714286, 175.78571429],
                           [ 12.5      ,  66.92857143, 121.35714286, 175.78571429],
                           [ 12.5      ,  66.92857143, 121.35714286, 175.78571429],
                           [ 12.5      ,  66.92857143, 121.35714286, 175.78571429]])
    grid_yyR = np.array([[ 12.5      ,  12.5      ,  12.5      ,  12.5      ],
                           [ 66.92857143,  66.92857143,  66.92857143,  66.92857143],
                           [121.35714286, 121.35714286, 121.35714286, 121.35714286],
                           [175.78571429, 175.78571429, 175.78571429, 175.78571429]])
    
    # Define grid markers for left hand (for right-hand data adjustment)
    grid_xxL = np.array([[-246.5      , -192.07142857, -137.64285714,  -83.21428571],
                           [-246.5      , -192.07142857, -137.64285714,  -83.21428571],
                           [-246.5      , -192.07142857, -137.64285714,  -83.21428571],
                           [-246.5      , -192.07142857, -137.64285714,  -83.21428571]])
    
    grid_yyL = np.array([[ 12.5      ,  12.5      ,  12.5      ,  12.5      ],
                           [ 66.92857143,  66.92857143,  66.92857143,  66.92857143],
                           [121.35714286, 121.35714286, 121.35714286, 121.35714286],
                           [175.78571429, 175.78571429, 175.78571429, 175.78571429]])
    
    results = {}
    
    # Iterate over every (subject, hand) key in the dictionary
    for key in All_Subject_tBBTs_errors:
        subject, hand = key
        trial_coords = {}  # Dictionary to store the new coordinates for each trial
        
        # Iterate over all trials for the current subject and hand
        for trial_index, trial_entry in All_Subject_tBBTs_errors[key].items():
            # Get block_errors data and blocks_without_points
            points = All_Subject_tBBTs_errors[(subject, hand)][trial_index]['block_errors']

            blocks_without_points = trial_entry.get('blocks_without_points', None)
            if points is None or blocks_without_points is None:
                continue

            # Choose grid arrays based on hand: 'left' uses grid_xxR/yyR, 'right' uses grid_xxL/yyL
            if hand.lower() == 'non_dominant':
                grid_x = (grid_xxR - 12.5).flatten()
                grid_y = (grid_yyR - 12.5).flatten()
            elif hand.lower() == 'dominant':
                grid_x = (grid_xxL - 12.5).flatten()
                grid_x = grid_x[::-1]
                grid_y = (grid_yyL - 12.5).flatten()
            else:
                continue
            memberships = [point['membership'] for point in points]

            trial_data = []  # List to store the new coordinates for this trial
            data_index = 0

            # Loop over each block position (total 16 blocks)
            for i in range(16 - len(blocks_without_points)):
                current_block = memberships[data_index]
                # If this block was not marked as missing
                if current_block not in blocks_without_points:
                    # Get block_points from the p3_data
                    x_coords = [point['point'][0] for point in points][data_index]
                    y_coords = [point['point'][1] for point in points][data_index]
                    data_index += 1


                    # Adjust the coordinates: add 12.5 offset and subtract grid offset for the current block
                    new_x = x_coords - grid_x.flatten()[current_block]
                    new_y = y_coords - grid_y.flatten()[current_block]
                    trial_data.append((new_x, new_y, current_block))
            trial_coords[trial_index] = trial_data
        results[key] = trial_coords

    return results

Combine_blocks = Combine_16_blocks(All_Subject_tBBTs_errors)


# def analyze_and_plot_16_locations_xy(Combine_blocks, cmap_choice, participant):
#     """
#     Plot real x, y coordinates for each of the 16 locations, calculate mean x, mean y, 
#     and compute spread (standard deviation x and y), with same XY limits and center at (0,0).

#     Parameters:
#         Combine_blocks (dict): (subject, hand) -> list of (new_x, new_y, block) or dict of trials
#         cmap_choice: matplotlib colormap
#         participant: The participant to analyze (e.g., subject ID).
#     """

#     # Initialize data storage for each location
#     location_data = {hand: {loc: {'xs': [], 'ys': [], 'trials': []} for loc in range(16)} for hand in ['non_dominant', 'dominant']}
#     location_stats = {hand: {} for hand in ['non_dominant', 'dominant']}

#     # Collect coordinates for the specified participant
#     for key, data in Combine_blocks.items():
#         subject, hand = key
#         if subject != participant:
#             continue
#         hand_lower = hand.lower()
#         if isinstance(data, dict):
#             for trial_idx, trial_data in data.items():
#                 for new_x, new_y, block in trial_data:
#                     if hand_lower in location_data and block in location_data[hand_lower]:
#                         location_data[hand_lower][block]['xs'].append(new_x)
#                         location_data[hand_lower][block]['ys'].append(new_y)
#                         location_data[hand_lower][block]['trials'].append(trial_idx)
#         else:
#             for new_x, new_y, block in data:
#                 if hand_lower in location_data and block in location_data[hand_lower]:
#                     location_data[hand_lower][block]['xs'].append(new_x)
#                     location_data[hand_lower][block]['ys'].append(new_y)

#     # Define custom layout mappings
#     left_layout = [
#         [12, 13, 14, 15],  # top row
#         [8, 9, 10, 11],
#         [4, 5, 6, 7],
#         [0, 1, 2, 3]       # bottom row
#     ]
    
#     right_layout = [
#         [15, 14, 13, 12],      # mirrored top row
#         [11, 10, 9, 8],
#         [7, 6, 5, 4],
#         [3, 2, 1, 0]   # mirrored bottom row
#     ]

#     # Plot for each hand and location
#     for hand in ['non_dominant', 'dominant']:
#         fig, axes = plt.subplots(4, 4, figsize=(8, 8))
#         plt.subplots_adjust(wspace=0.1, hspace=0.1)

#         layout = left_layout if hand == 'non_dominant' else right_layout

#         for row in range(4):
#             for col in range(4):
#                 loc = layout[row][col]
#                 ax = axes[row, col]
#                 xs = np.array(location_data[hand][loc]['xs'])
#                 ys = np.array(location_data[hand][loc]['ys'])

#                 if len(xs) > 0 and len(ys) > 0:
#                     # Calculate mean x, mean y, std x, and std y
#                     mean_x = np.mean(xs)
#                     mean_y = np.mean(ys)
#                     std_x = np.std(xs)
#                     std_y = np.std(ys)
#                     location_stats[hand][loc] = {'mean_x': mean_x, 'mean_y': mean_y, 'std_x': std_x, 'std_y': std_y}

#                     # Scatter plot with grey color
#                     ax.scatter(xs, ys, color="grey", alpha=0.5, edgecolor='k')
#                     # Plot mean point
#                     ax.scatter(mean_x, mean_y, color='red', s=50, marker='X')

#                 ax.set_xlim(-8, 8)
#                 ax.set_ylim(-8, 8)
#                 ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
#                 ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
#                 ax.axis('off')  # Remove axis ticks and labels
#                 # ax.set_title(f'{hand} . Loc {loc}', fontsize=10)

#         plt.tight_layout()
#         plt.show()

#     return location_stats

# # Figure 2B - one participant 
# stats = analyze_and_plot_16_locations_xy(Combine_blocks, cmap_choice, participant="07/22/HW")

# # Figure 2B - one participant one location

def analyze_and_plot_selected_locations_xy(Combine_blocks, cmap_choice, participant, selected_indices):
    """
    Plot real x, y coordinates for selected locations, calculate mean x, mean y, 
    and compute spread (standard deviation x and y), with same XY limits and center at (0,0).

    Parameters:
        Combine_blocks (dict): (subject, hand) -> list of (new_x, new_y, block) or dict of trials
        cmap_choice: matplotlib colormap
        participant: The participant to analyze (e.g., subject ID).
        selected_indices: List of selected location indices to plot.
    """

    # Initialize data storage for each location
    location_data = {hand: {loc: {'xs': [], 'ys': [], 'trials': []} for loc in selected_indices} for hand in ['non_dominant', 'dominant']}
    location_stats = {hand: {} for hand in ['non_dominant', 'dominant']}

    # Collect coordinates for the specified participant
    for key, data in Combine_blocks.items():
        subject, hand = key
        if subject != participant:
            continue
        hand_lower = hand.lower()
        if isinstance(data, dict):
            for trial_idx, trial_data in data.items():
                for new_x, new_y, block in trial_data:
                    if hand_lower in location_data and block in location_data[hand_lower]:
                        location_data[hand_lower][block]['xs'].append(new_x)
                        location_data[hand_lower][block]['ys'].append(new_y)
                        location_data[hand_lower][block]['trials'].append(trial_idx)
        else:
            for new_x, new_y, block in data:
                if hand_lower in location_data and block in location_data[hand_lower]:
                    location_data[hand_lower][block]['xs'].append(new_x)
                    location_data[hand_lower][block]['ys'].append(new_y)

    # Define custom layout mappings
    left_layout = [
        [12, 13, 14, 15],  # top row
        [8, 9, 10, 11],
        [4, 5, 6, 7],
        [0, 1, 2, 3]       # bottom row
    ]
    
    right_layout = [
        [15, 14, 13, 12],      # mirrored top row
        [11, 10, 9, 8],
        [7, 6, 5, 4],
        [3, 2, 1, 0]   # mirrored bottom row
    ]

    # Plot for each hand and selected locations
    for hand in ['non_dominant', 'dominant']:
        fig, axes = plt.subplots(1, len(selected_indices), figsize=(4 * len(selected_indices), 4))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        layout = left_layout if hand == 'non_dominant' else right_layout

        for i, loc in enumerate(selected_indices):
            ax = axes[i] if len(selected_indices) > 1 else axes
            xs = np.array(location_data[hand][loc]['xs'])
            ys = np.array(location_data[hand][loc]['ys'])

            if len(xs) > 0 and len(ys) > 0:
                # Calculate mean x, mean y, std x, and std y
                mean_x = np.mean(xs)
                mean_y = np.mean(ys)
                std_x = np.std(xs)
                std_y = np.std(ys)
                location_stats[hand][loc] = {'mean_x': mean_x, 'mean_y': mean_y, 'std_x': std_x, 'std_y': std_y}

                # Scatter plot with grey color
                ax.scatter(xs, ys, color="grey", alpha=0.5, s=100, edgecolor='k')
                print(len(xs), len(ys))
                # Plot mean point
                ax.scatter(mean_x, mean_y, color='red', s=50, marker='X')

            ax.set_xlim(-8, 8)
            ax.set_xticks([-8, -4, 0, 4, 8])
            ax.set_ylim(-8, 8)
            ax.set_yticks([-8, -4, 0, 4, 8])

            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
            # ax.axis('off')  # Remove axis ticks and labels\
            ax.set_xlabel('X (mm)', fontsize=14)
            ax.set_ylabel('Y (mm)', fontsize=14)
            if hand == 'dominant':
                ax.set_title(f'Leftward placement', fontsize=16)
            else:
                ax.set_title(f'Rightward placement', fontsize=16)

            sns.despine()

        plt.tight_layout()
        plt.show()

    return location_stats

# Example usage - plot only selected indices
selected_indices = [11]
stats = analyze_and_plot_selected_locations_xy(Combine_blocks, cmap_choice, participant="07/22/HW", selected_indices=selected_indices)






# def analyze_and_plot_16_locations_xy_overlay(Combine_blocks, cmap_choice):
#     """
#     Overlay real x, y coordinates for each of the 16 locations across all participants, calculate mean x, mean y, 
#     and compute spread (standard deviation x and y), with same XY limits and center at (0,0).

#     Parameters:
#         Combine_blocks (dict): (subject, hand) -> list of (new_x, new_y, block) or dict of trials
#         cmap_choice: matplotlib colormap
#     """

#     # Initialize data storage for each location
#     location_data = {hand: {loc: {'xs': [], 'ys': []} for loc in range(16)} for hand in ['non_dominant', 'dominant']}
#     location_stats = {hand: {} for hand in ['non_dominant', 'dominant']}

#     # Collect coordinates across all participants
#     for key, data in Combine_blocks.items():
#         _, hand = key
#         hand_lower = hand.lower()
#         if isinstance(data, dict):
#             for trial_data in data.values():
#                 for new_x, new_y, block in trial_data:
#                     if hand_lower in location_data and block in location_data[hand_lower]:
#                         location_data[hand_lower][block]['xs'].append(new_x)
#                         location_data[hand_lower][block]['ys'].append(new_y)
#         else:
#             for new_x, new_y, block in data:
#                 if hand_lower in location_data and block in location_data[hand_lower]:
#                     location_data[hand_lower][block]['xs'].append(new_x)
#                     location_data[hand_lower][block]['ys'].append(new_y)

#     # Define custom layout mappings
#     left_layout = [
#         [12, 13, 14, 15],  # top row
#         [8, 9, 10, 11],
#         [4, 5, 6, 7],
#         [0, 1, 2, 3]       # bottom row
#     ]
    
#     right_layout = [
#         [15, 14, 13, 12],      # mirrored top row
#         [11, 10, 9, 8],
#         [7, 6, 5, 4],
#         [3, 2, 1, 0]   # mirrored bottom row
#     ]

#     # Plot for each hand and location
#     for hand in ['non_dominant', 'dominant']:
#         fig, axes = plt.subplots(4, 4, figsize=(8, 8))
#         plt.subplots_adjust(wspace=0.1, hspace=0.1)

#         layout = left_layout if hand == 'non_dominant' else right_layout

#         for row in range(4):
#             for col in range(4):
#                 loc = layout[row][col]
#                 ax = axes[row, col]
#                 xs = np.array(location_data[hand][loc]['xs'])
#                 ys = np.array(location_data[hand][loc]['ys'])

#                 if len(xs) > 0 and len(ys) > 0:
#                     # Calculate mean x, mean y, std x, and std y
#                     mean_x = np.mean(xs)
#                     mean_y = np.mean(ys)
#                     std_x = np.std(xs)
#                     std_y = np.std(ys)
#                     location_stats[hand][loc] = {'mean_x': mean_x, 'mean_y': mean_y, 'std_x': std_x, 'std_y': std_y}

#                     # Scatter plot with grey color
#                     ax.scatter(xs, ys, color="grey", alpha=0.1, edgecolor='k')
#                     # Plot mean point
#                     ax.scatter(mean_x, mean_y, color='red', s=50, marker='X')

#                 ax.set_xlim(-8, 8)
#                 ax.set_ylim(-8, 8)
#                 ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
#                 ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
#                 ax.axis('off')  # Remove axis ticks and labels

#         plt.tight_layout()
#         plt.show()

#     return location_stats

# stats_overlay = analyze_and_plot_16_locations_xy_overlay(Combine_blocks, cmap_choice)

# Figure 2D - All participants density heatmap overlay
def analyze_and_plot_16_locations_density_heatmap_overlay(Combine_blocks, cmap_choice, bin_width=0.5):
    """
    Overlay density heatmaps for each of the 16 locations across all participants, calculate mean x, mean y, 
    and compute spread (standard deviation x and y), with same XY limits and center at (0,0).

    Parameters:
        Combine_blocks (dict): (subject, hand) -> list of (new_x, new_y, block) or dict of trials
        cmap_choice: matplotlib colormap
        bin_width (float): Width of each bin for the heatmap.
    """

    # Initialize data storage for each location
    location_data = {hand: {loc: {'xs': [], 'ys': []} for loc in range(16)} for hand in ['non_dominant', 'dominant']}
    location_stats = {hand: {} for hand in ['non_dominant', 'dominant']}

    # Collect coordinates across all participants
    for key, data in Combine_blocks.items():
        _, hand = key
        hand_lower = hand.lower()
        if isinstance(data, dict):
            for trial_data in data.values():
                for new_x, new_y, block in trial_data:
                    if hand_lower in location_data and block in location_data[hand_lower]:
                        location_data[hand_lower][block]['xs'].append(new_x)
                        location_data[hand_lower][block]['ys'].append(new_y)
        else:
            for new_x, new_y, block in data:
                if hand_lower in location_data and block in location_data[hand_lower]:
                    location_data[hand_lower][block]['xs'].append(new_x)
                    location_data[hand_lower][block]['ys'].append(new_y)

    # Define custom layout mappings
    left_layout = [
        [12, 13, 14, 15],  # top row
        [8, 9, 10, 11],
        [4, 5, 6, 7],
        [0, 1, 2, 3]       # bottom row
    ]
    
    right_layout = [
        [15, 14, 13, 12],      # mirrored top row
        [11, 10, 9, 8],
        [7, 6, 5, 4],
        [3, 2, 1, 0]   # mirrored bottom row
    ]

    # Plot for each hand and location
    for hand in ['non_dominant', 'dominant']:
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        layout = left_layout if hand == 'non_dominant' else right_layout

        for row in range(4):
            for col in range(4):
                loc = layout[row][col]
                ax = axes[row, col]
                xs = np.array(location_data[hand][loc]['xs'])
                ys = np.array(location_data[hand][loc]['ys'])

                if len(xs) > 0 and len(ys) > 0:
                    # Calculate density heatmap
                    x_min, x_max = np.min(xs), np.max(xs)
                    y_min, y_max = np.min(ys), np.max(ys)
                    x_bins = np.arange(x_min, x_max + bin_width, bin_width)
                    y_bins = np.arange(y_min, y_max + bin_width, bin_width)
                    heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=[x_bins, y_bins], density=True)
                    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

                    # Plot heatmap
                    ax.imshow(heatmap.T, extent=extent, origin='lower', cmap=cmap_choice, aspect='auto')

                ax.set_xlim(-8, 8)
                ax.set_ylim(-8, 8)
                ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
                ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
                ax.axis('off')  # Remove axis ticks and labels

        plt.tight_layout()
        plt.show()

    return location_stats

stats_overlay = analyze_and_plot_16_locations_density_heatmap_overlay(Combine_blocks, cmap_choice, bin_width=1)

### Analyze distribution (uniformity, bias, quadrants) and plot polar histograms (rose diagrams)
def analyze_and_plot_left_right(Combine_blocks, cmap_choice, subject=None, bin_width=np.pi/10):
    """
    Analyze distribution (uniformity, bias, quadrants) and plot polar histograms (rose diagrams).

    Performs chi-square tests for uniformity across quadrants, non-dominant vs dominant, top vs bottom.
    Computes mean resultant vector direction and Rayleigh test for circular uniformity.

    Parameters:
        Combine_blocks (dict): (subject, hand) -> list of (new_x, new_y, block) or dict of trials
        cmap_choice: matplotlib colormap
        subject: The subject to analyze (e.g., subject ID). If None, analyze all subjects.
        bin_width: Width of each bin for the polar histogram (in radians).
    """

    xs_non_dominant, ys_non_dominant, xs_dominant, ys_dominant = [], [], [], []

    # Collect coordinates
    for key, data in Combine_blocks.items():
        subject_key, hand = key
        if subject is not None and subject_key != subject:
            continue
        hand_lower = hand.lower()
        if isinstance(data, dict):
            coords_iter = [pt for trial in data.values() for pt in trial]
        else:
            coords_iter = data

        for new_x, new_y, _ in coords_iter:
            if hand_lower == 'non_dominant':
                xs_non_dominant.append(new_x)
                ys_non_dominant.append(new_y)
            elif hand_lower == 'dominant':
                xs_dominant.append(new_x)
                ys_dominant.append(new_y)

    xs_non_dominant, ys_non_dominant = np.array(xs_non_dominant), np.array(ys_non_dominant)
    xs_dominant, ys_dominant = np.array(xs_dominant), np.array(ys_dominant)

    # Convert to polar
    r_non_dominant = np.sqrt(xs_non_dominant**2 + ys_non_dominant**2)
    theta_non_dominant = np.arctan2(ys_non_dominant, xs_non_dominant)

    r_dominant = np.sqrt(xs_dominant**2 + ys_dominant**2)
    theta_dominant = np.arctan2(ys_dominant, xs_dominant)

    # -------------------------
    #  Statistical tests
    # -------------------------
    def circular_chi_tests(xs, ys, theta):
        results = {}

        # Quadrants counts
        quad_counts = [
            np.sum((xs >= 0) & (ys >= 0)),  # Q1
            np.sum((xs < 0) & (ys >= 0)),   # Q2
            np.sum((xs < 0) & (ys < 0)),    # Q3
            np.sum((xs >= 0) & (ys < 0)),   # Q4
        ]
        chi_result = chisquare(quad_counts)
        results['quadrants_statistic'] = chi_result.statistic
        results['quadrants_p'] = chi_result.pvalue

        # Non-dominant vs Dominant counts
        lr_counts = [np.sum(xs < 0), np.sum(xs >= 0)]
        chi_result = chisquare(lr_counts)
        results['non_dominant_vs_dominant_statistic'] = chi_result.statistic
        results['non_dominant_vs_dominant_p'] = chi_result.pvalue

        # Top vs Bottom counts
        tb_counts = [np.sum(ys >= 0), np.sum(ys < 0)]
        chi_result = chisquare(tb_counts)
        results['top_vs_bottom_statistic'] = chi_result.statistic
        results['top_vs_bottom_p'] = chi_result.pvalue

        # Mean direction (circular mean)
        results['mean_direction_rad'] = circmean(theta, high=np.pi, low=-np.pi)

        # Rayleigh test for circular uniformity (approximation)
        n = len(theta)
        R = np.sqrt(np.sum(np.cos(theta))**2 + np.sum(np.sin(theta))**2) / n
        z = n * R**2
        p_rayleigh = np.exp(-z) * (1 + (2*z - z**2)/(4*n) - (24*z - 132*z**2 + 76*z**3 - 9*z**4)/(288*n**2))
        results['rayleigh_p'] = p_rayleigh

        return results

    stats_non_dominant = circular_chi_tests(xs_non_dominant, ys_non_dominant, theta_non_dominant)
    stats_dominant = circular_chi_tests(xs_dominant, ys_dominant, theta_dominant)

    # -------------------------
    #  Plot rose diagrams with mean direction arrow
    # -------------------------
    fig, axes = plt.subplots(1, 2, subplot_kw=dict(projection='polar'), figsize=(12, 6))

    # Non-dominant hand
    bin_edges_non_dominant = np.arange(-np.pi, np.pi + bin_width, bin_width)
    counts_non_dominant, _ = np.histogram(theta_non_dominant, bins=bin_edges_non_dominant, weights=r_non_dominant)
    axes[1].tick_params(axis='both', which='major', labelsize=18)
    width_non_dominant = bin_width
    max_non_dominant = counts_non_dominant.max() if counts_non_dominant.max() != 0 else 1
    norm_non_dominant = counts_non_dominant / max_non_dominant

    axes[1].bar(bin_edges_non_dominant[:-1], counts_non_dominant, width=width_non_dominant, bottom=0.0,
                color=[cmap_choice(val) for val in norm_non_dominant],
                edgecolor='k', alpha=0.75)
    # axes[1].set_title(f"Non-Dominant Hand: {subject if subject else 'All Subjects'}", y=1.05)
    # Draw mean direction arrow
    axes[1].arrow(stats_non_dominant['mean_direction_rad'], 0, 0, max_non_dominant, width=0.05, color='red', alpha=0.8)

    # Annotate n at the top right
    axes[1].annotate(f"n = {len(theta_non_dominant)}", xy=(1, 1), xycoords='axes fraction', fontsize=18, ha='right', va='top')

    # Dominant hand
    bin_edges_dominant = np.arange(-np.pi, np.pi + bin_width, bin_width)
    counts_dominant, _ = np.histogram(theta_dominant, bins=bin_edges_dominant, weights=r_dominant)
    axes[0].tick_params(axis='both', which='major', labelsize=18)
    width_dominant = bin_width
    max_dominant = counts_dominant.max() if counts_dominant.max() != 0 else 1
    norm_dominant = counts_dominant / max_dominant

    axes[0].bar(bin_edges_dominant[:-1], counts_dominant, width=width_dominant, bottom=0.0,
                color=[cmap_choice(val) for val in norm_dominant],
                edgecolor='k', alpha=0.75)
    # axes[0].set_title(f"Dominant Hand: {subject if subject else 'All Subjects'}", y=1.05)
    # Draw mean direction arrow
    axes[0].arrow(stats_dominant['mean_direction_rad'], 0, 0, max_dominant, width=0.05, color='red', alpha=0.8)

    # Annotate n at the top right
    axes[0].annotate(f"n = {len(theta_dominant)}", xy=(1, 1), xycoords='axes fraction', fontsize=18, ha='right', va='top')

    # Set the same radial scale for both plots
    max_r = max(max_non_dominant, max_dominant)
    axes[0].set_ylim(0, max_r)
    axes[1].set_ylim(0, max_r)

    plt.tight_layout()
    plt.show()

    return {"non_dominant": stats_non_dominant, "dominant": stats_dominant}

# Figure 2E - All participants density heatmap overlay across 16 locations
stats_all = analyze_and_plot_left_right(Combine_blocks, cmap_choice, bin_width=np.pi/12)

# Figure 2C - one participant density heatmap overlay across 16 locations
stats_single = analyze_and_plot_left_right(Combine_blocks, cmap_choice, subject="07/22/HW", bin_width=np.pi/12)






### Analyze distribution (uniformity, bias, quadrants) and plot polar histograms (rose diagrams)
# Figure 2E - All participants density heatmap overlay across 16 locations
def analyze_and_plot_left_right(Combine_blocks, cmap_choice, subject=None, bin_width=np.pi/10):
    """
    Analyze distribution and plot polar histograms (rose diagrams)
    with shared radial scaling across hands. Also return circular error
    directions of dominant vs non-dominant hands for each subject.
    """

    subject_means = {"non_dominant": [], "dominant": []}
    overall_means = {}
    circular_error_directions = {}

    # -------------------------
    # Collect subject mean directions
    # -------------------------
    for (subject_key, hand), data in Combine_blocks.items():
        if subject is not None and subject_key != subject:
            continue

        hand = hand.lower()
        xs, ys = [], []

        if isinstance(data, dict):
            coords_iter = [pt for trial in data.values() for pt in trial]
        else:
            coords_iter = data

        for new_x, new_y, _ in coords_iter:
            xs.append(new_x)
            ys.append(new_y)

        if len(xs) > 0:
            theta = np.arctan2(ys, xs)
            mean_dir = circmean(theta, high=np.pi, low=-np.pi)
            subject_means[hand].append(mean_dir)

            # Store circular error directions for each subject
            if subject_key not in circular_error_directions:
                circular_error_directions[subject_key] = {}
            if hand not in circular_error_directions[subject_key]:
                circular_error_directions[subject_key][hand] = []
            circular_error_directions[subject_key][hand].append(mean_dir)

    # -------------------------
    # Compute overall means
    # -------------------------
    for hand in ["dominant", "non_dominant"]:
        if len(subject_means[hand]) > 0:
            overall_means[hand] = circmean(
                subject_means[hand], high=np.pi, low=-np.pi
            )
        else:
            overall_means[hand] = np.nan

    # -------------------------
    # Shared binning and scaling
    # -------------------------
    bin_edges = np.linspace(
        -np.pi, np.pi, int(2 * np.pi / bin_width) + 1
    )

    counts_all = []
    for hand in ["dominant", "non_dominant"]:
        counts, _ = np.histogram(subject_means[hand], bins=bin_edges)
        counts_all.append(counts)

    global_max = max([c.max() for c in counts_all if c.size > 0])
    global_max = max(global_max, 1)  # safety

    # -------------------------
    # Plot
    # -------------------------
    fig, axes = plt.subplots(
        1, 2, subplot_kw=dict(projection="polar"), figsize=(12, 6)
    )

    for ax, hand, counts in zip(
        axes, ["dominant", "non_dominant"], counts_all
    ):
        norm_counts = counts / global_max

        ax.bar(
            bin_edges[:-1],
            counts,
            width=bin_width,
            bottom=0.0,
            color=[cmap_choice(v) for v in norm_counts],
            edgecolor="k",
            alpha=0.75,
        )

        # Subject mean arrows
        for mean_dir in subject_means[hand]:
            ax.arrow(
                mean_dir,
                0,
                0,
                global_max * 0.5,
                width=0.01,
                color="black",
                alpha=0.7,
            )

        # Overall mean arrow
        if not np.isnan(overall_means[hand]):
            ax.arrow(
                overall_means[hand],
                0,
                0,
                global_max,
                width=0.05,
                color="red",
                alpha=0.9,
                label="Overall Mean",
            )

        ax.set_ylim(0, global_max)
        # ax.set_title(f"{hand.replace('_', ' ').title()} Hand", y=1.08, fontsize=16)
        ax.tick_params(labelsize=14)
        # ax.legend(loc="upper right", fontsize=12)
        ax.annotate(f"n = {len(subject_means[hand])}", xy=(1, 1), xycoords="axes fraction",
                    fontsize=18, ha="right", va="top")

    plt.tight_layout()
    plt.show()

    return {
        "overall_means": overall_means,
        "circular_error_directions": circular_error_directions,
    }

stats_all = analyze_and_plot_left_right(Combine_blocks, cmap_choice, bin_width=np.pi/12)


import numpy as np
from scipy.stats import circmean
from scipy.stats import ttest_rel
from pingouin import intraclass_corr
import pandas as pd

# Example: extract circular error directions per subject
dominant_errors = [stats_all['circular_error_directions'][subj]['dominant'] 
                   for subj in stats_all['circular_error_directions'] if 'dominant' in stats_all['circular_error_directions'][subj]]

non_dominant_errors = [stats_all['circular_error_directions'][subj]['non_dominant'] 
                       for subj in stats_all['circular_error_directions'] if 'non_dominant' in stats_all['circular_error_directions'][subj]]

# Compute circular mean per subject
dominant_mean = np.array([circmean(trials, high=np.pi, low=-np.pi) for trials in dominant_errors])
non_dominant_mean = np.array([circmean(trials, high=np.pi, low=-np.pi) for trials in non_dominant_errors])


def circular_correlation(alpha, beta):
    alpha = np.array(alpha)
    beta = np.array(beta)
    alpha_bar = circmean(alpha, high=np.pi, low=-np.pi)
    beta_bar = circmean(beta, high=np.pi, low=-np.pi)
    num = np.sum(np.sin(alpha - alpha_bar) * np.sin(beta - beta_bar))
    den = np.sqrt(np.sum(np.sin(alpha - alpha_bar)**2) * np.sum(np.sin(beta - beta_bar)**2))
    return num / den

def circular_correlation_pvalue(alpha, beta, n_perm=10000, seed=42):
    np.random.seed(seed)
    r_obs = circular_correlation(alpha, beta)
    count = 0
    for _ in range(n_perm):
        beta_shuffled = np.random.permutation(beta)
        r_perm = circular_correlation(alpha, beta_shuffled)
        if abs(r_perm) >= abs(r_obs):
            count += 1
    p_value = count / n_perm
    return r_obs, p_value

# Example usage
r, p = circular_correlation_pvalue(dominant_mean, non_dominant_mean)
print("Circular correlation r =", r, "p-value =", p)


def plot_correlation(data_x, data_y, x_label, y_label, title, plot=True, x_range=None, y_range=None):
    """
    Calculate correlation and plot the relationship between two datasets.

    Args:
        data_x (list or array): Data for the x-axis.
        data_y (list or array): Data for the y-axis.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.
        plot (bool): Whether to display the plot. Default is True.
        x_range (tuple): Tuple specifying the x-axis range as (min, max). Default is None.
        y_range (tuple): Tuple specifying the y-axis range as (min, max). Default is None.

    Returns:
        tuple: Correlation coefficient and p-value.
    """
    # Calculate correlation and p-value
    r, p = circular_correlation_pvalue(dominant_mean, non_dominant_mean)

    if plot:
        # Plot correlation
        plt.figure(figsize=(6, 6))
        plt.scatter(data_x, data_y, s=50, alpha=0.7)

        plt.xlabel(x_label, fontsize=18)
        plt.ylabel(y_label, fontsize=18)

        plt.xticks(fontsize=16)

        # Determine x-axis integer ticks
        if x_range:
            plt.xlim(x_range)
            x_min, x_max = int(np.floor(x_range[0])), int(np.ceil(x_range[1]))
        else:
            x_min, x_max = int(np.floor(min(data_x))), int(np.ceil(max(data_x)))
        ax = plt.gca()  # Get the current axis
        ax.set_xticks(range(x_min, x_max + 1))  # only integer ticks

        plt.yticks(fontsize=16)
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}'))

        # Set x and y axis ranges if provided
        if x_range:
            plt.xlim(x_range)
        if y_range:
            plt.ylim(y_range)

        # Plot mean points as black
        mean_x = np.mean(data_x)
        mean_y = np.mean(data_y)
        plt.scatter(mean_x, mean_y, color='red', edgecolors='black', s=150, marker='s', label='Mean Point', zorder=5)

        # Plot line of unity
        min_val = min(min(data_x), min(data_y))
        max_val = max(max(data_x), max(data_y))
        plt.plot([min_val, max_val], [min_val, max_val], color='blue', linestyle='--', label='Line of Unity')

        sns.despine()

        # Annotate correlation and p-value at the top left corner
        plt.annotate(f'circular correlation = {r:.2f}\n n = {len(data_x)}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=18, ha='left', va='top')

        plt.show()
    return r, p

plot_correlation(
    dominant_mean,
    non_dominant_mean,
    'Dominant hand \n tBBT error direction (radians)',
    'Non-dominant hand \n tBBT error direction (radians)',
    'tBBT Dominant vs Non-dominant Hand Circular Error Direction Correlation',
)





#degree
# Example: extract circular error directions per subject
dominant_errors = [stats_all['circular_error_directions'][subj]['dominant'] 
                   for subj in stats_all['circular_error_directions'] if 'dominant' in stats_all['circular_error_directions'][subj]]

non_dominant_errors = [stats_all['circular_error_directions'][subj]['non_dominant'] 
                       for subj in stats_all['circular_error_directions'] if 'non_dominant' in stats_all['circular_error_directions'][subj]]

# Compute circular mean per subject
dominant_mean = np.array([circmean(trials, high=np.pi, low=-np.pi) for trials in dominant_errors])
non_dominant_mean = np.array([circmean(trials, high=np.pi, low=-np.pi) for trials in non_dominant_errors])


def circular_correlation(alpha, beta):
    alpha = np.array(alpha)
    beta = np.array(beta)
    alpha_bar = circmean(alpha, high=np.pi, low=-np.pi)
    beta_bar = circmean(beta, high=np.pi, low=-np.pi)
    num = np.sum(np.sin(alpha - alpha_bar) * np.sin(beta - beta_bar))
    den = np.sqrt(np.sum(np.sin(alpha - alpha_bar)**2) * np.sum(np.sin(beta - beta_bar)**2))
    return num / den

def circular_correlation_pvalue(alpha, beta, n_perm=10000, seed=42):
    np.random.seed(seed)
    r_obs = circular_correlation(alpha, beta)
    count = 0
    for _ in range(n_perm):
        beta_shuffled = np.random.permutation(beta)
        r_perm = circular_correlation(alpha, beta_shuffled)
        if abs(r_perm) >= abs(r_obs):
            count += 1
    p_value = count / n_perm
    return r_obs, p_value

# Example usage
r, p = circular_correlation_pvalue(dominant_mean, non_dominant_mean)
print("Circular correlation r =", r, "p-value =", p)


def plot_correlation(data_x, data_y, x_label, y_label, title, plot=True, x_range=None, y_range=None):
    """
    Calculate correlation and plot the relationship between two datasets.

    Args:
        data_x (list or array): Data for the x-axis (in radians).
        data_y (list or array): Data for the y-axis (in radians).
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.
        plot (bool): Whether to display the plot. Default is True.
        x_range (tuple): Tuple specifying the x-axis range as (min, max) in degrees. Default is None.
        y_range (tuple): Tuple specifying the y-axis range as (min, max) in degrees. Default is None.

    Returns:
        tuple: Correlation coefficient and p-value.
    """
    # Convert radians to degrees (0 to 360)
    data_x_deg = np.degrees(data_x) % 360
    data_y_deg = np.degrees(data_y) % 360

    # Calculate correlation and p-value
    r, p = circular_correlation_pvalue(data_x, data_y)

    if plot:
        # Plot correlation
        plt.figure(figsize=(6, 6))
        plt.scatter(data_x_deg, data_y_deg, s=50, alpha=0.7)

        plt.xlabel(x_label, fontsize=18)
        plt.ylabel(y_label, fontsize=18)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        # Determine x-axis integer ticks
        if x_range:
            plt.xlim(x_range)
            x_min, x_max = int(np.floor(x_range[0])), int(np.ceil(x_range[1]))
        else:
            x_min, x_max = 0, 360
        ax = plt.gca()  # Get the current axis
        ax.set_xticks(range(x_min, x_max + 1, 90))  # Tick every 60 degrees
        ax.set_yticks(range(x_min, x_max + 1, 90))  # Tick every 60 degrees

        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}°'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}°'))

        # Set x and y axis ranges if provided
        if x_range:
            plt.xlim(x_range)
        if y_range:
            plt.ylim(y_range)

        # Plot mean points as a red square
        mean_x = np.degrees(circmean(data_x, high=np.pi, low=-np.pi)) % 360
        mean_y = np.degrees(circmean(data_y, high=np.pi, low=-np.pi)) % 360
        print("Mean X (degrees):", mean_x, "Mean Y (degrees):", mean_y)
        plt.scatter(mean_x, mean_y, color='red', edgecolors='black', s=150, marker='s', label='Circmean', zorder=5)

        # Plot line of unity
        plt.plot([0, 360], [0, 360], color='blue', linestyle='--', label='Line of Unity')

        sns.despine()

        # Annotate correlation and p-value at the top left corner
        plt.annotate(f'circular correlation = {r:.2f}\n n = {len(data_x)}', xy=(0.05, 1), xycoords='axes fraction', fontsize=18, ha='left', va='top')

        plt.show()
    return r, p

plot_correlation(
    dominant_mean,
    non_dominant_mean,
    'Dominant hand \n tBBT error direction (degrees)',
    'Non-dominant hand \n tBBT error direction (degrees)',
    'tBBT Dominant vs Non-dominant Hand Circular Error Direction Correlation',
    x_range=(0, 360),
    y_range=(0, 360),
)



###----------------------------------------------------------
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

# Generate colors and draw icon (not mirrored)
placement_location_colors = generate_placement_colors(show_plot=False)


# def compute_centroids_and_spread(Combine_blocks):
#     """
#     Compute mean X, mean Y, standard deviation X, standard deviation Y, and total error for each grid target (16 locations)
#     across all participants for each hand.

#     Args:
#         Combine_blocks (dict): (subject, hand) -> list of (new_x, new_y, block) or dict of trials.

#     Returns:
#         dict: A dictionary containing the computed statistics for each hand and location.
#     """
#     # Initialize data storage for each location
#     location_data = {hand: {loc: {'xs': [], 'ys': []} for loc in range(16)} for hand in ['non_dominant', 'dominant']}

#     # Collect coordinates for each location across all participants
#     for key, data in Combine_blocks.items():
#         _, hand = key
#         hand_lower = hand.lower()
#         if isinstance(data, dict):
#             coords_iter = [pt for trial in data.values() for pt in trial]
#         else:
#             coords_iter = data

#         for new_x, new_y, block in coords_iter:
#             if hand_lower in location_data and block in location_data[hand_lower]:
#                 location_data[hand_lower][block]['xs'].append(new_x)
#                 location_data[hand_lower][block]['ys'].append(new_y)

#     # Compute statistics for each location
#     location_stats = {hand: {} for hand in ['non_dominant', 'dominant']}
#     for hand in ['non_dominant', 'dominant']:
#         for loc in range(16):
#             xs = np.array(location_data[hand][loc]['xs'])
#             ys = np.array(location_data[hand][loc]['ys'])

#             if len(xs) > 0 and len(ys) > 0:
#                 mean_x = np.mean(xs)
#                 mean_y = np.mean(ys)
#                 std_x = np.std(xs)
#                 std_y = np.std(ys)
#                 total_error = np.sqrt(mean_x**2 + mean_y**2)
#                 location_stats[hand][loc] = {
#                     'mean_x': mean_x,
#                     'mean_y': mean_y,
#                     'std_x': std_x,
#                     'std_y': std_y,
#                     'total_error': total_error
#                 }
#             else:
#                 location_stats[hand][loc] = {
#                     'mean_x': np.nan,
#                     'mean_y': np.nan,
#                     'std_x': np.nan,
#                     'std_y': np.nan,
#                     'total_error': np.nan
#                 }

#     return location_stats

# # Example usage
# centroids_and_spread = compute_centroids_and_spread(Combine_blocks)

# def compare_centroids_mean_with_colors(centroids_and_spread, placement_location_colors, metric):
#     """
#     Compare a specified metric (mean_x or mean_y) between non-dominant and dominant hands across all locations 
#     using a paired t-test, plot the results as a scatter plot with a linear correlation line, and 
#     color the points based on their placement location. Also calculate the linear correlation.

#     Args:
#         centroids_and_spread (dict): Dictionary containing mean_x or mean_y for non-dominant and dominant hands.
#         placement_location_colors (np.ndarray): Array of colors for each location.
#         metric (str): The metric to compare ('mean_x' or 'mean_y').

#     Returns:
#         tuple: t-statistic, p-value of the paired t-test, correlation coefficient, and p-value of the correlation.
#     """
#     non_dominant_metric = []
#     dominant_metric = []
#     colors = []

#     for loc in range(16):
#         non_dominant_value = centroids_and_spread['non_dominant'][loc][metric]
#         dominant_value = centroids_and_spread['dominant'][loc][metric]
#         if not np.isnan(non_dominant_value) and not np.isnan(dominant_value):
#             non_dominant_metric.append(non_dominant_value)
#             dominant_metric.append(dominant_value)
#             colors.append(placement_location_colors[loc])

#     # Perform paired t-test
#     t_stat, p_value_ttest = ttest_rel(non_dominant_metric, dominant_metric)

#     # Calculate linear correlation
#     corr_coeff, p_value_corr = pearsonr(non_dominant_metric, dominant_metric)

#     # Plot scatter plot with linear correlation line
#     plt.figure(figsize=(8, 6))
#     for i in range(len(non_dominant_metric)):
#         plt.scatter(non_dominant_metric[i], dominant_metric[i], color=colors[i], edgecolor='black', s=100, label=f"Loc {i+1}" if i < 16 else None)

#     # Add linear correlation line
#     m, b = np.polyfit(non_dominant_metric, dominant_metric, 1)
#     plt.plot(non_dominant_metric, np.array(non_dominant_metric) * m + b, color='red', linestyle='--', label='Correlation Line')

#     plt.xlabel(f'Non-Dominant Hand {metric.replace("_", " ").capitalize()}', fontsize=12)
#     plt.ylabel(f'Dominant Hand {metric.replace("_", " ").capitalize()}', fontsize=12)
#     plt.grid(True)

#     # Annotate correlation coefficient
#     plt.annotate(f'r = {corr_coeff:.2f}, p = {p_value_corr:.3f}', 
#                  xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top')

#     plt.show()

#     return t_stat, p_value_ttest, corr_coeff, p_value_corr

# # Perform the paired t-test and calculate correlation for mean_x
# t_stat_x, p_value_ttest_x, corr_coeff_x, p_value_corr_x = compare_centroids_mean_with_colors(centroids_and_spread, placement_location_colors, 'mean_x')

# # Perform the paired t-test and calculate correlation for mean_y
# t_stat_y, p_value_ttest_y, corr_coeff_y, p_value_corr_y = compare_centroids_mean_with_colors(centroids_and_spread, placement_location_colors, 'mean_y')



def compute_centroids_and_spread_for_all_subjects(Combine_blocks):
    """
    Compute mean X, mean Y, standard deviation X, standard deviation Y, and total error for each grid target (16 locations)
    for all subjects individually for each hand.

    Args:
        Combine_blocks (dict): (subject, hand) -> list of (new_x, new_y, block) or dict of trials.

    Returns:
        dict: A dictionary containing the computed statistics for each subject, hand, and location.
    """
    # Initialize data storage for each subject
    all_subjects_stats = {}

    # Collect data for each subject
    for key, data in Combine_blocks.items():
        subject, hand = key
        if subject not in all_subjects_stats:
            all_subjects_stats[subject] = {hand: {loc: {'xs': [], 'ys': []} for loc in range(16)}}
        elif hand not in all_subjects_stats[subject]:
            all_subjects_stats[subject][hand] = {loc: {'xs': [], 'ys': []} for loc in range(16)}

        if isinstance(data, dict):
            coords_iter = [pt for trial in data.values() for pt in trial]
        else:
            coords_iter = data

        for new_x, new_y, block in coords_iter:
            all_subjects_stats[subject][hand][block]['xs'].append(new_x)
            all_subjects_stats[subject][hand][block]['ys'].append(new_y)

    # Compute statistics for each subject, hand, and location
    for subject, hands in all_subjects_stats.items():
        for hand, locations in hands.items():
            for loc in range(16):
                xs = np.array(locations[loc]['xs'])
                ys = np.array(locations[loc]['ys'])

                if len(xs) > 0 and len(ys) > 0:
                    mean_x = np.mean(xs)
                    mean_y = np.mean(ys)
                    std_x = np.std(xs)
                    std_y = np.std(ys)
                    total_error = np.sqrt(mean_x**2 + mean_y**2)
                    locations[loc] = {
                        'mean_x': mean_x,
                        'mean_y': mean_y,
                        'std_x': std_x,
                        'std_y': std_y,
                        'total_error': total_error
                    }
                else:
                    locations[loc] = {
                        'mean_x': np.nan,
                        'mean_y': np.nan,
                        'std_x': np.nan,
                        'std_y': np.nan,
                        'total_error': np.nan
                    }

    return all_subjects_stats


def compare_centroids_mean_with_colors_for_all_subjects(all_subjects_stats, placement_location_colors, metric, subject=None):
    """
    Compare a specified metric (mean_x or mean_y) between non-dominant and dominant hands across all locations 
    for all subjects using paired t-tests, plot the results as scatter plots with linear correlation lines, and 
    color the points based on their placement location. Also calculate the linear correlation for each subject.

    Args:
        all_subjects_stats (dict): Dictionary containing mean_x or mean_y for non-dominant and dominant hands for all subjects.
        placement_location_colors (np.ndarray): Array of colors for each location.
        metric (str): The metric to compare ('mean_x' or 'mean_y').
        subject (str, optional): If provided, plot data for the specified subject only. Otherwise, plot for all subjects.

    Returns:
        dict: A dictionary containing t-test and correlation results for each subject.
    """
    results = {}

    for subj, hands in all_subjects_stats.items():
        non_dominant_metric = []
        dominant_metric = []
        colors = []

        for loc in range(16):
            non_dominant_value = hands['non_dominant'][loc][metric]
            dominant_value = hands['dominant'][loc][metric]
            if not np.isnan(non_dominant_value) and not np.isnan(dominant_value):
                non_dominant_metric.append(non_dominant_value)
                dominant_metric.append(dominant_value)
                colors.append(placement_location_colors[loc])

        # Perform paired t-test
        t_stat, p_value_ttest = ttest_rel(non_dominant_metric, dominant_metric)

        # Calculate linear correlation
        corr_coeff, p_value_corr = pearsonr(non_dominant_metric, dominant_metric)

        # Store results
        results[subj] = {
            't_stat': t_stat,
            'p_value_ttest': p_value_ttest,
            'corr_coeff': corr_coeff,
            'p_value_corr': p_value_corr
        }

        # Plot only if the subject matches or if no specific subject is provided
        if subject is None or subj == subject:
            # Plot scatter plot with linear correlation line
            plt.figure(figsize=(6, 6))
            for i in range(len(non_dominant_metric)):
                plt.scatter(dominant_metric[i],non_dominant_metric[i], color=colors[i], edgecolor='black', s=100, label=f"Loc {i+1}" if i < 16 else None)

            # Add linear correlation line
            m, b = np.polyfit(dominant_metric,non_dominant_metric , 1)
            plt.plot(dominant_metric, np.array(dominant_metric) * m + b, color='red', linestyle='--', label='Correlation Line')

            # Plot line of unity
            min_val = -3
            max_val = 3
            plt.plot([min_val, max_val], [min_val, max_val], color='blue', linestyle='--', label='Line of Unity')

            # Determine x-axis integer ticks
            if dominant_metric:
                x_min, x_max = -3, 3
                plt.xticks(range(x_min, x_max + 1))

            # Determine y-axis integer ticks
            if non_dominant_metric:
                y_min, y_max = -3, 3
                plt.yticks(range(y_min, y_max + 1))

            plt.ylabel(f'Non-dominant hand \n {metric.replace("_", " ")} (mm)', fontsize=18)
            plt.xlabel(f'Dominant hand \n {metric.replace("_", " ")} (mm)', fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid(False)
            plt.axis('equal')

            # Annotate correlation coefficient
            plt.annotate(f'r = {corr_coeff:.2f}, p = {p_value_corr:.3f} \n n = {len(non_dominant_metric)}', 
                         xy=(0.05, 0.95), xycoords='axes fraction', fontsize=18, ha='left', va='top')

            sns.despine()
            plt.show()

    return results


# Example usage for all subjects
all_subjects_centroids_and_spread = compute_centroids_and_spread_for_all_subjects(Combine_blocks)
all_subjects_results_x = compare_centroids_mean_with_colors_for_all_subjects(all_subjects_centroids_and_spread, placement_location_colors, 'mean_x', subject='07/22/HW')
all_subjects_results_y = compare_centroids_mean_with_colors_for_all_subjects(all_subjects_centroids_and_spread, placement_location_colors, 'mean_y', subject='07/22/HW')


def plot_boxplot_all_subjects_results(all_subjects_results_x, all_subjects_results_y, metric, title):
    """
    Plot a boxplot of the specified metric for all subjects for both X and Y metrics,
    and overlay data points using swarmplot. Add a horizontal line at y=0 and test if the values
    are significantly different from 0.

    Args:
        all_subjects_results_x (dict): Dictionary containing results for all subjects for X metrics.
        all_subjects_results_y (dict): Dictionary containing results for all subjects for Y metrics.
        metric (str): The metric to plot ('t_stat', 'p_value_ttest', 'corr_coeff', or 'p_value_corr').
        title (str): Title of the boxplot.
    """
    
    subjects = list(all_subjects_results_x.keys())
    data_x = [all_subjects_results_x[s][metric] for s in subjects]
    data_y = [all_subjects_results_y[s][metric] for s in subjects]

    # Build long-format DataFrame for seaborn
    df = pd.DataFrame({
        "Subject": subjects * 2,
        "Value": data_x + data_y,
        "Condition": ["mean X"] * len(subjects) + ["mean Y"] * len(subjects)
    })

    plt.figure(figsize=(6, 6))

    sns.boxplot(
        data=df, x="Condition", y="Value",
        linewidth=1.5, boxprops=dict(facecolor='none', edgecolor='black')
    )

    sns.swarmplot(
        data=df, x="Condition", y="Value",
        color="black", size=8, alpha=0.75
    )

    # Add a horizontal line at y=0
    plt.axhline(0, color='grey', linestyle='--', linewidth=1.5, alpha=0.5)

    # Test if the values are significantly different from 0
    t_stat_x, p_value_x = ttest_1samp(data_x, 0)
    t_stat_y, p_value_y = ttest_1samp(data_y, 0)

    # Print the p-values
    print(f"p-value for X: {p_value_x:.3f}")
    print(f"p-value for Y: {p_value_y:.3f}")

    plt.ylabel('Correlation', fontsize=18)
    plt.xlabel('', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(-1, 1)    
    plt.axis('equal')

    sns.despine()
    plt.tight_layout()
    plt.show()

# Example usage:
plot_boxplot_all_subjects_results(
    all_subjects_results_x,
    all_subjects_results_y,
    'corr_coeff',
    'Correlation Coefficient (mean_x vs mean_y)'
)



def assess_significance_between_hands(sBBTResult, iBBT_average_total_time_results, tBBT_average_total_time_results, tBBT_average_block_distance, dominant_mean, non_dominant_mean):
    """
    Assess whether the differences between dominant and non-dominant hand metrics are significant for sBBT, iBBT, tBBT tasks, 
    tBBT average block distance, and tBBT error direction.

    Args:
        sBBTResult (pd.DataFrame): DataFrame containing sBBT scores for dominant and non-dominant hands.
        iBBT_average_total_time_results (dict): Dictionary containing iBBT average total time for dominant and non-dominant hands.
        tBBT_average_total_time_results (dict): Dictionary containing tBBT average total time for dominant and non-dominant hands.
        tBBT_average_block_distance (dict): Dictionary containing tBBT average block distance for dominant and non-dominant hands.
        dominant_mean (array): Array of circular mean directions for the dominant hand (in radians).
        non_dominant_mean (array): Array of circular mean directions for the non-dominant hand (in radians).

    Returns:
        dict: A dictionary containing t-statistics and p-values for each task.
    """

    results = {}

    # sBBT task
    sBBT_dominant = sBBTResult['dominant']
    sBBT_nondominant = sBBTResult['non_dominant']
    t_stat, p_value = ttest_rel(sBBT_dominant, sBBT_nondominant)
    results['sBBT'] = {'t_stat': t_stat, 'p_value': p_value}

    # iBBT task
    iBBT_dominant = [hands['dominant'] for hands in iBBT_average_total_time_results.values() if 'dominant' in hands]
    iBBT_nondominant = [hands['non_dominant'] for hands in iBBT_average_total_time_results.values() if 'non_dominant' in hands]
    t_stat, p_value = ttest_rel(iBBT_dominant, iBBT_nondominant)
    results['iBBT'] = {'t_stat': t_stat, 'p_value': p_value}

    # tBBT task
    tBBT_dominant = [hands['dominant'] for hands in tBBT_average_total_time_results.values() if 'dominant' in hands]
    tBBT_nondominant = [hands['non_dominant'] for hands in tBBT_average_total_time_results.values() if 'non_dominant' in hands]
    t_stat, p_value = ttest_rel(tBBT_dominant, tBBT_nondominant)
    results['tBBT'] = {'t_stat': t_stat, 'p_value': p_value}

    # tBBT average block distance
    tBBT_block_distance_dominant = [hands['dominant'] for hands in tBBT_average_block_distance.values() if 'dominant' in hands]
    tBBT_block_distance_nondominant = [hands['non_dominant'] for hands in tBBT_average_block_distance.values() if 'non_dominant' in hands]
    t_stat, p_value = ttest_rel(tBBT_block_distance_dominant, tBBT_block_distance_nondominant)
    results['tBBT_block_distance'] = {'t_stat': t_stat, 'p_value': p_value}

    # tBBT error direction (convert radians to degrees before t-test)
    dominant_mean_deg = np.degrees(dominant_mean) % 360
    non_dominant_mean_deg = np.degrees(non_dominant_mean) % 360
    t_stat, p_value = ttest_rel(dominant_mean_deg, non_dominant_mean_deg)
    results['tBBT_error_direction'] = {'t_stat': t_stat, 'p_value': p_value}

    return results


# Example usage
significance_results = assess_significance_between_hands(
    sBBTResult, 
    iBBT_average_total_time_results, 
    tBBT_average_total_time_results, 
    tBBT_average_block_distance, 
    dominant_mean, 
    non_dominant_mean
)

for task, result in significance_results.items():
    print(f"{task}: t-statistic = {result['t_stat']:.2f}, p-value = {result['p_value']:.3f}")
