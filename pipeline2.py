import os
import utils
import pprint
import pandas as pd
import seaborn as sns
import numpy as np

# --- PARAMETERS ---
time_window_method = 3
window_size = 0.1 
prominence_threshold_speed = 400
prominence_threshold_position = 80
speed_threshold = 500
# Traj_folder = "/Volumes/MNHS-MoCap/Yilin-Honours/Subject/Traj/2025/"
# Box_Traj_folder = "/Volumes/MNHS-MoCap/Yilin-Honours/Box/Traj/2025/"
Traj_folder = "/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025"
Box_Traj_folder = "/Users/yilinwu/Desktop/Yilin-Honours/Box/Traj/2025"
Figure_folder = "/Users/yilinwu/Desktop/honours/Thesis/figure"
DataProcess_folder = "/Users/yilinwu/Desktop/honours data/DataProcess"

# --- GET ALL DATES ---
All_dates = sorted(utils.get_subfolders_with_depth(Traj_folder, depth=3))
All_dates = All_dates[6:7]  # Limit to the first 10 dates for testing; remove this line to process all dates

# --- PROCESS EACH DATE ---
results = {}
for date in All_dates:
    results[date] = utils.process_date(date, Traj_folder, Box_Traj_folder, Figure_folder, DataProcess_folder, 
                                 prominence_threshold_speed, prominence_threshold_position)

# --- GET REACH SPEED SEGMENTS ---
reach_speed_segments = utils.get_reach_speed_segments(results)

# --- CALCULATE REACH METRICS ---
reach_metrics = utils.calculate_reach_metrics(reach_speed_segments, results)
# pprint.pprint(reach_metrics['reach_durations']['06/19/1']['right']['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/06/19/1/CZ_tBBT01.csv'])

# --- DEFINE TIME WINDOWS BASED ON SELECTED METHOD ---
test_windows = utils.define_time_windows(time_window_method, reach_speed_segments, reach_metrics, frame_rate=200, window_size=window_size)

# --- CALCULATE REACH METRICS SPECIFIC TO TIME WINDOW ---
reach_TW_metrics = utils.calculate_reach_metrics_for_time_windows(test_windows, results)

# --- Save ALL LDLJ VALUES BY SUBJECT, HAND, AND TRIAL ---
utils.save_ldlj_values(reach_TW_metrics, DataProcess_folder)

# note: different ways to calculate reach LDLJ
# All_LDLJ = utils.calculate_LDLJ_for_time_windows(test_windows, results)

# --- CALCULATE SPARC FOR EACH TEST WINDOW FOR ALL DATES, HANDS, AND TRIALS ---
reach_sparc = utils.calculate_reach_sparc(test_windows, results)

# --- Save ALL SPARC VALUES BY SUBJECT, HAND, AND TRIAL ---
utils.save_sparc_values(reach_sparc, DataProcess_folder)

# --- PLOT EACH SEGMENT SPEED AS SUBPLOT WITH LDLJ AND SPARC VALUES ---
for date in reach_TW_metrics['reach_LDLJ']:
    for hand in ['right', 'left']:
        for trial in reach_TW_metrics['reach_LDLJ'][date][hand]:
            # Create a plot save path for the trial
            plot_save_path = os.path.join(Figure_folder, date.replace("/", "_"))
            os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)

            # Call the utility function to plot segments with LDLJ and SPARC
            utils.plot_segments_with_ldlj_and_sparc(
                date=date,
                hand=hand,
                trial=trial,
                test_windows=test_windows,
                results=results,
                reach_TW_metrics=reach_TW_metrics,
                reach_sparc=reach_sparc,
                save_path=plot_save_path
            )




# --- PLOT LINE PLOTS FOR LDLJ AND SPARC VALUES WITH TWO Y-AXES ---
for date in reach_TW_metrics['reach_LDLJ']:
    for hand in ['right', 'left']:
        for trial in reach_TW_metrics['reach_LDLJ'][date][hand]:
            # Use the previously created plot_save_path for each subject
            plot_save_path = os.path.join(Figure_folder, date.replace("/", "_"))
            os.makedirs(plot_save_path, exist_ok=True)
            import matplotlib.pyplot as plt

            # Extract LDLJ and SPARC values for the trial
            ldlj_values = reach_TW_metrics['reach_LDLJ'][date][hand][trial]
            sparc_values = reach_sparc[date][hand][trial]

            # Create a figure for the line plot
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Plot LDLJ values on the first y-axis
            ax1.set_xlabel('Time Window Index')
            ax1.set_ylabel('LDLJ', color='blue')
            ax1.plot(ldlj_values, label='LDLJ', color='blue', marker='o')
            ax1.tick_params(axis='y', labelcolor='blue')

            # Create a second y-axis for SPARC values
            ax2 = ax1.twinx()
            ax2.set_ylabel('SPARC', color='orange')
            ax2.plot(sparc_values, label='SPARC', color='orange', marker='x')
            ax2.tick_params(axis='y', labelcolor='orange')

            # Add title and show the plot
            plt.title(f'LDLJ and SPARC Values for {date}, {hand.capitalize()} Hand, Trial {trial}')
            fig.tight_layout()
            plt.show()

            # # Save the plot to the specified path
            # plot_file = os.path.join(plot_save_path, f"{hand}_trial_{trial}_ldlj_sparc.png")
            # plt.savefig(plot_file)
            # plt.close()



# --- PLOT HEATMAPS FOR LDLJ AND SPARC VALUES ---
import matplotlib.pyplot as plt

for date in reach_TW_metrics['reach_LDLJ']:
    for hand in ['right', 'left']:
        # Use the previously created plot_save_path for each subject
        plot_save_path = os.path.join(Figure_folder, date.replace("/", "_"))
        os.makedirs(plot_save_path, exist_ok=True)

        # Extract LDLJ and SPARC values for all trials
        trials = list(reach_TW_metrics['reach_LDLJ'][date][hand].keys())
        short_trials = [trial.split('/')[-1] for trial in trials]  # Shorten trial names
        ldlj_values = [reach_TW_metrics['reach_LDLJ'][date][hand][trial] for trial in trials]
        sparc_values = [reach_sparc[date][hand][trial] for trial in trials]

        # Convert LDLJ and SPARC values to 2D arrays for heatmap
        ldlj_array = np.array(ldlj_values)
        sparc_array = np.array(sparc_values)

        # Define x-axis labels as "Reach 1" to "Reach 16"
        x_labels = list(range(1, ldlj_array.shape[1] + 1))

        # Plot LDLJ heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(ldlj_array, annot=True, fmt=".2f", cmap="Blues", xticklabels=x_labels, yticklabels=short_trials)
        plt.title(f'LDLJ Heatmap for {date}, {hand.capitalize()} Hand')
        plt.xlabel('Reach')
        plt.ylabel('Trial')
        plt.tight_layout()
        plt.show()

        # Plot SPARC heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(sparc_array, annot=True, fmt=".2f", cmap="Oranges", xticklabels=x_labels, yticklabels=short_trials)
        plt.title(f'SPARC Heatmap for {date}, {hand.capitalize()} Hand')
        plt.xlabel('Reach')
        plt.ylabel('Trial')
        plt.tight_layout()
        plt.show()





# --- PLOT CORRELATION BETWEEN REACH LDLJ AND SPARC VALUES AS SUBPLOTS ---
for date in reach_TW_metrics['reach_LDLJ']:
    for hand in ['right', 'left']:
        trials = list(reach_TW_metrics['reach_LDLJ'][date][hand].keys())
        num_trials = len(trials)

        # Create a figure with subplots arranged in an 8x4 grid
        rows, cols = 8, 4
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(5 * cols, 4 * rows), sharey=True)
        fig.suptitle(f'Correlation Between Z-Scored LDLJ and SPARC for {date}, {hand.capitalize()} Hand', fontsize=16)

        # Flatten axes for easier indexing
        axes = axes.flatten()

        for i, trial in enumerate(trials):
            # Extract LDLJ and SPARC values for the trial
            ldlj_values = reach_TW_metrics['reach_LDLJ'][date][hand][trial]
            sparc_values = reach_sparc[date][hand][trial]

            # Z-score the LDLJ and SPARC values
            ldlj_zscores = (ldlj_values - np.mean(ldlj_values)) / np.std(ldlj_values)
            sparc_zscores = (sparc_values - np.mean(sparc_values)) / np.std(sparc_values)

            # Plot correlation as a scatter plot in the subplot
            ax = axes[i]
            ax.scatter(ldlj_zscores, sparc_zscores, color='purple', alpha=0.7)
            ax.set_title(f'Trial {trial.split("/")[-1]}')
            ax.set_xlabel('Z-Scored LDLJ')
            if i % cols == 0:
                ax.set_ylabel('Z-Scored SPARC')
            ax.grid(True)

        # Hide any unused subplots
        for j in range(len(trials), len(axes)):
            axes[j].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
        plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pingouin import intraclass_corr

def smoothness_diagnostic_report(reach_TW_metrics, reach_sparc, date, hand, plot=True):
    trials = list(reach_TW_metrics['reach_LDLJ'][date][hand].keys())
    short_trials = [t.split('/')[-1] for t in trials]

    # Extract arrays
    ldlj_array = np.array([reach_TW_metrics['reach_LDLJ'][date][hand][t] for t in trials])
    sparc_array = np.array([reach_sparc[date][hand][t] for t in trials])

    # Store raw arrays for Bland-Altman
    ldlj_array_raw = ldlj_array.copy()
    sparc_array_raw = sparc_array.copy()

    # Z-score for correlation and ICC (per trial)
    ldlj_array = (ldlj_array - np.mean(ldlj_array, axis=1, keepdims=True)) / np.std(ldlj_array, axis=1, keepdims=True)
    sparc_array = (sparc_array - np.mean(sparc_array, axis=1, keepdims=True)) / np.std(sparc_array, axis=1, keepdims=True)

    # === Trial-Level Correlation ===
    trial_corrs = []
    for i in range(len(trials)):
        ldlj = ldlj_array[i]
        sparc = sparc_array[i]
        if np.std(ldlj) == 0 or np.std(sparc) == 0:
            trial_corrs.append(np.nan)
        else:
            corr, _ = pearsonr(ldlj, sparc)
            trial_corrs.append(corr)

    # === Reach-Level Correlation Across Trials ===
    reach_corrs = []
    for r in range(ldlj_array.shape[1]):
        try:
            corr, _ = pearsonr(ldlj_array[:, r], sparc_array[:, r])
            reach_corrs.append(corr)
        except:
            reach_corrs.append(np.nan)

    # === Bland-Altman Data (using raw values) ===
    ldlj_flat = ldlj_array_raw.flatten()
    sparc_flat = sparc_array_raw.flatten()
    mean_vals = (ldlj_flat + sparc_flat) / 2
    diff_vals = ldlj_flat - sparc_flat
    ba_mean = np.mean(diff_vals)
    ba_upper = ba_mean + 1.96 * np.std(diff_vals)
    ba_lower = ba_mean - 1.96 * np.std(diff_vals)

    # === ICC ===
    df_ldlj = pd.DataFrame(ldlj_array, columns=[f'Reach {i+1}' for i in range(ldlj_array.shape[1])])
    df_ldlj = df_ldlj.reset_index().melt(id_vars='index', var_name='reach', value_name='ldlj')
    icc_ldlj = intraclass_corr(data=df_ldlj, targets='reach', raters='index', ratings='ldlj').round(3)

    df_sparc = pd.DataFrame(sparc_array, columns=[f'Reach {i+1}' for i in range(sparc_array.shape[1])])
    df_sparc = df_sparc.reset_index().melt(id_vars='index', var_name='reach', value_name='sparc')
    icc_sparc = intraclass_corr(data=df_sparc, targets='reach', raters='index', ratings='sparc').round(3)

    # === PLOTTING ===
    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Smoothness Diagnostic: {hand.capitalize()} Hand, {date}", fontsize=16)

        # 1. Trial correlation histogram
        axs[0, 0].hist([c for c in trial_corrs if not np.isnan(c)], bins=10, color='mediumpurple', edgecolor='black')
        axs[0, 0].set_title("Trial-wise LDLJ-SPARC Correlation")
        axs[0, 0].set_xlabel("Pearson Correlation")
        axs[0, 0].set_ylabel("Number of Trials")
        axs[0, 0].axvline(0, color='red', linestyle='--')
        axs[0, 0].grid(True)

        # 2. Reach correlation line
        axs[0, 1].plot(reach_corrs, marker='o', color='teal')
        axs[0, 1].set_title("Reach-wise Correlation Across Trials")
        axs[0, 1].set_xlabel("Reach Index")
        axs[0, 1].set_ylabel("Pearson r")
        axs[0, 1].axhline(0, color='red', linestyle='--')
        axs[0, 1].grid(True)

        # 3. Bland-Altman
        axs[1, 0].scatter(mean_vals, diff_vals, alpha=0.5, color='orange')
        axs[1, 0].axhline(ba_mean, color='blue', linestyle='--', label='Mean Difference')
        axs[1, 0].axhline(ba_upper, color='red', linestyle='--', label='Upper LoA')
        axs[1, 0].axhline(ba_lower, color='red', linestyle='--', label='Lower LoA')
        axs[1, 0].set_title("Bland-Altman: LDLJ vs SPARC (Raw Values)")
        axs[1, 0].set_xlabel("Mean of LDLJ and SPARC")
        axs[1, 0].set_ylabel("LDLJ - SPARC")
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # 4. ICC bar plot
        icc_vals = [icc_ldlj.loc[icc_ldlj['Type'] == 'ICC2', 'ICC'].values[0],
                    icc_sparc.loc[icc_sparc['Type'] == 'ICC2', 'ICC'].values[0]]
        axs[1, 1].bar(['LDLJ ICC', 'SPARC ICC'], icc_vals, color=['blue', 'orange'])
        axs[1, 1].set_ylim(0, 1)
        axs[1, 1].set_title("Intra-Class Correlation (ICC2)")
        axs[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # === TEXT REPORT ===
    report = {
        "mean_trial_corr": np.nanmean(trial_corrs),
        "std_trial_corr": np.nanstd(trial_corrs),
        "mean_reach_corr": np.nanmean(reach_corrs),
        "std_reach_corr": np.nanstd(reach_corrs),
        "bland_altman_mean_diff": ba_mean,
        "bland_altman_upper": ba_upper,
        "bland_altman_lower": ba_lower,
        "icc_ldlj": icc_ldlj.loc[icc_ldlj['Type'] == 'ICC2', 'ICC'].values[0],
        "icc_sparc": icc_sparc.loc[icc_sparc['Type'] == 'ICC2', 'ICC'].values[0],
    }

    return report

report = smoothness_diagnostic_report(reach_TW_metrics, reach_sparc, date='06/19/CZ', hand='right')
print(report)





































# --- PLOT JERK SEGMENTS WITH OVERLAYED SPEED ---
for date in reach_TW_metrics['reach_LDLJ']:
    for hand in ['right', 'left']:
        for trial in reach_TW_metrics['reach_LDLJ'][date][hand]:
            # Use the previously created plot_save_path for each subject
            plot_save_path = os.path.join(Figure_folder, date.replace("/", "_"))
            os.makedirs(plot_save_path, exist_ok=True)
            utils.plot_jerk_segments(
                date=date,
                hand=hand,
                trial=trial,
                test_windows=test_windows,
                results=results,
                reach_TW_metrics=reach_TW_metrics,
                save_path=plot_save_path
            )

# # --- PLOT SEGMENTS ---
# for date in reach_TW_metrics['reach_LDLJ']:
#     for hand in ['right', 'left']:
#         for trial in reach_TW_metrics['reach_LDLJ'][date][hand]:
#             # Use the previously created plot_save_path for each subject
#             plot_save_path = os.path.join(Figure_folder, date.replace("/", "_"))
#             os.makedirs(plot_save_path, exist_ok=True)
#             utils.plot_segments(
#                 date=date,
#                 hand=hand,
#                 trial=trial,
#                 test_windows=test_windows,
#                 results=results,
#                 reach_TW_metrics=reach_TW_metrics,
#                 save_path=plot_save_path
#             )

# --- PLOT REACH LDLJ VALUES OVER TRIALS ---
for date in reach_TW_metrics['reach_LDLJ']:
    for hand in ['right', 'left']:
        # Use the previously created plot_save_path for each subject
        plot_save_path = os.path.join(Figure_folder, date.replace("/", "_"))
        os.makedirs(plot_save_path, exist_ok=True)
        utils.plot_reach_ldlj_over_trials(
            reach_TW_metrics=reach_TW_metrics,
            date=date,
            hand=hand,
            save_path=plot_save_path
        )

# --- PLOT REACH LDLJ VALUES AGAINST REACH DURATIONS FOR EACH DATE AND HAND ---
for date in reach_TW_metrics['reach_LDLJ']:
    for hand in ['right', 'left']:
        # Use the previously created plot_save_path for each subject
        plot_save_path = os.path.join(Figure_folder, date.replace("/", "_"))
        os.makedirs(plot_save_path, exist_ok=True)
        utils.plot_reach_ldlj_vs_duration(
            reach_TW_metrics=reach_TW_metrics,
            reach_metrics=reach_metrics,
            date=date,
            hand=hand,
            save_path=plot_save_path
        )

# --- CALCULATE REACH LDLJ VALUES AGAINST REACH DURATIONS AND CORRELATIONS ---
reach_ldlj_duration_correlations = {}
for date in reach_TW_metrics['reach_LDLJ']:
    reach_ldlj_duration_correlations[date] = {}
    for hand in ['right', 'left']:
        reach_ldlj_duration_correlations[date][hand] = utils.calculate_ldlj_vs_duration_correlation(
            reach_TW_metrics=reach_TW_metrics,
            reach_metrics=reach_metrics,
            date=date,
            hand=hand
        )

# --- PLOT REACH LDLJ DURATION CORRELATIONS AS HISTOGRAM BY HAND ---
utils.plot_ldlj_duration_correlations_histogram_by_hand(
    reach_ldlj_duration_correlations=reach_ldlj_duration_correlations,
    save_path=Figure_folder
)

# --- PLOT REACH LDLJ DURATION CORRELATIONS AS HISTOGRAM ---
utils.plot_ldlj_duration_correlations_histogram(
    reach_ldlj_duration_correlations=reach_ldlj_duration_correlations,
    save_path=Figure_folder
)


























import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from scipy.signal import find_peaks
import math
from scipy.stats import zscore
from scipy.stats import ttest_1samp
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# # --- CSV_TO_TRAJ_DATA ---
# def CSV_To_traj_data(file_path):
#     """
#     Reads a CSV file containing trajectory data, processes it, and extracts specific columns based on predefined indices.

#     Args:
#         file_path (str): Path to the CSV file.

#     Returns:
#         dict: A dictionary containing extracted data for each prefix.
#     """
#     # Read the file, skipping the first 5 rows
#     df = pd.read_csv(
#         file_path,
#         skiprows=4,
#         sep=r"\s+|,",  # Split on whitespace or commas
#         engine="python"
#     )

#     # Extract Frame from column 1 (second column)
#     Frame = df.iloc[:, 0]

#     # Calculate the time for each frame based on a 200 Hz frame capture rate
#     time = Frame / 200  # Time in seconds

#     # Define a function to extract X, Y, Z, VX, VY, VZ, AX, AY, AZ, MX, MVX, MAX for a given prefix and column indices
#     def extract_columns(prefix, indices):
#         return {
#             f"{prefix}_X": df.iloc[:, indices[0]],
#             f"{prefix}_Y": df.iloc[:, indices[1]],
#             f"{prefix}_Z": df.iloc[:, indices[2]],
#             f"{prefix}_VX": df.iloc[:, indices[3]],
#             f"{prefix}_VY": df.iloc[:, indices[4]],
#             f"{prefix}_VZ": df.iloc[:, indices[5]],
#             f"{prefix}_AX": df.iloc[:, indices[6]],
#             f"{prefix}_AY": df.iloc[:, indices[7]],
#             f"{prefix}_AZ": df.iloc[:, indices[8]],
#             f"{prefix}_radial_pos": df.iloc[:, indices[9]], # radial position; the distance from the origin 
#             f"{prefix}_radial_vel": df.iloc[:, indices[10]], # radial velocity; How fast is the distance from the origin changing over time
#             f"{prefix}_radial_acc": df.iloc[:, indices[11]],# radial acceleration
#         }

#     # Define the column indices for each prefix
#     column_indices = {
#         "C7": [2, 3, 4, 74, 75, 76, 146, 147, 148, 218, 242, 266],
#         "T10": [5, 6, 7, 77, 78, 79, 149, 150, 151, 219, 243, 267],
#         "CLAV": [8, 9, 10, 80, 81, 82, 152, 153, 154, 220, 244, 268],
#         "STRN": [11, 12, 13, 83, 84, 85, 155, 156, 157, 221, 245, 269],
#         "LSHO": [14, 15, 16, 86, 87, 88, 158, 159, 160, 222, 246, 270],
#         "LUPA": [17, 18, 19, 89, 90, 91, 161, 162, 163, 223, 247, 271],
#         "LUPB": [20, 21, 22, 92, 93, 94, 164, 165, 166, 224, 248, 272],
#         "LUPC": [23, 24, 25, 95, 96, 97, 167, 168, 169, 225, 249, 273],
#         "LELB": [26, 27, 28, 98, 99, 100, 170, 171, 172, 226, 250, 274],
#         "LMEP": [29, 30, 31, 101, 102, 103, 173, 174, 175, 227, 251, 275],
#         "LWRA": [32, 33, 34, 104, 105, 106, 176, 177, 178, 228, 252, 276],
#         "LWRB": [35, 36, 37, 107, 108, 109, 179, 180, 181, 229, 253, 277],
#         "LFRA": [38, 39, 40, 110, 111, 112, 182, 183, 184, 230, 254, 278],
#         "LFIN": [41, 42, 43, 113, 114, 115, 185, 186, 187, 231, 255, 279],
#         "RSHO": [44, 45, 46, 116, 117, 118, 188, 189, 190, 232, 256, 280],
#         "RUPA": [47, 48, 49, 119, 120, 121, 191, 192, 193, 233, 257, 281],
#         "RUPB": [50, 51, 52, 122, 123, 124, 194, 195, 196, 234, 258, 282],
#         "RUPC": [53, 54, 55, 125, 126, 127, 197, 198, 199, 235, 259, 283],
#         "RELB": [56, 57, 58, 128, 129, 130, 200, 201, 202, 236, 260, 284],
#         "RMEP": [59, 60, 61, 131, 132, 133, 203, 204, 205, 237, 261, 285],
#         "RWRA": [62, 63, 64, 134, 135, 136, 206, 207, 208, 238, 262, 286],
#         "RWRB": [65, 66, 67, 137, 138, 139, 209, 210, 211, 239, 263, 287],
#         "RFRA": [68, 69, 70, 140, 141, 142, 212, 213, 214, 240, 264, 288],
#         "RFIN": [71, 72, 73, 143, 144, 145, 215, 216, 217, 241, 265, 289],
#     }

#     # Extract the traj_data for each prefix
#     traj_data = {}
#     for prefix, indices in column_indices.items():
#         traj_data.update(extract_columns(prefix, indices))


#     return traj_data, Frame, time

# # --- CSV_TO_TRAJ_DATA ---
# def CSV_To_traj_data(file_path, marker_name):
#     """
#     Reads a CSV file containing trajectory data, processes it, and extracts specific columns based on predefined indices.

#     Args:
#         file_path (str): Path to the CSV file.
#         marker_name (str): The marker name to extract data for.

#     Returns:
#         dict: A dictionary containing extracted data for the specified marker.
#     """
#     # Define the column indices for each prefix
#     column_indices = {
#         "C7": [2, 3, 4, 74, 75, 76, 146, 147, 148, 218, 242, 266],
#         "T10": [5, 6, 7, 77, 78, 79, 149, 150, 151, 219, 243, 267],
#         "CLAV": [8, 9, 10, 80, 81, 82, 152, 153, 154, 220, 244, 268],
#         "STRN": [11, 12, 13, 83, 84, 85, 155, 156, 157, 221, 245, 269],
#         "LSHO": [14, 15, 16, 86, 87, 88, 158, 159, 160, 222, 246, 270],
#         "LUPA": [17, 18, 19, 89, 90, 91, 161, 162, 163, 223, 247, 271],
#         "LUPB": [20, 21, 22, 92, 93, 94, 164, 165, 166, 224, 248, 272],
#         "LUPC": [23, 24, 25, 95, 96, 97, 167, 168, 169, 225, 249, 273],
#         "LELB": [26, 27, 28, 98, 99, 100, 170, 171, 172, 226, 250, 274],
#         "LMEP": [29, 30, 31, 101, 102, 103, 173, 174, 175, 227, 251, 275],
#         "LWRA": [32, 33, 34, 104, 105, 106, 176, 177, 178, 228, 252, 276],
#         "LWRB": [35, 36, 37, 107, 108, 109, 179, 180, 181, 229, 253, 277],
#         "LFRA": [38, 39, 40, 110, 111, 112, 182, 183, 184, 230, 254, 278],
#         "LFIN": [41, 42, 43, 113, 114, 115, 185, 186, 187, 231, 255, 279],
#         "RSHO": [44, 45, 46, 116, 117, 118, 188, 189, 190, 232, 256, 280],
#         "RUPA": [47, 48, 49, 119, 120, 121, 191, 192, 193, 233, 257, 281],
#         "RUPB": [50, 51, 52, 122, 123, 124, 194, 195, 196, 234, 258, 282],
#         "RUPC": [53, 54, 55, 125, 126, 127, 197, 198, 199, 235, 259, 283],
#         "RELB": [56, 57, 58, 128, 129, 130, 200, 201, 202, 236, 260, 284],
#         "RMEP": [59, 60, 61, 131, 132, 133, 203, 204, 205, 237, 261, 285],
#         "RWRA": [62, 63, 64, 134, 135, 136, 206, 207, 208, 238, 262, 286],
#         "RWRB": [65, 66, 67, 137, 138, 139, 209, 210, 211, 239, 263, 287],
#         "RFRA": [68, 69, 70, 140, 141, 142, 212, 213, 214, 240, 264, 288],
#         "RFIN": [71, 72, 73, 143, 144, 145, 215, 216, 217, 241, 265, 289],
#     }

#     if marker_name not in column_indices:
#         raise ValueError(f"Marker name '{marker_name}' not found in column indices.")

#     # Read only the required columns for the specified marker
#     indices = column_indices[marker_name]
#     cols_to_read = [0] + indices  # Include the Frame column (index 0)
#     df = pd.read_csv(
#         file_path,
#         skiprows=4,
#         usecols=cols_to_read,
#         sep=r"\s+|,",  # Split on whitespace or commas
#         engine="python",
#     )

#     # Extract Frame and calculate time
#     Frame = df.iloc[:, 0]
#     time = Frame / 200  # Time in seconds

#     # Extract the marker data
#     traj_data = {
#         f"{marker_name}_X": df.iloc[:, 1],
#         f"{marker_name}_Y": df.iloc[:, 2],
#         f"{marker_name}_Z": df.iloc[:, 3],
#         f"{marker_name}_VX": df.iloc[:, 4],
#         f"{marker_name}_VY": df.iloc[:, 5],
#         f"{marker_name}_VZ": df.iloc[:, 6],
#         f"{marker_name}_AX": df.iloc[:, 7],
#         f"{marker_name}_AY": df.iloc[:, 8],
#         f"{marker_name}_AZ": df.iloc[:, 9],
#         f"{marker_name}_radial_pos": df.iloc[:, 10],
#         f"{marker_name}_radial_vel": df.iloc[:, 11],
#         f"{marker_name}_radial_acc": df.iloc[:, 12],
#     }

#     return traj_data, Frame, time

