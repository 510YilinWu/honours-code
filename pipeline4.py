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
from scipy.stats import chisquare
from scipy.stats import circmean, rayleigh

import pandas as pd
import seaborn as sns

import pingouin as pg

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
# # # -------------------------------------------------------------------------------------------------------------------

# PART 1: CHECK IF DATA PROCESSING IS DONE AND LOAD RESULTS
# --- CHECK CALIBRATION FOLDERS FOR PICKLE FILES ---
utils4.check_calibration_folders_for_pickle(All_dates, tBBT_Image_folder)

# --- LOAD ALL SUBJECTS' tBBT ERROR FROM IMAGE, SAVE AS pickle file---
Block_Distance = utils4.load_selected_subject_errors(All_dates, DataProcess_folder)

# --- LOAD RESULTS FROM PICKLE FILE "processed_results.pkl" ---
results = utils1.load_selected_subject_results(All_dates, DataProcess_folder)

# # # -------------------------------------------------------------------------------------------------------------------

# Specify the path to the pickle file
pickle_file = "/Users/yilinwu/Desktop/honours data/DataProcess/All_Subject_tBBTs_errors.pkl"
cmap_choice = LinearSegmentedColormap.from_list("WhiteGreenBlue", ["white", "green", "blue"], N=256)

# Load the pickle file without using a function
with open(pickle_file, "rb") as file:
    All_Subject_tBBTs_errors = pickle.load(file)

### Combine data from both hands and plot combined density heatmap with grid markers
def plot_combined_xy_density(All_Subject_tBBTs_errors, cmap_choice):
    """
    Combines x and y data from both hands in the errors dictionary, plots a 2D density heatmap,
    and overlays grid markers as black crosses.
    
    Parameters:
        All_Subject_tBBTs_errors (dict): Dictionary containing error data.
        cmap_choice: A valid matplotlib colormap.
    """
    import matplotlib.pyplot as plt

    combined_xs, combined_ys = [], []

    for h in ['left', 'right']:
        xs, ys = [], []
        # Loop over all (subject, hand) keys in the errors dictionary
        for key in All_Subject_tBBTs_errors:
            subject_key, hand_key = key
            if hand_key != h:
                continue
            # For each trial in the subject-hand entry, extract the p3_block2 data points
            for trial in All_Subject_tBBTs_errors[key]:
                trial_data = All_Subject_tBBTs_errors[key][trial]
                # Get the 3D points from p3_block2 (if available)
                p3_data = trial_data.get('p3_block2', None)
                if p3_data is None:
                    continue
                p3_data = np.array(p3_data)
                # Ensure data is in (n_points, 3) format
                if p3_data.ndim == 2 and p3_data.shape[0] == 3:
                    p3_data = p3_data.T
                elif p3_data.ndim == 1:
                    p3_data = p3_data.reshape(-1, 3)
                if p3_data.size == 0:
                    continue

                # Check for invalid x-values based on hand
                if hand_key == 'left':
                    if np.any(p3_data[:, 0] <= -50):
                        invalid_idx = np.where(p3_data[:, 0] <= -50)[0]
                        print(f"Error: Subject '{subject_key}', hand '{hand_key}', trial '{trial}' has x values <= -50 at indices {invalid_idx} with values {p3_data[invalid_idx, 0]}")
                elif hand_key == 'right':
                    if np.any(p3_data[:, 0] >= 0):
                        invalid_idx = np.where(p3_data[:, 0] >= 0)[0]
                        print(f"Error: Subject '{subject_key}', hand '{hand_key}', trial '{trial}' has non-negative x values at indices {invalid_idx} with values {p3_data[invalid_idx, 0]}")
                xs.extend(p3_data[:, 0])
                ys.extend(p3_data[:, 1])
        
        if len(xs) > 0 and len(ys) > 0:
            combined_xs.extend(xs)
            combined_ys.extend(ys)
        else:
            print(f"No data available for {h} hand.")

    combined_xs = np.array(combined_xs)
    combined_ys = np.array(combined_ys)
    print(len(combined_xs), len(combined_ys))
    print("Combined Data: X max:", np.max(combined_xs), "X min:", np.min(combined_xs),
            "Y max:", np.max(combined_ys), "Y min:", np.min(combined_ys))

    if combined_xs.size == 0 or combined_ys.size == 0:
        print("No combined data available.")
    else:
        fig, ax = plt.subplots(figsize=(12, 5))
        hb = ax.hist2d(combined_xs, combined_ys, bins=50, cmap=cmap_choice)
        fig.colorbar(hb[3], ax=ax)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title("Right                                                        Left")
        
        # Define grid markers to overlay as black crosses
        grid_xxR = np.array([[ 12.5       ,  66.92857143, 121.35714286, 175.78571429],
                                [ 12.5       ,  66.92857143, 121.35714286, 175.78571429],
                                [ 12.5       ,  66.92857143, 121.35714286, 175.78571429],
                                [ 12.5       ,  66.92857143, 121.35714286, 175.78571429]])
    
        grid_yyR = np.array([[ 12.5       ,  12.5       ,  12.5       ,  12.5       ],
                                [ 66.92857143,  66.92857143,  66.92857143,  66.92857143],
                                [121.35714286, 121.35714286, 121.35714286, 121.35714286],
                                [175.78571429, 175.78571429, 175.78571429, 175.78571429]])
    
        grid_xxL = np.array([[-246.5       , -192.07142857, -137.64285714,  -83.21428571],
                                [-246.5       , -192.07142857, -137.64285714,  -83.21428571],
                                [-246.5       , -192.07142857, -137.64285714,  -83.21428571],
                                [-246.5       , -192.07142857, -137.64285714,  -83.21428571]])
    
        grid_yyL = np.array([[ 12.5       ,  12.5       ,  12.5       ,  12.5       ],
                                [ 66.92857143,  66.92857143,  66.92857143,  66.92857143],
                                [121.35714286, 121.35714286, 121.35714286, 121.35714286],
                                [175.78571429, 175.78571429, 175.78571429, 175.78571429]])
    
        # Overlay grid markers as black crosses with all values subtracted by 12.5
        ax.scatter((grid_xxR - 12.5).flatten(), (grid_yyR - 12.5).flatten(),
                    marker='x', color='black', s=50, linewidths=2)
        ax.scatter((grid_xxL - 12.5).flatten(), (grid_yyL - 12.5).flatten(),
                    marker='x', color='black', s=50, linewidths=2)


    
        plt.tight_layout()
        plt.show()

### Combine 16 blocks data into one for each subject and hand
def Combine_16_blocks(All_Subject_tBBTs_errors):
    """
    For each subject and hand in All_Subject_tBBTs_errors, iterate over all trials,
    extract the 'p3_block2' data and 'blocks_without_points', and compute new coordinates based on
    blockMembership and grid adjustments.
    
    For left-hand entries, grid values from grid_xxR and grid_yyR are used.
    For right-hand entries, grid values from grid_xxL and grid_yyL are used.
    
    Returns:
        dict: Mapping from (subject, hand) to another dict mapping trial index to a list of tuples (new_x, new_y, block)
    """
    # Block membership mapping (order of blocks)
    blockMembership = [12, 1, 14, 3, 8, 13, 2, 15, 0, 5, 6, 11, 4, 9, 7, 10]

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
        results[key] = {}  # dictionary to store per trial new block coordinates
        
        # Loop over all trial indices for the given (subject, hand)
        for trial in sorted(All_Subject_tBBTs_errors[key].keys()):
            trial_entry = All_Subject_tBBTs_errors[key][trial]
            # Get p3_block2 data and blocks_without_points
            p3_data = trial_entry.get('p3_block2', None)
            blocks_without_points = trial_entry.get('blocks_without_points', None)
            if p3_data is None or blocks_without_points is None:
                continue

            # Ensure p3_data is a numpy array and properly shaped as (n_points, 3)
            p3_data = np.array(p3_data)
            if p3_data.ndim == 2 and p3_data.shape[0] == 3:
                p3_data = p3_data.T
            elif p3_data.ndim == 1:
                p3_data = p3_data.reshape(-1, 3)
            if p3_data.size == 0:
                continue
            
            # Choose grid arrays based on hand: 'left' uses grid_xxR/yyR, 'right' uses grid_xxL/yyL
            if hand.lower() == 'left':
                grid_x = (grid_xxR - 12.5).flatten()
                grid_y = (grid_yyR - 12.5).flatten()
            elif hand.lower() == 'right':
                grid_x = (grid_xxL - 12.5).flatten()[::-1]
                grid_y = (grid_yyL - 12.5).flatten()
            else:
                continue

            data_index = 0
            new_coords = []  # List to store the new coordinates for this trial
            
            # Loop over each of the 16 blocks
            for i in range(16):
                current_block = blockMembership[i]
                # If this block was not marked as missing
                if current_block not in blocks_without_points:
                    # Get block_points from the p3_data
                    block_points = p3_data[data_index]
                    data_index += 1
                    # print(f"Subject: {subject}, Hand: {hand}, Trial: {trial}, Block: {current_block}, "
                    #       f"Block Points: {block_points}, Grid Point: ({grid_x[current_block]}, {grid_y[current_block]})")
                    new_x = block_points[0] - grid_x[current_block]
                    new_y = block_points[1] - grid_y[current_block]                   
                    new_coords.append((new_x, new_y, blockMembership[i]))
            results[key][trial] = new_coords
        # print(f"Processed key: {key}")
    
    return results

### Plot left and right hand 16 blocks as one density histogram, overlaying multiple trials
def plot_left_right_hand_new_coordinates_density(Combine_blocks, cmap=None, bins=60, xlim=(-15, 15), ylim=(-15, 15)):
    """
    Plots 2D density histograms (hist2d) for new coordinates of left-hand and right-hand data
    as subplots, overlaying multiple trial data. 0.0 is kept at the center of the view.

    Parameters:
        Combine_blocks (dict): Dictionary mapping (subject, hand) to either:
                               - a list of tuples (new_x, new_y, block), or
                               - a dict mapping trial identifiers to lists of tuples (new_x, new_y, block).
        cmap: A matplotlib colormap. If None, defaults to a white-to-red colormap.
        bins (int): Number of bins for the histogram.
        xlim (tuple): x-axis limits (centered at 0).
        ylim (tuple): y-axis limits (centered at 0).
    """
    import matplotlib.pyplot as plt

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("WhiteRed", ["white", "red"], N=256)

    xs_left, ys_left = [], []  # Right hand plot (will show left-hand data)
    xs_right, ys_right = [], []  # Left hand plot (will show right-hand data)

    # Loop through Combine_blocks which may contain multiple trials per key.
    for key, data in Combine_blocks.items():
        subject, hand = key
        # If multiple trials, data is a dictionary, otherwise a list
        if isinstance(data, dict):
            for trial in data:
                trial_coords = data[trial]
                for new_x, new_y, block in trial_coords:
                    if hand.lower() == 'left':
                        xs_left.append(new_x)
                        ys_left.append(new_y)
                    elif hand.lower() == 'right':
                        xs_right.append(new_x)
                        ys_right.append(new_y)
        else:
            for new_x, new_y, block in data:
                if hand.lower() == 'left':
                    xs_left.append(new_x)
                    ys_left.append(new_y)
                elif hand.lower() == 'right':
                    xs_right.append(new_x)
                    ys_right.append(new_y)  

    xs_left = np.array(xs_left)
    ys_left = np.array(ys_left)
    xs_right = np.array(xs_right)
    ys_right = np.array(ys_right)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    hb1 = ax1.hist2d(xs_right, ys_right, bins=bins, cmap=cmap)
    ax1.plot(0, 0, 'ko', markersize=8)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.grid(False)
    fig.colorbar(hb1[3], ax=ax1)
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_title("Right")

    hb2 = ax2.hist2d(xs_left, ys_left, bins=bins, cmap=cmap)
    ax2.plot(0, 0, 'ko', markersize=8)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.grid(False)
    fig.colorbar(hb2[3], ax=ax2)
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.set_title("Left")

    plt.tight_layout()
    plt.show()

### Plot left and right hand 16 blocks as one polar histograms (rose diagrams)
def plot_left_right_hand_polar_histogram(Combine_blocks, cmap_choice):
    """
    Plots polar histograms (rose diagrams) for left-hand and right-hand new coordinates as subplots.
    If for a subject-hand key there are multiple trials (i.e. data is a dict of trial keys mapping to lists of tuples),
    this function overlays all the trials into one histogram.
    
    Parameters:
        Combine_blocks (dict): Dictionary mapping keys (subject, hand) either to a list of tuples (new_x, new_y, block)
                               or to a dict mapping trial identifiers to lists of tuples.
        cmap_choice: A matplotlib colormap to map histogram density.
    """
    import matplotlib.pyplot as plt

    # Combine coordinates across all trials for each hand
    xs_left, ys_left, xs_right, ys_right = [], [], [], []

    # Collect coordinates
    for key, data in Combine_blocks.items():
        _, hand = key
        hand_lower = hand.lower()
        if isinstance(data, dict):
            coords_iter = [pt for trial in data.values() for pt in trial]
        else:
            coords_iter = data

        for new_x, new_y, _ in coords_iter:
            if hand_lower == 'left':
                xs_left.append(new_x)
                ys_left.append(new_y)
            elif hand_lower == 'right':
                xs_right.append(new_x)
                ys_right.append(new_y)

    xs_left, ys_left = np.array(xs_left), np.array(ys_left)
    xs_right, ys_right = np.array(xs_right), np.array(ys_right)

    # Convert Cartesian coordinates to polar: compute radius and angles.
    r_left = np.sqrt(xs_left**2 + ys_left**2)
    theta_left = np.arctan2(ys_left, xs_left)

    r_right = np.sqrt(xs_right**2 + ys_right**2)
    theta_right = np.arctan2(ys_right, xs_right)

    num_bins = 20

    _, axes = plt.subplots(1, 2, subplot_kw=dict(projection='polar'), figsize=(12, 6))

    # Left hand polar histogram
    counts_left, bin_edges_left = np.histogram(theta_left, bins=num_bins, weights=r_left)
    width_left = bin_edges_left[1] - bin_edges_left[0]
    max_left = counts_left.max() if counts_left.max() != 0 else 1
    normalized_counts_left = counts_left / max_left

    axes[1].bar(bin_edges_left[:-1], counts_left, width=width_left, bottom=0.0,
                color=[cmap_choice(val) for val in normalized_counts_left],
                edgecolor='k', alpha=0.75)
    axes[1].set_title("Left Hand")

    # Right hand polar histogram
    counts_right, bin_edges_right = np.histogram(theta_right, bins=num_bins, weights=r_right)
    width_right = bin_edges_right[1] - bin_edges_right[0]
    max_right = counts_right.max() if counts_right.max() != 0 else 1
    normalized_counts_right = counts_right / max_right

    axes[0].bar(bin_edges_right[:-1], counts_right, width=width_right, bottom=0.0,
                color=[cmap_choice(val) for val in normalized_counts_right],
                edgecolor='k', alpha=0.75)
    axes[0].set_title("Right Hand")

    plt.tight_layout()
    plt.show()


plot_combined_xy_density(All_Subject_tBBTs_errors, cmap_choice)
Combine_blocks = Combine_16_blocks(All_Subject_tBBTs_errors)  # Now processes all trials instead of only trial index 1
plot_left_right_hand_new_coordinates_density(Combine_blocks, cmap_choice)
plot_left_right_hand_polar_histogram(Combine_blocks, cmap_choice)

### Analyze distribution (uniformity, bias, quadrants) and plot polar histograms (rose diagrams)
def analyze_and_plot_left_right(Combine_blocks, cmap_choice, num_bins=20):
    """
    Analyze distribution (uniformity, bias, quadrants) and plot polar histograms (rose diagrams).

    Performs chi-square tests for uniformity across quadrants, left vs right, top vs bottom.
    Computes mean resultant vector direction and Rayleigh test for circular uniformity.

    Parameters:
        Combine_blocks (dict): (subject, hand) -> list of (new_x, new_y, block) or dict of trials
        cmap_choice: matplotlib colormap
        num_bins: number of bins for polar histogram
    """

    xs_left, ys_left, xs_right, ys_right = [], [], [], []

    # Collect coordinates
    for key, data in Combine_blocks.items():
        _, hand = key
        hand_lower = hand.lower()
        if isinstance(data, dict):
            coords_iter = [pt for trial in data.values() for pt in trial]
        else:
            coords_iter = data

        for new_x, new_y, _ in coords_iter:
            if hand_lower == 'left':
                xs_left.append(new_x)
                ys_left.append(new_y)
            elif hand_lower == 'right':
                xs_right.append(new_x)
                ys_right.append(new_y)

    xs_left, ys_left = np.array(xs_left), np.array(ys_left)
    xs_right, ys_right = np.array(xs_right), np.array(ys_right)

    # Convert to polar
    r_left = np.sqrt(xs_left**2 + ys_left**2)
    theta_left = np.arctan2(ys_left, xs_left)

    r_right = np.sqrt(xs_right**2 + ys_right**2)
    theta_right = np.arctan2(ys_right, xs_right)

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

        # Left vs Right counts
        lr_counts = [np.sum(xs < 0), np.sum(xs >= 0)]
        chi_result = chisquare(lr_counts)
        results['left_vs_right_statistic'] = chi_result.statistic
        results['left_vs_right_p'] = chi_result.pvalue

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

    stats_left = circular_chi_tests(xs_left, ys_left, theta_left)
    stats_right = circular_chi_tests(xs_right, ys_right, theta_right)

    # -------------------------
    #  Plot rose diagrams with mean direction arrow
    # -------------------------
    fig, axes = plt.subplots(1, 2, subplot_kw=dict(projection='polar'), figsize=(12, 6))

    # Left hand
    counts_left, bin_edges_left = np.histogram(theta_left, bins=num_bins, weights=r_left)
    width_left = bin_edges_left[1] - bin_edges_left[0]
    max_left = counts_left.max() if counts_left.max() != 0 else 1
    norm_left = counts_left / max_left

    axes[1].bar(bin_edges_left[:-1], counts_left, width=width_left, bottom=0.0,
                color=[cmap_choice(val) for val in norm_left],
                edgecolor='k', alpha=0.75)
    axes[1].set_title("Left Hand", y=1.05)
    # Draw mean direction arrow
    axes[1].arrow(stats_left['mean_direction_rad'], 0, 0, max_left, width=0.05, color='red', alpha=0.8)

    # Right hand
    counts_right, bin_edges_right = np.histogram(theta_right, bins=num_bins, weights=r_right)
    width_right = bin_edges_right[1] - bin_edges_right[0]
    max_right = counts_right.max() if counts_right.max() != 0 else 1
    norm_right = counts_right / max_right

    axes[0].bar(bin_edges_right[:-1], counts_right, width=width_right, bottom=0.0,
                color=[cmap_choice(val) for val in norm_right],
                edgecolor='k', alpha=0.75)
    axes[0].set_title("Right Hand", y=1.05)
    # Draw mean direction arrow
    axes[0].arrow(stats_right['mean_direction_rad'], 0, 0, max_right, width=0.05, color='red', alpha=0.8)

    plt.tight_layout()
    plt.show()

    return {"left": stats_left, "right": stats_right}

stats = analyze_and_plot_left_right(Combine_blocks, cmap_choice)

print("Left hand")
for key, value in stats['left'].items():
    print(f"{key}: {value:.6f}")
print("\nRight hand")
for key, value in stats['right'].items():
    print(f"{key}: {value:.6f}")

subject = '08/07/DA'
hand = 'left'
target_file = '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/08/07/DA/DA_tBBT18.csv'

# Get all file keys in the trial dictionary (results[subject][hand][1])
file_keys = list(results[subject][hand][1].keys())

# Find the index of the target file in the file keys list
target_index = file_keys.index(target_file)
print("Index of target file:", target_index)

### Plot p3_box2 and p3_block2 coordinates in a 3D scatter plot for a specific subject, hand, and trial
utils4.plot_p3_coordinates(All_Subject_tBBTs_errors, subject='07/22/HW', hand='left', trial_index=1)

### Plot hand trajectory with velocity-coded coloring and highlighted segments
utils4.plot_trajectory(results, subject='07/22/HW', hand='left', trial=1,
                file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv',
                overlay_trial=0, velocity_segment_only=True, plot_mode='segment')

### Combine hand trajectory and error coordinates in a single 3D plot
utils4.combined_plot_trajectory_and_errors(results, All_Subject_tBBTs_errors,
                                      subject='07/22/HW', hand='left',
                                      trial=1, trial_index=1,
                                      file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv',
                                      overlay_trial=0, velocity_segment_only=True, plot_mode='segment')

# # # -------------------------------------------------------------------------------------------------------------------

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
# test_windows_2_1: From movement start to velocity peak (focuses on movement buildup)
# test_windows_2_2: From velocity peak to movement end (focuses on movement deceleration)
# test_windows_3: Symmetric window around velocity peak (captures activity before and after peak) (500 ms total)
# test_windows_4: 100 ms before velocity peak (captures lead-up dynamics)
# test_windows_5: 100 ms after velocity peak (captures immediate post-peak activity)
# test_windows_6: Custom time window centered around the midpoint of each segment 
test_windows_1, test_windows_2_1, test_windows_2_2, test_windows_3, test_windows_4, test_windows_5, test_windows_6 = utils2.define_time_windows(reach_speed_segments, reach_metrics, fs=200, window_size=0.25)


def compute_test_window_7(results, reach_speed_segments, reach_metrics):
    # Helper function for test window 7:
    def get_onset_termination(speed, seg_start, seg_end, threshold):
        # Find index of peak velocity within the segment.
        peak_index = seg_start + int(np.argmax(speed[seg_start:seg_end]))
        # Search backwards from the peak until speed falls below the threshold.
        onset = seg_start
        for j in range(peak_index, seg_start - 1, -1):
            if speed[j] < threshold:
                onset = j + 1
                break
        # Search forwards from the peak until speed falls below the threshold.
        termination = seg_end
        for j in range(peak_index, seg_end):
            if speed[j] < threshold:
                termination = j
                break
        return onset, termination

    # Test window 7: Onset is the first frame where the speed goes above 5% of the maximum velocity
    #         and termination is the first frame where the speed drops below 5% of the maximum velocity.
    test_windows_7 = {
        date: {
            hand: {
                trial: [
                    # For each segment, determine onset and termination using the speed array.
                    # Marker: "RFIN" if hand=="right", else "LFIN".
                    # Get the speed series from the results dictionary.
                    # Use the pre-calculated maximum velocity for the segment from reach_metrics.
                    (lambda seg, i: get_onset_termination(
                        results[date][hand][1][trial]['traj_space']["RFIN" if hand == "right" else "LFIN"][1],
                        seg[0],
                        seg[1],
                        0.1 * reach_metrics['reach_v_peaks'][date][hand][trial][i]
                    ))(segment, i)
                    for i, segment in enumerate(reach_speed_segments[date][hand][trial])
                ]
                for trial in reach_speed_segments[date][hand]
            }
            for hand in reach_speed_segments[date]
        }
        for date in reach_speed_segments
    }
    return test_windows_7

test_windows_7 = compute_test_window_7(results, reach_speed_segments, reach_metrics)


# --- CALCULATE REACH METRICS SPECIFIC TO TIME WINDOW ---
# reach_acc_peaks
# reach_jerk_peaks
# reach_LDLJ
reach_TW_metrics_test_windows_1 = utils2.calculate_reach_metrics_for_time_windows_Normalizing(test_windows_1, results)
reach_TW_metrics_test_windows_2_1 = utils2.calculate_reach_metrics_for_time_windows_Normalizing(test_windows_2_1, results)
reach_TW_metrics_test_windows_2_2 = utils2.calculate_reach_metrics_for_time_windows_Normalizing(test_windows_2_2, results)
reach_TW_metrics_test_windows_3 = utils2.calculate_reach_metrics_for_time_windows_Normalizing(test_windows_3, results)
reach_TW_metrics_test_windows_4 = utils2.calculate_reach_metrics_for_time_windows_Normalizing(test_windows_4, results)
reach_TW_metrics_test_windows_5 = utils2.calculate_reach_metrics_for_time_windows_Normalizing(test_windows_5, results)
reach_TW_metrics_test_windows_6 = utils2.calculate_reach_metrics_for_time_windows_Normalizing(test_windows_7, results)

# --- CALCULATE SPARC FOR EACH TEST WINDOW FOR ALL DATES, HANDS, AND TRIALS ---
reach_sparc_test_windows_1_Normalizing = utils2.calculate_reach_sparc_Normalizing(test_windows_1, results)
reach_sparc_test_windows_2_1_Normalizing = utils2.calculate_reach_sparc_Normalizing(test_windows_2_1, results)
reach_sparc_test_windows_2_2_Normalizing = utils2.calculate_reach_sparc_Normalizing(test_windows_2_2, results)
reach_sparc_test_windows_3_Normalizing = utils2.calculate_reach_sparc_Normalizing(test_windows_3, results)
reach_sparc_test_windows_4_Normalizing = utils2.calculate_reach_sparc_Normalizing(test_windows_4, results)
reach_sparc_test_windows_5_Normalizing = utils2.calculate_reach_sparc_Normalizing(test_windows_5, results)
reach_sparc_test_windows_6_Normalizing = utils2.calculate_reach_sparc_Normalizing(test_windows_7, results)

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





def plot_sbbt_boxplot(sBBTResult):
    """
    Plot a boxplot of sBBT scores by hand (Non-dominant vs Dominant) on a given axis.
    It overlays a swarmplot and draws dashed lines connecting each subject's pair of scores.
    
    Parameters:
        ax: matplotlib.axes.Axes - the axis on which to plot.
        sBBTResult: DataFrame - indexed by subjects with columns "non_dominant" and "dominant".
    """
    # Create a figure and axis for the boxplot
    fig, ax = plt.subplots(figsize=(6, 5))
    # Prepare a DataFrame for sBBT scores per subject.
    df_scores = pd.DataFrame({
        "Subject": sBBTResult.index,
        "Non-dominant": sBBTResult["non_dominant"],
        "Dominant": sBBTResult["dominant"]
    })
    
    # Melt the DataFrame to long format.
    df_melt = df_scores.melt(id_vars="Subject", var_name="Hand", value_name="sBBT Score")
    
    # Define the order for hands.
    order = ["Non-dominant", "Dominant"]
    
    # Plot the boxplot and swarmplot.
    sns.boxplot(x="Hand", y="sBBT Score", data=df_melt, palette="Set2", order=order, ax=ax)
    sns.swarmplot(x="Hand", y="sBBT Score", data=df_melt, color="black", size=5, alpha=0.8, order=order, ax=ax)
    
    # Set title and labels.
    # ax.set_title("Box Plot of sBBT Scores by Hand")
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel("Hand", fontsize=20)
    ax.set_ylabel("sBBT Score", fontsize=20)
    
    # # Draw dashed lines connecting each subject's non-dominant and dominant scores.
    # for subject in df_scores["Subject"]:
    #     nd_val = df_scores.loc[df_scores["Subject"] == subject, "Non-dominant"].values[0]
    #     d_val = df_scores.loc[df_scores["Subject"] == subject, "Dominant"].values[0]
    #     ax.plot([0, 1], [nd_val, d_val], color="gray", linestyle="--", linewidth=1, alpha=0.7)

plot_sbbt_boxplot(sBBTResult)

# -------------------------------------------------------------------------------------------------------------------
# Prepare long-format DataFrame using sBBTResult values
df = pd.DataFrame({
    'Subject': list(range(1, len(sBBTResult) + 1)) * 2,
    'Test': ['Test1'] * len(sBBTResult) + ['Test2'] * len(sBBTResult),
    'Right': list(sBBTResult['Right']) + list(sBBTResult['Right.1']),
    'Left': list(sBBTResult['Left']) + list(sBBTResult['Left.1'])
})

# Compute ICC for Right hand scores
icc_right = pg.intraclass_corr(data=df, targets='Subject', raters='Test', ratings='Right')
print("ICC for Right hand:")
print(icc_right[['Type', 'ICC', 'F', 'pval', 'CI95%']])

# Compute ICC for Left hand scores
icc_left = pg.intraclass_corr(data=df, targets='Subject', raters='Test', ratings='Left')
print("ICC for Left hand:")
print(icc_left[['Type', 'ICC', 'F', 'pval', 'CI95%']])
# -------------------------------------------------------------------------------------------------------------------

import scipy.stats as stats

# Calculate Spearman correlation for the Right hand scores (Test1 vs Test2)
right_corr, right_p = stats.spearmanr(sBBTResult['Right'], sBBTResult['Right.1'])
print("Spearman correlation for Right hand:", right_corr, "p-value:", right_p)

# Calculate Spearman correlation for the Left hand scores (Test1 vs Test2)
left_corr, left_p = stats.spearmanr(sBBTResult['Left'], sBBTResult['Left.1'])
print("Spearman correlation for Left hand:", left_corr, "p-value:", left_p)

# --------------------------------------------------------------------
# plot Bland-Altman plots for Left and Right hand scores
def plot_bland_altman(sBBTResult, show_value_in_legend=True):
    import matplotlib.pyplot as plt

    # Prepare data for Left hand
    left1 = pd.Series(sBBTResult['Left'])
    left2 = pd.Series(sBBTResult['Left.1'])
    mean_left = (left1 + left2) / 2
    diff_left = left1 - left2
    md_left = diff_left.mean()
    sd_left = diff_left.std()

    # Prepare data for Right hand
    right1 = pd.Series(sBBTResult['Right'])
    right2 = pd.Series(sBBTResult['Right.1'])
    mean_right = (right1 + right2) / 2
    diff_right = right1 - right2
    md_right = diff_right.mean()
    sd_right = diff_right.std()

    if show_value_in_legend:
        label_md_left = f'Mean Diff = {md_left:.2f}'
        label_plus_left = f'+1.96 SD = {(md_left + 1.96 * sd_left):.2f}'
        label_minus_left = f'-1.96 SD = {(md_left - 1.96 * sd_left):.2f}'

        label_md_right = f'Mean Diff = {md_right:.2f}'
        label_plus_right = f'+1.96 SD = {(md_right + 1.96 * sd_right):.2f}'
        label_minus_right = f'-1.96 SD = {(md_right - 1.96 * sd_right):.2f}'
    else:
        # When not showing values in legend, only the right hand plot will show generic legend labels.
        label_md_left = label_plus_left = label_minus_left = None
        label_md_right = 'Mean Diff'
        label_plus_right = '+1.96 SD'
        label_minus_right = '-1.96 SD'

    print(
        f"Left Hand - Mean Difference: {md_left:.2f} ± {sd_left:.2f} "
        f"(95% CI: {md_left - 1.96 * sd_left:.2f} to {md_left + 1.96 * sd_left:.2f})"
    )
    print(
        f"Right Hand - Mean Difference: {md_right:.2f} ± {sd_right:.2f} "
        f"(95% CI: {md_right - 1.96 * sd_right:.2f} to {md_right + 1.96 * sd_right:.2f})"
    )

plot_bland_altman(sBBTResult, show_value_in_legend=False)

# --------------------------------------------------------------------

def plot_sbbt_scores(sBBTResult, figsize=(14, 6)):
    """
    Plots side-by-side boxplots for the right and left hand sBBT scores.
    Each plot overlays each subject's paired Test1 and Test2 scores with a colored line:
      - Green if Test2 > Test1 (improvement)
      - Orange if Test2 equals Test1
      - Red if Test2 < Test1
    sBBT score is defined as number of blocks transferred in one minute (n of blocks).

    Parameters:
        sBBTResult (DataFrame): A DataFrame with index for subjects and columns:
                                'Right', 'Right.1', 'Left', 'Left.1'
        figsize (tuple): Figure size.
    """
    import matplotlib.pyplot as plt
    
    # Create DataFrames for Right and Left hands, preserving the subject index.
    df_right = pd.DataFrame({
        'Subject': sBBTResult.index,
        '1': sBBTResult['Right'],
        '2': sBBTResult['Right.1']
    })
    df_left = pd.DataFrame({
        'Subject': sBBTResult.index,
        '1': sBBTResult['Left'],
        '2': sBBTResult['Left.1']
    })
    
    plt.figure(figsize=figsize)
    
    # Plot for Left hand
    plt.subplot(1, 2, 1)
    melted_left = df_left.melt(id_vars='Subject', value_vars=['1', '2'],
                               var_name='Test', value_name='sBBT Score')
    sns.boxplot(x='Test', y='sBBT Score', data=melted_left, color='lightgray')
    plt.ylabel("sBBT Score (n of blocks)", fontsize=16)
    plt.xticks(fontsize=16)  # Increase xtick label size for left hand plot
    for _, row in df_left.iterrows():
        color = 'green' if row['2'] > row['1'] else ('orange' if row['2'] == row['1'] else 'red')
        plt.plot([0, 1], [row['1'], row['2']],
                 marker='o', color=color, linewidth=1, alpha=0.7)
    plt.title("Left", fontsize=18)    

    # Plot for Right hand
    plt.subplot(1, 2, 2)
    melted_right = df_right.melt(id_vars='Subject', value_vars=['1', '2'],
                                 var_name='Test', value_name='sBBT Score')
    sns.boxplot(x='Test', y='sBBT Score', data=melted_right, color='lightgray')
    plt.ylabel("sBBT Score (n of blocks)", fontsize=16)
    plt.xticks(fontsize=16)  # Increase xtick label size for right hand plot
    # Overlay each subject's paired scores as a line:
    for _, row in df_right.iterrows():
        color = 'green' if row['2'] > row['1'] else ('orange' if row['2'] == row['1'] else 'red')
        plt.plot([0, 1], [row['1'], row['2']],
                 marker='o', color=color, linewidth=1, alpha=0.7)
    plt.title("Right", fontsize=18)
    
    plt.tight_layout()
    plt.show()

plot_sbbt_scores(sBBTResult)

# # Get the value from the "non_dominant" column for the row where Subject is 'CZ'
# value = sBBTResult.loc[sBBTResult["Subject"] == "CZ", "non_dominant"].values
# print("Value:", value)

sBBTResult['Right.1']-sBBTResult['Right']
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
def overlay_timewindow(results, reach_speed_segments_1=None, reach_speed_segments_2=None, 
                       subject="06/19/CZ", hand="right", trial=1,
                       file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/06/19/CZ/CZ_tBBT01.csv',
                       fs=200, target_samples=101, seg_index=0):
    """
    Plots normalized jerk (and other signals) from trajectory data using two different time window segmentations.
    For the selected segment (seg_index), this function extracts the two segments using reach_speed_segments_1 
    and reach_speed_segments_2 and overlays their corresponding signals. The x-axis now displays the actual frame 
    indices from the original trial.

    Parameters:
        results (dict): Dictionary containing trajectory data.
        reach_speed_segments_1, reach_speed_segments_2 (dict or None): Dictionaries with segmentation ranges 
            with structure: reach_speed_segments[subject][hand][file_path] = list of (start_idx, end_idx) tuples.
        subject (str): Subject identifier.
        hand (str): Hand identifier.
        trial (int): Trial index.
        file_path (str): Path to the CSV file.
        fs (int): Sampling rate in Hz.
        target_samples (int): Number of samples for the normalized jerk signal.
        seg_index (int): Index of the segment to plot.

    This function extracts the trajectory data using marker 'RFIN' for right hand or 'LFIN' for left hand,
    computes a normalized jerk signal (via linear interpolation) and plots five overlaid subplots:
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
    
    seg_ranges1 = reach_speed_segments_1[subject][hand][file_path]
    seg_ranges2 = reach_speed_segments_2[subject][hand][file_path]
    
    if seg_index < 0 or seg_index >= len(seg_ranges1) or seg_index >= len(seg_ranges2):
        print("Invalid seg_index provided.")
        return

    # Retrieve the indices for the selected segment from both segmentations.
    start_idx1, end_idx1 = seg_ranges1[seg_index]
    start_idx2, end_idx2 = seg_ranges2[seg_index]

    # Extract the trajectory segments for each segmentation.
    pos_seg1 = position_full[start_idx1:end_idx1]
    vel_seg1 = velocity_full[start_idx1:end_idx1]
    acc_seg1 = acceleration_full[start_idx1:end_idx1]
    jerk_seg1 = jerk_full[start_idx1:end_idx1]

    pos_seg2 = position_full[start_idx2:end_idx2]
    vel_seg2 = velocity_full[start_idx2:end_idx2]
    acc_seg2 = acceleration_full[start_idx2:end_idx2]
    jerk_seg2 = jerk_full[start_idx2:end_idx2]

    # Process Segment 1
    duration1 = len(jerk_seg1) / fs
    t_orig1 = np.linspace(0, duration1, num=len(jerk_seg1))
    t_std1 = np.linspace(0, duration1, num=target_samples)
    warped_jerk1 = np.interp(t_std1, t_orig1, jerk_seg1)
    jerk_squared_integral1 = np.trapezoid(warped_jerk1**2, t_std1)
    vpeak1 = vel_seg1.max()
    dimensionless_jerk1 = (duration1**3 / vpeak1**2) * jerk_squared_integral1
    LDLJ1 = -math.log(abs(dimensionless_jerk1), math.e)

    # Process Segment 2
    duration2 = len(jerk_seg2) / fs
    t_orig2 = np.linspace(0, duration2, num=len(jerk_seg2))
    t_std2 = np.linspace(0, duration2, num=target_samples)
    warped_jerk2 = np.interp(t_std2, t_orig2, jerk_seg2)
    jerk_squared_integral2 = np.trapezoid(warped_jerk2**2, t_std2)
    vpeak2 = vel_seg2.max()
    dimensionless_jerk2 = (duration2**3 / vpeak2**2) * jerk_squared_integral2
    LDLJ2 = -math.log(abs(dimensionless_jerk2), math.e)

    # Use the real frame indices as the x-axis.
    x_vals1 = range(start_idx1, end_idx1)
    x_vals2 = range(start_idx2, end_idx2)

    # Create a figure with 5 subplots to overlay the two segments for each signal.
    fig, axs = plt.subplots(5, 1, figsize=(12, 12))
    
    # Overlay Position
    axs[0].plot(x_vals1, pos_seg1, color='blue', linewidth=2, label='Segment 1')
    axs[0].plot(x_vals2, pos_seg2, color='lime', linestyle='--', linewidth=2, label='Segment 2')
    axs[0].set_title('Position')
    axs[0].legend(loc='upper right')

    # Overlay Velocity
    axs[1].plot(x_vals1, vel_seg1, color='blue', linewidth=2, label='Segment 1')
    axs[1].plot(x_vals2, vel_seg2, color='lime', linestyle='--', linewidth=2, label='Segment 2')
    axs[1].set_title('Velocity')
    # axs[1].legend(loc='upper right')

    # Overlay Acceleration
    axs[2].plot(x_vals1, acc_seg1, color='blue', linewidth=2, label='Segment 1')
    axs[2].plot(x_vals2, acc_seg2, color='lime', linestyle='--', linewidth=2, label='Segment 2')
    axs[2].set_title('Acceleration')
    # axs[2].legend(loc='upper right')

    # Overlay Original Jerk
    axs[3].plot(x_vals1, jerk_seg1, color='blue', linewidth=2, label='Segment 1')
    axs[3].plot(x_vals2, jerk_seg2, color='lime', linestyle='--', linewidth=2, label='Segment 2')
    axs[3].set_title('Original Jerk')
    # axs[3].legend(loc='upper right')
    axs[3].set_xlabel("Frame Index", fontsize=12)

    # Overlay Normalized Jerk with computed LDLJ values
    percent_x1 = np.linspace(0, 100, len(warped_jerk1))
    percent_x2 = np.linspace(0, 100, len(warped_jerk2))
    axs[4].plot(percent_x1, warped_jerk1, linestyle='--', color='blue', linewidth=2,
                label=f'LDLJ: {LDLJ1:.2f}')
    axs[4].plot(percent_x2, warped_jerk2, linestyle='--', color='lime', linewidth=2,
                label=f'LDLJ: {LDLJ2:.2f}')
    
    # Setting tick positions based on the duration of segment 1
    axs[4].set_xticks(np.linspace(0, 100, 6))
    axs[4].set_xticklabels([f"{int(p)}%" for p in np.linspace(0, 100, 6)])
    axs[4].set_title(f'Normalized Jerk - Overlay Segments {seg_index+1}')
    axs[4].legend(loc='upper right')
    axs[4].set_xlabel("Percentage of Segment Duration", fontsize=12)


    plt.tight_layout()
    plt.show()

overlay_timewindow(results, test_windows_1, test_windows_6, subject="07/22/HW", hand="left", trial=1,
                   file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv',
                   fs=200, target_samples=101, seg_index=1)




def plot_3d_trajectory_segment(results, test_windows_1, test_windows_6, subject, hand, trial, file_path, seg_index=0):
    """
    Extracts a 3D trajectory segment based on paired start and end indices from test_windows_1 
    and overlays another segment extracted from test_windows_6 in the same 3D plot.
    
    Parameters:
        results (dict): Dictionary containing trajectory data.
        test_windows_1 (dict): Dictionary with paired start and end indices (first test window).
        test_windows_6 (dict): Dictionary with paired start and end indices (second test window).
        subject (str): Subject identifier.
        hand (str): 'right' or 'left'.
        trial (int): Trial index.
        file_path (str): Path to the CSV file.
        seg_index (int): Index of the segment to extract (default is 0).
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

    # Get the full trajectory data
    traj_data = results[subject][hand][trial][file_path]['traj_data']
    coord_prefix = "RFIN_" if hand == "right" else "LFIN_"
    coord_x_full = np.array(traj_data[coord_prefix + "X"])
    coord_y_full = np.array(traj_data[coord_prefix + "Y"])
    coord_z_full = np.array(traj_data[coord_prefix + "Z"])

    # Get the paired start and end indices for the first test window
    start_idx1, end_idx1 = test_windows_1[subject][hand][file_path][seg_index]
    # Extract the segment from test_windows_1
    traj_x1 = coord_x_full[start_idx1:end_idx1]
    traj_y1 = coord_y_full[start_idx1:end_idx1]
    traj_z1 = coord_z_full[start_idx1:end_idx1]

    # Get the paired start and end indices for the second test window (test_windows_6)
    start_idx6, end_idx6 = test_windows_6[subject][hand][file_path][seg_index]
    # Extract the segment from test_windows_6
    traj_x6 = coord_x_full[start_idx6:end_idx6]
    traj_y6 = coord_y_full[start_idx6:end_idx6]
    traj_z6 = coord_z_full[start_idx6:end_idx6]

    # Plot the extracted trajectory segments in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Test Window 1 segment
    ax.scatter(traj_x1, traj_y1, traj_z1, c='blue', marker='o', s=10, label='Test Window 1')
    # Plot Test Window 6 segment
    ax.scatter(traj_x6, traj_y6, traj_z6, c='lime', marker='^', s=80, label='Test Window 6')

    # Highlight the start and end of test_windows_6
    ax.scatter(coord_x_full[start_idx6], coord_y_full[start_idx6], coord_z_full[start_idx6],
               c='red', marker='D', s=100, label='TW6 Start')
    ax.scatter(coord_x_full[end_idx6 - 1], coord_y_full[end_idx6 - 1], coord_z_full[end_idx6 - 1],
               c='black', marker='D', s=100, label='TW6 End')

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.legend()
    plt.show()


# Example usage:
plot_3d_trajectory_segment(
    results, 
    test_windows_1, 
    test_windows_6, 
    subject="07/22/HW", 
    hand="left", 
    trial=1,
    file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv',
    seg_index=1
)

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
plot_single_trial_p_v_a_j_Nj_in_one(results, test_windows_1, subject="07/22/HW", hand="left", trial=1,
                              file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv',
                              fs=200, target_samples=101, plot_option=2, seg_index=1)

# Option 3: Overlay all segments
plot_single_trial_p_v_a_j_Nj_in_one(results, test_windows_1, subject="07/22/HW", hand="left", trial=1,
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
# utils5.process_and_save_combined_metrics_acorss_TWs(Block_Distance, reach_metrics,
#                                                     reach_sparc_test_windows_1_Normalizing, reach_TW_metrics_test_windows_1,
#                                                     reach_sparc_test_windows_3_Normalizing, reach_TW_metrics_test_windows_3,
#                                                     All_dates, DataProcess_folder)

utils5.process_and_save_combined_metrics_acorss_TWs(
    Block_Distance, reach_metrics,
    reach_sparc_test_windows_1_Normalizing,
    reach_sparc_test_windows_2_1_Normalizing,
    reach_sparc_test_windows_2_2_Normalizing,
    reach_sparc_test_windows_3_Normalizing,
    reach_sparc_test_windows_4_Normalizing,
    reach_sparc_test_windows_5_Normalizing,
    reach_sparc_test_windows_6_Normalizing,
    reach_TW_metrics_test_windows_1,
    reach_TW_metrics_test_windows_2_1,
    reach_TW_metrics_test_windows_2_2,
    reach_TW_metrics_test_windows_3,
    reach_TW_metrics_test_windows_4,
    reach_TW_metrics_test_windows_5,
    reach_TW_metrics_test_windows_6,
    All_dates, DataProcess_folder)

# # -------------------------------------------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------------------------------------------- 
# --- LOAD ALL COMBINED METRICS PER SUBJECT FROM PICKLE FILE ---
all_combined_metrics_acorss_TWs = utils5.load_selected_subject_results_acorss_TWs(All_dates, DataProcess_folder)

# Swap and rename metrics for consistency
all_combined_metrics_acorss_TWs = utils5.swap_and_rename_metrics(all_combined_metrics_acorss_TWs, All_dates)

# Filter all_combined_metrics based on distance and count NaNs
filtered_metrics_acorss_TWs, total_nan_acorss_TWs, Nan_counts_per_subject_per_hand_acorss_TWs, Nan_counts_per_index_acorss_TWs = utils5.filter_combined_metrics_and_count_nan(all_combined_metrics_acorss_TWs)

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
outliers = plot_histograms(filtered_metrics_acorss_TWs, sd_multiplier=4, overlay_median=True, overlay_sd=False, overlay_iqr=True)

# Update filtered metrics and count NaN replacements based on distance and duration thresholds: distance_threshold=15, duration_threshold=1.6
updated_metrics_acorss_TWs, Cutoff_counts_per_subject_per_hand_acorss_TWs, Cutoff_counts_per_index_acorss_TWs, total_nan_per_subject_hand_acorss_TWs = utils5.update_filtered_metrics_and_count(filtered_metrics_acorss_TWs)
# # -------------------------------------------------------------------------------------------------------------------
def plot_metric_boxplots(updated_metrics_acorss_TWs, metrics=["TW3_LDLJ", "TW3_sparc", "durations", "distance"], use_median=False):
    import matplotlib.pyplot as plt

    # Set up the subplot grid.
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
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
        # ax.set_title(f"Box Plot of {metric.lower()} {label} Values by Hand\n{test_title}")
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

# Call the function to plot boxplots for all four metrics in a 2x2 grid,
# using the median and IQR instead of mean and std.
# Example call:
plot_metric_boxplots(updated_metrics_acorss_TWs, metrics=["TW2_1_LDLJ", "TW2_2_LDLJ", "durations", "distance"], use_median=True)

## -------------------------------------------------------------------------------------------------------------------
# Do reach types that are faster on average also tend to be less accurate on average?
result_Check_SAT_in_trials_mean_median_of_reach_indices = utils6.Check_SAT_in_trials_mean_median_of_reach_indices(updated_metrics_acorss_TWs, '07/22/HW', 'durations', 'distance', stat_type="median")

# Within one reach location, is there still a speed–accuracy trade-off across repetitions?
utils6.scatter_plot_duration_distance_by_choice(updated_metrics_acorss_TWs, overlay_hands=False, selected_subjects=['07/22/HW'], special_indices=[0], show_hyperbolic_fit=False, color_mode="uniform", show_median_overlay=False)
_, corr_results, result_Check_SAT_in_reach_indices_by_hand_by_subject, heatmap_medians = utils6.Check_SAT_in_reach_indices_by_index_or_subject(updated_metrics_acorss_TWs, '07/22/HW', grouping="hand_by_subject", hyperbolic=False)

utils6.heatmap_spearman_correlation_reach_indices_signifcant(corr_results, hand="both", simplified=False, return_medians=True, overlay_median=True)
## -------------------------------------------------------------------------------------------------------------------
mean_stats, median_stats = utils6.calculate_trials_mean_median_of_reach_indices(updated_metrics_acorss_TWs, 'durations', 'distance')

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
                
                # Compute projection points on the 45° line and calculate their signed distances to (0,0)
                for i, (x, y) in enumerate(zip(z_durations, z_distances)):
                    # The projection of (x, y) onto the line x = y is:
                    proj_x = (x + y) / 2
                    proj_y = proj_x  # Since the line is x = y
                    
                    # The distance from (0,0) to the projection point along the line (with sign)
                    # is given by the dot product of the projection point with the unit vector along (1,1)
                    projection_distance = (x + y) / math.sqrt(2)
                    
                    subj = subjects[i]
                    if subj not in distances_dict:
                        distances_dict[subj] = {}
                    distances_dict[subj][reach_index + 1] = projection_distance
                    
                    # Scatter the projected point (orange x)
                    ax.scatter(proj_x, proj_y, color='orange', marker='x', s=30)
                    # Draw a dashed line connecting the original and projected points
                    ax.plot([x, proj_x], [y, proj_y], color='purple', linestyle=':', linewidth=0.8)
                
                ax.axhline(y=0, color='red')
                ax.axvline(x=0, color='red')
            ax.grid(True)
        else:
            raise ValueError("Invalid plot_type. Options are 'raw', 'zscore', 'distance'.")
    
    plt.tight_layout()
    plt.show()
    
    if plot_type == 'distance' and return_distances:
        return distances_dict

# zscore acorss subjects for median_stats.
MotorAcuity_Mean = {}
# Plot distance data and get computed MotorAcuity. plot_type = 'distance' / 'zscore' / 'raw'
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
updated_metrics_zscore = get_updated_metrics_zscore(updated_metrics_acorss_TWs, show_plots=True)

## -------------------------------------------------------------------------------------------------------------------
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
                                                                  updated_metrics_acorss_TWs)
# # -------------------------------------------------------------------------------------------------------------------
# Scatter plots for independent metrics (ldlj and sparc) versus durations and distance
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
    Plots histograms of median Spearman correlations for each subject,
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

for var in ['TW1_LDLJ', 'TW2_1_LDLJ', 'TW2_2_LDLJ', 'TW3_LDLJ', 'TW4_LDLJ', 'TW5_LDLJ', 'TW6_LDLJ',
            'TW1_sparc', 'TW2_1_sparc', 'TW2_2_sparc', 'TW3_sparc', 'TW4_sparc', 'TW5_sparc', 'TW6_sparc']:
    indep_var = var
    for dep_var in ['durations', 'distance', 'MotorAcuity']:
        print(f"Processing {indep_var} vs {dep_var}")
        
        # Example call: using independent variable and dependent variable for a specific subject/hand
        correlation_results = plot_indep_dep_scatter(
            updated_metrics_acorss_TWs, updated_metrics_zscore,
            subject='07/22/HW', hand='non_dominant',
            indep_var=indep_var, dep_var=dep_var, cols=4
        )
        
        # Example call: using independent variable and dependent variable for both hands
        heatmap_results = heatmap_spearman_correlation_all_subjects(
            updated_metrics_acorss_TWs, updated_metrics_zscore, hand="both",
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
## -------------------------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------------------------
## cross-check ldlj vs sparc for a single subject and hand within reach index
## -------------------------------------------------------------------------------------------------------------------

def plot_ldlj_vs_sparc(updated_metrics, ldlj, sparc, subject='07/22/HW', hand='non_dominant'):
    import matplotlib.pyplot as plt
    
    # Retrieve the Ldlj and Sparc data for the chosen subject and hand
    ldlj_data = updated_metrics[subject][hand][ldlj]
    sparc_data = updated_metrics[subject][hand][sparc]

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
        plt.xlabel(ldlj)
        plt.ylabel(sparc)
        plt.title(f"Index {reach_index + 1}\nSpearman: {corr:.2f}, p: {p_value:.3f}")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def heatmap_spearman_correlation_ldlj_sparc(updated_metrics, ldlj, sparc, hand="non_dominant",
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
                ldlj_data = updated_metrics[subject][hand_key].get(ldlj, {})
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
            ldlj_data = updated_metrics[subject][hand_key].get(ldlj, {})
            sparc_data = updated_metrics[subject][hand_key].get(sparc, {})
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


# plot Ldlj vs Sparc for a specific subject and hand
# plot_ldlj_vs_sparc(updated_metrics, subject='07/22/HW', hand='non_dominant')
plot_ldlj_vs_sparc(updated_metrics_acorss_TWs, 'TW3_LDLJ', 'TW3_sparc', subject='07/22/HW', hand='non_dominant')

# Example call to compute and plot heatmap of Spearman correlations (with p-values) between Ldlj and Sparc for both hands
heatmap_results = heatmap_spearman_correlation_ldlj_sparc(
    updated_metrics_acorss_TWs, 'TW3_LDLJ', 'TW3_sparc', hand="both",
    simplified=True, return_medians=True, overlay_median=True
)

# plot overlayed histograms for both hands.
plot_corr_histogram_overlay(heatmap_results, significant_only=False, significance_threshold=0.05)

# plot histogram with statistics
plot_histogram_spearman_corr_with_stats_reach_indices_by_subject(heatmap_results, show_value_on_legend=True)
## -------------------------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------------------------
## cross-check ldlj vs sparc for all subjects and both hands acoross all reach indices
## -------------------------------------------------------------------------------------------------------------------
def plot_ldlj_sparc_correlation(updated_metrics, ldlj, sparc):
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
            for trial in updated_metrics[subject][hand][ldlj]:
                ldlj_values.extend(updated_metrics[subject][hand][ldlj][trial])
                sparc_values.extend(updated_metrics[subject][hand][sparc][trial])

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
            ax.set_xlabel(ldlj)
            ax.set_ylabel(sparc)
            ax.set_title(f"{subject} - {hand}\nSpearman: {corr:.2f}, p: {p_value:.4f}")
            ax.grid(True)
            plot_idx += 1

    # Turn off any remaining subplots if they exist
    for ax in axs[plot_idx:]:
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()

    return correlation_results

def plot_ldlj_sparc_histogram(ldlj_sparc_correlations):
    import matplotlib.pyplot as plt

    corr_values = []
    p_values = []
    for subject, hands in ldlj_sparc_correlations.items():
        for hand, (corr, p_val) in hands.items():
            if not np.isnan(corr):
                corr_values.append(corr)
                p_values.append(p_val)

    plt.figure(figsize=(8, 6))
    plt.hist(corr_values, bins=10, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel('Spearman Correlation')
    plt.ylabel('Frequency')
    plt.title('Histogram of Ldlj vs Sparc Correlations')

    median_corr = np.median(corr_values)
    plt.axvline(median_corr, color='red', linestyle='--', linewidth=2, label=f"Median: {median_corr:.2f}")
    plt.legend()
    plt.grid(True)
    plt.show()

    p_values_array = np.array(p_values)
    significant_count = np.sum(p_values_array < 0.05)
    percentage_significant = (significant_count / len(p_values_array)) * 100
    print(f"Percentage of p-values < 0.05: {percentage_significant:.2f}%")

# Example call to plot Ldlj vs Sparc correlations for all subjects and hands
ldlj_sparc_correlations = plot_ldlj_sparc_correlation(updated_metrics_acorss_TWs, 'TW3_LDLJ', 'TW3_sparc')
# Example call to plot histogram of Ldlj vs Sparc correlations
plot_ldlj_sparc_histogram(ldlj_sparc_correlations)

# -------------------------------------------------------------------------------------------------------------------

def create_metrics_dataframe(updated_metrics_acorss_TWs, updated_metrics_zscore_by_trial, sBBTResult, MotorExperiences):
    """
    Creates a DataFrame from the combined metrics stored in the updated_metrics_acorss_TWs dictionary.
    It loops over each subject, hand, trial, and location (assumed to be 16) and collects the available
    metrics into a list of dictionaries. If a 'durations' value is NaN, a message is printed and that record
    is skipped.

    Parameters:
        updated_metrics_acorss_TWs (dict): A nested dictionary that contains metrics per subject, hand, and trial.
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
    for subject in updated_metrics_acorss_TWs:

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

        for hand in updated_metrics_acorss_TWs[subject]:
            # Attempt to get the sBBTResult value for the subject and hand.
            try:
                # Get the value from the "non_dominant" column for the row where Subject matches subject_name
                sbbt_val = sBBTResult.loc[sBBTResult["Subject"] == subject_name, hand].values
            except KeyError:
                sbbt_val = np.nan

            for trial in updated_metrics_acorss_TWs[subject][hand]['durations']:
                for loc in range(16):
                    duration_val = updated_metrics_acorss_TWs[subject][hand]['durations'][trial][loc]
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

                            # reach metrics
                            'cartesian_distances': updated_metrics_acorss_TWs[subject][hand]['cartesian_distances'][trial][loc],
                            'path_distances': updated_metrics_acorss_TWs[subject][hand]['path_distances'][trial][loc],
                            'v_peaks': updated_metrics_acorss_TWs[subject][hand]['v_peaks'][trial][loc],

                            # Dependent variables: task performance metrics
                            'durations': duration_val,
                            'distance': updated_metrics_acorss_TWs[subject][hand]['distance'][trial][loc],
                            'MotorAcuity': updated_metrics_zscore_by_trial[subject][hand]['MotorAcuity'][trial][loc],
                            
                            # Independent variables: time window metrics
                            # Time Window 1
                            'TW1_acc_peaks': updated_metrics_acorss_TWs[subject][hand]['TW1_acc_peaks'][trial][loc],
                            'TW1_jerk_peaks': updated_metrics_acorss_TWs[subject][hand]['TW1_jerk_peaks'][trial][loc],
                            'TW1_LDLJ': updated_metrics_acorss_TWs[subject][hand]['TW1_LDLJ'][trial][loc],
                            'TW1_sparc': updated_metrics_acorss_TWs[subject][hand]['TW1_sparc'][trial][loc],
                            # Time Window 2-1
                            'TW2_1_acc_peaks': updated_metrics_acorss_TWs[subject][hand]['TW2_1_acc_peaks'][trial][loc],
                            'TW2_1_jerk_peaks': updated_metrics_acorss_TWs[subject][hand]['TW2_1_jerk_peaks'][trial][loc],
                            'TW2_1_LDLJ': updated_metrics_acorss_TWs[subject][hand]['TW2_1_LDLJ'][trial][loc],
                            'TW2_1_sparc': updated_metrics_acorss_TWs[subject][hand]['TW2_1_sparc'][trial][loc],
                            # Time Window 2-2
                            'TW2_2_acc_peaks': updated_metrics_acorss_TWs[subject][hand]['TW2_2_acc_peaks'][trial][loc],
                            'TW2_2_jerk_peaks': updated_metrics_acorss_TWs[subject][hand]['TW2_2_jerk_peaks'][trial][loc],
                            'TW2_2_LDLJ': updated_metrics_acorss_TWs[subject][hand]['TW2_2_LDLJ'][trial][loc],
                            'TW2_2_sparc': updated_metrics_acorss_TWs[subject][hand]['TW2_2_sparc'][trial][loc],
                            # Time Window 3
                            'TW3_acc_peaks': updated_metrics_acorss_TWs[subject][hand]['TW3_acc_peaks'][trial][loc],
                            'TW3_jerk_peaks': updated_metrics_acorss_TWs[subject][hand]['TW3_jerk_peaks'][trial][loc],
                            'TW3_LDLJ': updated_metrics_acorss_TWs[subject][hand]['TW3_LDLJ'][trial][loc],
                            'TW3_sparc': updated_metrics_acorss_TWs[subject][hand]['TW3_sparc'][trial][loc],
                            # Time Window 4
                            'TW4_acc_peaks': updated_metrics_acorss_TWs[subject][hand]['TW4_acc_peaks'][trial][loc],
                            'TW4_jerk_peaks': updated_metrics_acorss_TWs[subject][hand]['TW4_jerk_peaks'][trial][loc],
                            'TW4_LDLJ': updated_metrics_acorss_TWs[subject][hand]['TW4_LDLJ'][trial][loc],
                            'TW4_sparc': updated_metrics_acorss_TWs[subject][hand]['TW4_sparc'][trial][loc],
                            # Time Window 5
                            'TW5_acc_peaks': updated_metrics_acorss_TWs[subject][hand]['TW5_acc_peaks'][trial][loc],
                            'TW5_jerk_peaks': updated_metrics_acorss_TWs[subject][hand]['TW5_jerk_peaks'][trial][loc],
                            'TW5_LDLJ': updated_metrics_acorss_TWs[subject][hand]['TW5_LDLJ'][trial][loc],
                            'TW5_sparc': updated_metrics_acorss_TWs[subject][hand]['TW5_sparc'][trial][loc],
                            # Time Window 6
                            'TW6_acc_peaks': updated_metrics_acorss_TWs[subject][hand]['TW6_acc_peaks'][trial][loc],
                            'TW6_jerk_peaks': updated_metrics_acorss_TWs[subject][hand]['TW6_jerk_peaks'][trial][loc],
                            'TW6_LDLJ': updated_metrics_acorss_TWs[subject][hand]['TW6_LDLJ'][trial][loc],
                            'TW6_sparc': updated_metrics_acorss_TWs[subject][hand]['TW6_sparc'][trial][loc],
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
df = create_metrics_dataframe(updated_metrics_acorss_TWs, updated_metrics_zscore_by_trial, sBBTResult, MotorExperiences)
# -------------------------------------------------------------------------------------------------------------------

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import scipy.stats as stats
from scipy.stats import wilcoxon

from patsy import dmatrices

import seaborn as sns
import pickle

import pandas as pd
import seaborn as sns
import numpy as np
import math

import pingouin as pg
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

def model_info(model):
    llf = model.llf   # log-likelihood
    k = model.df_modelwc  # number of estimated parameters
    n = model.nobs   # number of observations

    aic = -2*llf + 2*k
    bic = -2*llf + k*np.log(n)
    return aic, bic

def compare_models(models, model_names):
    """
    Compare multiple statsmodels models (OLS or MixedLM) side by side using
    log-likelihood, AIC, and BIC.

    Parameters:
        models: list of fitted statsmodels models
        model_names: list of names for the models
    Returns:
        DataFrame summary of fit statistics
    """
    rows = []
    for name, model in zip(model_names, models):
        llf = model.llf
        k = model.df_modelwc
        n = model.nobs
        aic = -2*llf + 2*k
        bic = -2*llf + k*np.log(n)

        rows.append({
            "Model": name,
            "LogLik": llf,
            "AIC": aic,
            "BIC": bic
        })
    return pd.DataFrame(rows)

def plot_predictions(model, df, response_col="durations"):
    """
    Plot actual vs. predicted values for a fitted statsmodels model.
    
    Parameters:
        model: fitted statsmodels model (mixedlm or ols)
        df: DataFrame used for fitting
        response_col: the column name of the response variable
    """
    # Predicted values
    df["predicted"] = model.predict(df)

    plt.figure(figsize=(6, 6))
    plt.scatter(df[response_col], df["predicted"], alpha=0.3)
    plt.plot([df[response_col].min(), df[response_col].max()],
             [df[response_col].min(), df[response_col].max()],
             'r--', label="Perfect prediction")

    plt.xlabel("Actual " + response_col)
    plt.ylabel("Predicted " + response_col)
    plt.title(f"Predicted vs Actual: {response_col}")
    plt.legend()
    plt.show()

def plot_predictions_facet(model, df, response_col="durations", facet_by="Subject"):
    """
    Faceted scatter plots of actual vs. predicted values, grouped by Subject or Hand,
    and a second panel showing the distribution of residuals (predicted – actual) for each facet.

    Parameters:
        model: fitted statsmodels model (OLS or MixedLM)
        df: DataFrame used for fitting
        response_col: dependent variable name in df
        facet_by: column name to facet by ("Subject" or "Hand")
    """
    df = df.copy()
    df["predicted"] = model.predict(df)

    # Scatter plot of actual vs. predicted in facets.
    g = sns.lmplot(
        data=df,
        x=response_col,
        y="predicted",
        col=facet_by,
        hue=facet_by,
        col_wrap=4,
        scatter_kws={"alpha": 0.4, "s": 20},
        line_kws={"color": "red"}
    )

    # Add 1:1 diagonal reference line to each facet
    for ax in g.axes.flatten():
        lims = [
            min(df[response_col].min(), df["predicted"].min()),
            max(df[response_col].max(), df["predicted"].max())
        ]
        ax.plot(lims, lims, 'k--', alpha=0.7)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    g.set_axis_labels(f"Actual {response_col}", f"Predicted {response_col}")
    g.set_titles("{col_name}")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Model Predictions vs. Actual {response_col}, faceted by {facet_by}")

    # Calculate residuals and create a facet plot for their distributions.
    df["residuals"] = df["predicted"] - df[response_col]
    g2 = sns.FacetGrid(df, col=facet_by, hue=facet_by, col_wrap=4, height=4)
    g2.map(plt.hist, "residuals", bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    g2.set_axis_labels("Residuals (Predicted - Actual)", "Frequency")
    g2.set_titles("{col_name}")
    g2.fig.suptitle(f"Residual Distributions by {facet_by}", y=1.03)
    plt.tight_layout()
    plt.show()

def print_model_summaries(models):
    for i, model in enumerate(models):
        print(f"\nModel {i} results:")
        print(model.summary())
        aic_val, bic_val = model_info(model)
        print("AIC:", aic_val)
        print("BIC:", bic_val)

def check_multicollinearity(df, model_choice="model3_1", custom_formula=None):
    """
    Checks multicollinearity by computing the VIF for predictors of a specified model.
    
    Parameters:
        df (DataFrame): The data to analyze.
        model_choice (str): Choose a predefined model ("model3_1" or "model14_1").
                            If any other string is used, a custom formula must be provided via custom_formula.
        custom_formula (str, optional): A custom regression formula as a string if model_choice is not predefined.
        
    Prints:
        VIF values for the predictors based on the selected model or custom formula.
    """
    if model_choice == "model3_1":
        formula = ("distance_log ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + "
                   "TW6_LDLJ + TW6_sparc + MotorAcuity + durations_log")
        print("VIF for model3_1 predictors:")
    elif model_choice == "model14_1":
        formula = ("distance_log ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + "
                   "TW6_LDLJ + TW6_sparc + MotorAcuity + durations_log + sBBTResult")
        print("VIF for model14_1 predictors:")
    else:
        if custom_formula:
            formula = custom_formula
            print("VIF for custom model predictors:")
        else:
            print("Invalid model choice. Please choose either 'model3_1', 'model14_1', or provide a custom_formula.")
            return

    y, X = dmatrices(formula, df, return_type='dataframe')
    vif_df = pd.DataFrame({
        "Variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    print(vif_df)
    print("-" * 80)


with open("/Users/yilinwu/Desktop/honours data/DataProcess/df.pkl", "rb") as f:
    df = pickle.load(f)

print("DataFrame loaded with shape:", df.shape)










































# ---------------------------------------------------------------------------------------------------------------------



# --- Baseline control predictors (all models include these) ---
# Hand, distance-to-subject, distance-to-partition, random intercepts for Subject (29 groups)

# Model 0: Early baseline with TW6_LDLJ
model0 = smf.mixedlm(
    "distance ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ",
    df, groups=df["Subject"]
).fit(reml=True)

# Model 1: Early baseline with TW6_LDLJ & TW6_sparc
model1 = smf.mixedlm(
    "distance ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc",
    df, groups=df["Subject"]
).fit(reml=True)

# Model 2: Add MotorAcuity
model2 = smf.mixedlm(
    "distance ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc + MotorAcuity",
    df, groups=df["Subject"]
).fit(reml=True)

# Model 3: Add durations (core baseline)
model3 = smf.mixedlm(
    "distance ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc + MotorAcuity + durations",
    df, groups=df["Subject"]
).fit(reml=True)

# Model 3.1: Log-transform distance and durations
model3_1= smf.mixedlm(
    "distance_log ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc + MotorAcuity + durations_log",
    df, groups=df["Subject"]
).fit(reml=True)

# --- Kinematic refinements ---

# Model 4: Add v_peaks
model4 = smf.mixedlm(
    "distance ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc + MotorAcuity + durations + v_peaks",
    df, groups=df["Subject"]
).fit(reml=True)

# Model 5: Add TW6_acc_peaks
model5 = smf.mixedlm(
    "distance ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc + MotorAcuity + durations + TW6_acc_peaks",
    df, groups=df["Subject"]
).fit(reml=True)

# Model 6: Add TW6_jerk_peaks  (best-fitting model by AIC)
model6 = smf.mixedlm(
    "distance ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc + MotorAcuity + durations + TW6_jerk_peaks",
    df, groups=df["Subject"]
).fit(reml=True)

# Model 7: Add cartesian_distances
model7 = smf.mixedlm(
    "distance ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc + MotorAcuity + durations + cartesian_distances",
    df, groups=df["Subject"]
).fit(reml=True)

# Model 8: Add path_distances
model8 = smf.mixedlm(
    "distance ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc + MotorAcuity + durations + path_distances",
    df, groups=df["Subject"]
).fit(reml=True)

# --- Demographics ---

# Model 9: Add Gender and Age
model9 = smf.mixedlm(
    "distance ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc + MotorAcuity + durations + C(Gender) + Age",
    df, groups=df["Subject"]
).fit(reml=True)

# --- Habits ---

# Model 10: Add physical habit score
model10 = smf.mixedlm(
    "distance ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc + MotorAcuity + durations + physical_h_total_weighted",
    df, groups=df["Subject"]
).fit(reml=True)

# Model 11: Add musical habit score
model11 = smf.mixedlm(
    "distance ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc + MotorAcuity + durations + musical_h_total_weighted",
    df, groups=df["Subject"]
).fit(reml=True)

# Model 12: Add digital habit score
model12 = smf.mixedlm(
    "distance ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc + MotorAcuity + durations + digital_h_total_weighted",
    df, groups=df["Subject"]
).fit(reml=True)

# Model 13: Add overall habit score
model13 = smf.mixedlm(
    "distance ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc + MotorAcuity + durations + overall_h_total_weighted",
    df, groups=df["Subject"]
).fit(reml=True)

# --- sBBTResult ---
# Model 14: Add sBBTResult
model14 = smf.mixedlm(
    "distance ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc + MotorAcuity + durations + sBBTResult",
    df, groups=df["Subject"]
).fit(reml=True)

# Model 14_1: Log-transform distance and durations + sBBTResult
model14_1= smf.mixedlm(
    "distance_log ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc + MotorAcuity + durations_log + sBBTResult",
    df, groups=df["Subject"]
).fit(reml=True)


model_duration = smf.mixedlm(
    "durations_log ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc",
    df, groups=df["Subject"]
).fit(reml=True)
model_distance = smf.mixedlm(
    "distance_log ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc",
    df, groups=df["Subject"]
).fit(reml=True)

model_motoracuity = smf.mixedlm(
    "MotorAcuity ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc",
    df, groups=df["Subject"]
).fit(reml=True)

print_model_summaries([model_duration, model_distance, model_motoracuity])

# Calculate R squared (pseudo R² using r2_score on the predictions)

r2_duration = r2_score(df["durations_log"], model_duration.predict(df))
r2_distance = r2_score(df["distance_log"], model_distance.predict(df))
r2_motoracuity = r2_score(df["MotorAcuity"], model_motoracuity.predict(df))

print("\nR squared:")
print("Duration model: {:.4f}".format(r2_duration))
print("Distance model: {:.4f}".format(r2_distance))
print("MotorAcuity model: {:.4f}".format(r2_motoracuity))






# -------------------------------------------------------------------------------------------------------------------
models = [model0, model1, model2, model3, model3_1, model4, model5, model6, model7, model8,
          model9, model10, model11, model12, model13, model14, model14_1]
print_model_summaries(models)
# -------------------------------------------------------------------------------------------------------------------
compare_models(
    models,
    ["Model 0", "Model 1", "Model 2", "Model 3", "Model 3.1", "Model 4", "Model 5", "Model 6",
     "Model 7", "Model 8", "Model 9", "Model 10", "Model 11", "Model 12", "Model 13", "Model 14", "Model 14.1"]
)
# -------------------------------------------------------------------------------------------------------------------
plot_predictions(model3_1, df, response_col="distance_log")
plot_predictions_facet(model3_1, df, response_col="distance_log", facet_by="Subject")
plot_predictions_facet(model3_1, df, response_col="distance_log", facet_by="Hand")
# -------------------------------------------------------------------------------------------------------------------
plot_predictions(model14_1, df, response_col="distance_log")
plot_predictions_facet(model14_1, df, response_col="distance_log", facet_by="Subject")
plot_predictions_facet(model14_1, df, response_col="distance_log", facet_by="Hand")
# -------------------------------------------------------------------------------------------------------------------
# 👉 Drop or choose between variables with VIF > 5–10 (they overlap too much).
# 👉 Their correlation is very weak. They measure related but largely different things.

# Check multicollinearity for model3_1
check_multicollinearity(df, model_choice="model3_1")
# Check multicollinearity for model14_1
check_multicollinearity(df, model_choice="model14_1")

# -------------------------------------------------------------------------------------------------------------------
# check correlation between TW6_LDLJ and TW6_sparc using Spearman correlation
corr, p_value = stats.spearmanr(df["TW6_LDLJ"].dropna(), df["TW6_sparc"].dropna())
print(f"Spearman correlation between TW6_LDLJ and TW6_sparc: {corr}, p-value: {p_value}")
# -------------------------------------------------------------------------------------------------------------------
from statsmodels.sandbox.stats.multicomp import multipletests

#
# Compute subject‐level median stats for durations, distances, and MotorAcuity,
# plus get the sBBT score (using the first non‐NaN value per subject per hand).
hand_stats = (
    df.groupby(["Subject", "Hand"])
      .agg({
          "durations": "median",
          "distance": "median",
          "MotorAcuity": "median",
          "sBBTResult": lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan
      })
      .reset_index()
)

print("Subject-level median stats by hand:")
print(hand_stats)

# For each hand separately, compute Spearman correlations for sBBT vs durations, distance and MotorAcuity.
metrics = ["durations", "distance", "MotorAcuity"]
for hand in hand_stats["Hand"].unique():
    hand_df = hand_stats[hand_stats["Hand"] == hand]
    pvals = []
    corr_results = {}
    for metric in metrics:
        rho, p = spearmanr(hand_df["sBBTResult"], hand_df[metric], nan_policy="omit")
        pvals.append(p)
        corr_results[metric] = rho
    # Apply FDR correction for multiple comparisons
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    
    print(f"\nHand: {hand}")
    for i, metric in enumerate(metrics):
        print(f"Spearman correlation (sBBT vs median {metric}): rho = {corr_results[metric]:.3f}, raw p = {pvals[i]:.3f}, adjusted p = {pvals_corrected[i]:.3f}")

# Plot separate scatter plots for each hand.
import matplotlib.pyplot as plt

hands = hand_stats["Hand"].unique()
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for i, metric in enumerate(metrics):
    for hand in hands:
        hand_df = hand_stats[hand_stats["Hand"] == hand]
        axs[i].scatter(hand_df["sBBTResult"], hand_df[metric], alpha=0.8, label=f"{hand}")
    if metric == "durations":
        axs[i].set_ylabel("Median Duration")
        axs[i].set_title(f"sBBT vs Median Duration")
    elif metric == "distance":
        axs[i].set_ylabel("Median Distance")
        axs[i].set_title(f"sBBT vs Median Distance")
    elif metric == "MotorAcuity":
        axs[i].set_ylabel("Median MotorAcuity")
        axs[i].set_title(f"sBBT vs Median MotorAcuity")
    axs[i].set_xlabel("sBBT Score")
    axs[i].legend()

plt.tight_layout()
plt.show()


# Compute subject-level best performance stats (per hand):
# For durations and distance best performance are the minimum values,
# and for MotorAcuity the best performance is the maximum value.
hand_best_stats = (
    df.groupby(["Subject", "Hand"])
      .agg({
          "durations": "min",         # best = shortest duration
          "distance": "min",          # best = shortest distance
          "MotorAcuity": "max",       # best = highest MotorAcuity
          "sBBTResult": lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan
      })
      .reset_index()
)

print("\nSubject-level best performance stats by hand:")
print(hand_best_stats)

# For each hand, compute Spearman correlations (best durations, distance, MotorAcuity vs sBBT) with multiple comparisons correction.
for hand in hand_best_stats["Hand"].unique():
    hand_df = hand_best_stats[hand_best_stats["Hand"] == hand]
    pvals_best = []
    corr_best = {}
    for metric in metrics:
        rho, p = spearmanr(hand_df["sBBTResult"], hand_df[metric], nan_policy="omit")
        pvals_best.append(p)
        corr_best[metric] = rho
    reject, pvals_corrected_best, _, _ = multipletests(pvals_best, alpha=0.05, method="fdr_bh")
    
    print(f"\nHand: {hand}")
    for i, metric in enumerate(metrics):
        print(f"Spearman correlation (sBBT vs best {metric}): rho = {corr_best[metric]:.3f}, raw p = {pvals_best[i]:.3f}, adjusted p = {pvals_corrected_best[i]:.3f}")

# Plot scatter plots for best performance metrics.
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for i, metric in enumerate(metrics):
    for hand in hands:
        hand_df = hand_best_stats[hand_best_stats["Hand"] == hand]
        axs[i].scatter(hand_df["sBBTResult"], hand_df[metric], alpha=0.8, label=f"{hand}")
    if metric == "durations":
        axs[i].set_ylabel("Best (Shortest) Duration")
        axs[i].set_title(f"sBBT vs Shortest Duration")
    elif metric == "distance":
        axs[i].set_ylabel("Best (Shortest) Distance")
        axs[i].set_title(f"sBBT vs Shortest Distance")
    elif metric == "MotorAcuity":
        axs[i].set_ylabel("Best (Highest) MotorAcuity")
        axs[i].set_title(f"sBBT vs Highest MotorAcuity")
    axs[i].set_xlabel("sBBT Score")
    axs[i].legend()

plt.tight_layout()
plt.show()







import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pingouin import partial_corr
import numpy as np
from statsmodels.stats.multitest import multipletests
import os
import math
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# ============================
# 1. Fit Mixed-Effects Models
# ============================

model_duration = smf.mixedlm(
    "durations ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc",
    df, groups=df["Subject"]
).fit(reml=True)

model_distance = smf.mixedlm(
    "distance ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc",
    df, groups=df["Subject"]
).fit(reml=True)

model_motoracuity = smf.mixedlm(
    "MotorAcuity ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc",
    df, groups=df["Subject"]
).fit(reml=True)

# ============================
# 2. Print Model Summaries
# ============================

print("\n=== Model: Duration ===")
print(model_duration.summary())

print("\n=== Model: Distance ===")
print(model_distance.summary())

print("\n=== Model: Motor Acuity ===")
print(model_motoracuity.summary())

# ============================
# 3. Compute Marginal and Conditional R²
# (variance explained by fixed vs. fixed+random)
# ============================
def r2_mixedlm(model, df, response):
    """
    Nakagawa & Schielzeth R² for mixed models:
    - marginal R² (variance explained by fixed effects)
    - conditional R² (variance explained by fixed + random effects)
    """
    fe_var = np.var(np.dot(model.model.exog, model.fe_params))
    re_var = sum(model.cov_re.iloc[i, i] for i in range(model.cov_re.shape[0]))
    resid_var = model.scale
    total_var = fe_var + re_var + resid_var
    
    r2_marginal = fe_var / total_var
    r2_conditional = (fe_var + re_var) / total_var
    return {"response": response, "R2_marginal": r2_marginal, "R2_conditional": r2_conditional}

r2_results = pd.DataFrame([
    r2_mixedlm(model_duration, df, "durations"),
    r2_mixedlm(model_distance, df, "distance"),
    r2_mixedlm(model_motoracuity, df, "MotorAcuity")
])

print("\n=== Mixed Model R² Results ===")
print(r2_results)

# ============================
# 4. Semi-Partial R² (per predictor)
# Drop one predictor at a time → compare marginal R²
# ============================

predictors = ["C(Hand)", "C(Dis_to_subject)", "C(Dis_to_partition)", "TW6_LDLJ", "TW6_sparc"]

def semi_partial_r2(full_formula, df, response, predictors):
    results = []
    full_model = smf.mixedlm(full_formula, df, groups=df["Subject"]).fit(reml=True)
    full_r2 = r2_mixedlm(full_model, df, response)["R2_marginal"]

    for pred in predictors:
        reduced_formula = full_formula.replace(" + " + pred, "")
        reduced_model = smf.mixedlm(reduced_formula, df, groups=df["Subject"]).fit(reml=True)
        reduced_r2 = r2_mixedlm(reduced_model, df, response)["R2_marginal"]
        semi_r2 = full_r2 - reduced_r2
        results.append({"response": response, "predictor": pred, "semi_partial_R2": semi_r2})
    
    return pd.DataFrame(results)

sp_r2_duration = semi_partial_r2(
    "durations ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc",
    df, "durations", predictors)

sp_r2_distance = semi_partial_r2(
    "distance ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc",
    df, "distance", predictors)

sp_r2_motoracuity = semi_partial_r2(
    "MotorAcuity ~ C(Hand) + C(Dis_to_subject) + C(Dis_to_partition) + TW6_LDLJ + TW6_sparc",
    df, "MotorAcuity", predictors)

sp_r2_all = pd.concat([sp_r2_duration, sp_r2_distance, sp_r2_motoracuity])

print("\n=== Semi-Partial R² (Variance Explained by Each Predictor) ===")
print(sp_r2_all)

# Now you can export `sp_r2_all` as a table for your thesis.


# ============================

# 08/07/DA - non_dominant - Trial /Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/08/07/DA/DA_tBBT18.csv - Location 16


# -----------------------------------------------------------------------
# Find highest TW6_LDLJ
max_ldlj = df['TW6_LDLJ'].max()
row_with_max = df[df['TW6_LDLJ'] == max_ldlj]
print("Maximum TW6_LDLJ:", max_ldlj)
print("Found at:")
for index, row in row_with_max.iterrows():
    complete_trial_name = f"{row['Subject']} - {row['Hand']} - Trial {row['Trial']} - Location {row['Location']}"
    print(complete_trial_name)
# 08/07/DA - non_dominant - Trial /Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/08/07/DA/DA_tBBT18.csv - Location 16

# Find lowest TW6_LDLJ
min_ldlj = df['TW6_LDLJ'].min()
row_with_min = df[df['TW6_LDLJ'] == min_ldlj]
print("\nMinimum TW6_LDLJ:", min_ldlj)
print("Found at:")
for index, row in row_with_min.iterrows():
    complete_trial_name = f"{row['Subject']} - {row['Hand']} - Trial {row['Trial']} - Location {row['Location']}"
    print(complete_trial_name)
#07/30/JT - non_dominant - Trial /Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/30/JT/JT_tBBT60.csv - Location 8


### Plot hand trajectory with velocity-coded coloring and highlighted segments
def plot_trajectory(results, subject='07/22/HW', hand='right', trial=1,
                    file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT53.csv',
                    overlay_trial=0, velocity_segment_only=False, plot_mode='all'):
    """
    Plots individual coordinate plots and two 3D trajectory plots for the specified trial.
    Colors each trajectory point based on the instantaneous velocity for a selected segment 
    if velocity_segment_only is True; otherwise all points are colored according to velocity.
    Points outside a defined segment are colored lightgrey.
    
    Additionally, the 'plot_mode' option allows plotting:
      - 'all': the whole trial.
      - 'segment': only from the first highlight index to the last highlight index.
    
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
    
    marker = 'RFIN' if hand == 'right' else 'LFIN'

    # Compute instantaneous velocity from the trajectory space (assume constant sampling rate 200Hz)
    vel = results[subject][hand][trial][file_path]['traj_space'][marker][1]
    
    # Normalize velocities between 0 and 1
    v_min = np.min(vel)
    v_max = np.max(vel)
    if v_max - v_min > 0:
        v_norm = (vel - v_min) / (v_max - v_min)
    else:
        v_norm = np.ones_like(vel)
    
    # Map velocity to colors via Viridis with exponential scaling for contrast
    point_colors = [plt.cm.viridis(1 - (v_norm[i]**2)) for i in range(n_points)]
    
    # If velocity_segment_only is True, only the points within each paired segment retain their velocity color.
    if velocity_segment_only and highlight_indices:
        segments = []
        for idx in range(0, len(highlight_indices) - 1, 2):
            segments.append((highlight_indices[idx], highlight_indices[idx+1]))
        for i in range(n_points):
            in_segment = any(min(seg) <= i <= max(seg) for seg in segments)
            if not in_segment:
                point_colors[i] = mcolors.to_rgba('lightgrey')
    
    # Determine the indices to plot based on plot_mode option
    if plot_mode == 'segment' and highlight_indices:
        start_idx = min(highlight_indices[0], highlight_indices[-1])
        end_idx = max(highlight_indices[0], highlight_indices[-1])
    else:
        start_idx = 0
        end_idx = n_points - 1

    # Slice the data to plot
    plot_indices = np.arange(start_idx, end_idx + 1)
    coord_x_plot = coord_x[plot_indices]
    coord_y_plot = coord_y[plot_indices]
    coord_z_plot = coord_z[plot_indices]
    vel_plot    = np.array(vel)[plot_indices]
    colors_plot = [point_colors[i] for i in plot_indices]
    time_points = plot_indices / 200

    # Create the plot layout: 4 rows on the left (velocity, X, Y, Z) and one 3D plot on the right
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(nrows=4, ncols=2, width_ratios=[1, 1.2])
    
    ax_vel = fig.add_subplot(gs[0, 0])
    ax_vel.scatter(time_points, vel_plot, c=colors_plot, marker='o', s=5)
    ax_vel.set_ylabel('Velocity')
    ax_vel.set_title('Instantaneous Velocity')
    
    ax_x = fig.add_subplot(gs[1, 0])
    ax_x.scatter(time_points, coord_x_plot, c=colors_plot, marker='o', s=5)
    ax_x.set_ylabel('X')
    
    ax_y = fig.add_subplot(gs[2, 0])
    ax_y.scatter(time_points, coord_y_plot, c=colors_plot, marker='o', s=5)
    ax_y.set_ylabel('Y')
    
    ax_z = fig.add_subplot(gs[3, 0])
    ax_z.scatter(time_points, coord_z_plot, c=colors_plot, marker='o', s=5)
    ax_z.set_xlabel('Time (s)')
    ax_z.set_ylabel('Z')
    
    # Overlay markers at the designated highlight indices (if they fall within our plot range)
    for order, idx in enumerate(highlight_indices, start=1):
        if start_idx <= idx <= end_idx:
            t_val = idx / 200
            color = 'green' if order % 2 == 1 else 'blue'
            marker = 'o' if order % 2 == 1 else 'X'
            ax_vel.scatter(t_val, vel[idx], color=color, marker=marker, s=50)
            ax_x.scatter(t_val, coord_x[idx], color=color, marker=marker, s=50)
            ax_y.scatter(t_val, coord_y[idx], color=color, marker=marker, s=50)
            ax_z.scatter(t_val, coord_z[idx], color=color, marker=marker, s=50)
    
    # Right side: 3D Plot
    ax3d = fig.add_subplot(gs[:, 1], projection='3d')
    ax3d.scatter(coord_x_plot, coord_y_plot, coord_z_plot, c=colors_plot, marker='o', s=5)
    ax3d.set_xlabel(coord_prefix + "X (mm)", fontsize=14, labelpad=0)
    ax3d.set_ylabel(coord_prefix + "Y (mm)", fontsize=14, labelpad=0)
    ax3d.set_zlabel(coord_prefix + "Z (mm)", fontsize=14, labelpad=0)
    ax3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax3d.set_xlim([min(coord_x_plot), max(coord_x_plot)])
    ax3d.set_ylim([min(coord_y_plot), max(coord_y_plot)])
    ax3d.set_zlim([min(coord_z_plot), max(coord_z_plot)])


    
    # Overlay markers on the 3D plot for indices within plot range
    for order, idx in enumerate(highlight_indices, start=1):
        if start_idx <= idx <= end_idx:
            color = 'green' if order % 2 == 1 else 'blue'
            marker = 'o' if order % 2 == 1 else 'X'
            ax3d.scatter(coord_x[idx], coord_y[idx], coord_z[idx], color=color, marker=marker, s=50)

    # ax3d.text(coord_x[highlight_indices[0]], coord_y[highlight_indices[0]], coord_z[highlight_indices[0]],
    #         "start", color='green', fontsize=20)
    # ax3d.text(coord_x[highlight_indices[1]], coord_y[highlight_indices[1]], coord_z[highlight_indices[1]],
    #         "end", color='blue', fontsize=20)    
    plt.tight_layout()
    plt.show()
    
    # Additional 3D Trajectory Plot for the first segment from first to last highlight, if possible
    if len(highlight_indices) >= 2:
        seg_start = highlight_indices[14]
        seg_end = highlight_indices[15]
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
        
        ax3d_seg.set_xlabel(coord_prefix + "X (mm)", fontsize=14, labelpad=0)
        ax3d_seg.set_ylabel(coord_prefix + "Y (mm)", fontsize=14, labelpad=0)
        ax3d_seg.set_zlabel(coord_prefix + "Z (mm)", fontsize=14, labelpad=0)
        # ax3d_seg.set_title("Selected 3D Trajectory Segment")
        ax3d_seg.legend()
        plt.tight_layout()
        plt.show()

plot_trajectory(results, subject='08/07/DA', hand='left', trial=1,
                file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/08/07/DA/DA_tBBT18.csv',
                overlay_trial=0, velocity_segment_only=True, plot_mode='segment')



def plot_3d_trajectory_segment(results, test_windows_1, test_windows_6, subject, hand, trial, file_path, seg_index=0):
    """
    Extracts a 3D trajectory segment based on paired start and end indices from test_windows_1 
    and overlays another segment extracted from test_windows_6 in the same 3D plot.
    
    Parameters:
        results (dict): Dictionary containing trajectory data.
        test_windows_1 (dict): Dictionary with paired start and end indices (first test window).
        test_windows_6 (dict): Dictionary with paired start and end indices (second test window).
        subject (str): Subject identifier.
        hand (str): 'right' or 'left'.
        trial (int): Trial index.
        file_path (str): Path to the CSV file.
        seg_index (int): Index of the segment to extract (default is 0).
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

    # Get the full trajectory data
    traj_data = results[subject][hand][trial][file_path]['traj_data']
    coord_prefix = "RFIN_" if hand == "right" else "LFIN_"
    coord_x_full = np.array(traj_data[coord_prefix + "X"])
    coord_y_full = np.array(traj_data[coord_prefix + "Y"])
    coord_z_full = np.array(traj_data[coord_prefix + "Z"])

    # Get the paired start and end indices for the first test window
    start_idx1, end_idx1 = test_windows_1[subject][hand][file_path][seg_index]
    # Extract the segment from test_windows_1
    traj_x1 = coord_x_full[start_idx1:end_idx1]
    traj_y1 = coord_y_full[start_idx1:end_idx1]
    traj_z1 = coord_z_full[start_idx1:end_idx1]

    # Get the paired start and end indices for the second test window (test_windows_6)
    start_idx6, end_idx6 = test_windows_6[subject][hand][file_path][seg_index]
    # Extract the segment from test_windows_6
    traj_x6 = coord_x_full[start_idx6:end_idx6]
    traj_y6 = coord_y_full[start_idx6:end_idx6]
    traj_z6 = coord_z_full[start_idx6:end_idx6]

    # Plot the extracted trajectory segments in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Test Window 1 segment
    ax.scatter(traj_x1, traj_y1, traj_z1, c='blue', marker='o', s=10, label='Test Window 1')
    # Plot Test Window 6 segment
    ax.scatter(traj_x6, traj_y6, traj_z6, c='lime', marker='^', s=80, label='Test Window 6')

    # Highlight the start and end of test_windows_6
    ax.scatter(coord_x_full[start_idx6], coord_y_full[start_idx6], coord_z_full[start_idx6],
               c='red', marker='D', s=100, label='TW6 Start')
    ax.scatter(coord_x_full[end_idx6 - 1], coord_y_full[end_idx6 - 1], coord_z_full[end_idx6 - 1],
               c='black', marker='D', s=100, label='TW6 End')

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.legend()
    plt.show()


# Example usage:
plot_3d_trajectory_segment(
    results, 
    test_windows_1, 
    test_windows_6, 
    subject='07/30/JT', 
    hand="left", 
    trial=1,
    file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/30/JT/JT_tBBT60.csv',
    seg_index=7
)