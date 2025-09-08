import utils1 # Importing utils1 for data Pre-processing
import utils2 # Importing utils2 for reach metrics calculation and time window Specific calculation
import utils3 # Importing utils3 for plotting functions
import utils4 # Importing utils4 for image files
import utils5 # Importing utils5 for combining metrics
import utils6 # Importing utils6 for Data Analysis and Visualization
import utils7 # Importing utils7 for Motor Experiences
import utils8 # Importing utils8 for sBBTResult

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

# # --- PROCESS ALL SUBJECTS' IMAGES RETURN tBBT ERROR FROM IMAGE, SAVE AS pickle file---
# All_Subject_tBBTs_errors = utils4.process_all_subjects_images(All_dates, tBBT_Image_folder, DataProcess_folder)

# # # -------------------------------------------------------------------------------------------------------------------

# # PART 1: CHECK IF DATA PROCESSING IS DONE AND LOAD RESULTS
# # --- CHECK CALIBRATION FOLDERS FOR PICKLE FILES ---
# utils4.check_calibration_folders_for_pickle(All_dates, tBBT_Image_folder)

# # # --- LOAD ALL SUBJECTS' tBBT ERROR FROM IMAGE, SAVE AS pickle file---
# Block_Distance = utils4.load_selected_subject_errors(All_dates, DataProcess_folder)

# --- LOAD RESULTS FROM PICKLE FILE "processed_results.pkl" ---
results = utils1.load_selected_subject_results(All_dates, DataProcess_folder)

# # # # -------------------------------------------------------------------------------------------------------------------

# # PART 2: Reach Metrics Calculation
# # --- GET REACH SPEED SEGMENTS ---
# reach_speed_segments = utils2.get_reach_speed_segments(results)

# # --- CALCULATE REACH METRICS ---
# reach_metrics = utils2.calculate_reach_metrics(reach_speed_segments, results, fs=200)


# # --- DEFINE TIME WINDOWS BASED ON SELECTED METHOD ---
# # test_windows_1: Original full reach segments (start to end of movement)
# # test_windows_2: From movement start to velocity peak (focuses on movement buildup)
# # test_windows_3: Symmetric window around velocity peak (captures activity before and after peak) (500 ms total)
# # test_windows_4: 100 ms before velocity peak (captures lead-up dynamics)
# # test_windows_5: 100 ms after velocity peak (captures immediate post-peak activity)
# # test_windows_6: Custom time window centered around the midpoint of each segment 
# test_windows_1, test_windows_2, test_windows_3, test_windows_4, test_windows_5, test_windows_6 = utils2.define_time_windows(reach_speed_segments, reach_metrics, fs=200, window_size=0.25)

# # --- CALCULATE REACH METRICS SPECIFIC TO TIME WINDOW ---
# reach_TW_metrics = utils2.calculate_reach_metrics_for_time_windows(test_windows_3, results)

# # --- CALCULATE SPARC FOR EACH TEST WINDOW FOR ALL DATES, HANDS, AND TRIALS ---
# reach_sparc_test_windows_1 = utils2.calculate_reach_sparc(test_windows_1, results)
# reach_sparc_test_windows_2 = utils2.calculate_reach_sparc(test_windows_2, results)
# reach_sparc_test_windows_3 = utils2.calculate_reach_sparc(test_windows_3, results)

# # --- Save ALL LDLJ VALUES BY SUBJECT, HAND, AND TRIAL ---
# utils2.save_ldlj_values(reach_TW_metrics, DataProcess_folder)

# # --- Save ALL SPARC VALUES BY SUBJECT, HAND, AND TRIAL ---
# utils2.save_sparc_values(reach_sparc_test_windows_1, DataProcess_folder)

# # # # # -------------------------------------------------------------------------------------------------------------------

# # PART 3: Combine Metrics and Save Results
# # --- PROCESS AND SAVE COMBINED METRICS [DURATIONS, SPARC, LDLJ, AND DISTANCE, CALCULATED SPEED AND ACCURACY FOR ALL DATES]---
# utils5.process_and_save_combined_metrics(Block_Distance, reach_metrics, reach_sparc_test_windows_1, reach_TW_metrics, All_dates, DataProcess_folder)

# # -------------------------------------------------------------------------------------------------------------------
# --- LOAD ALL COMBINED METRICS PER SUBJECT FROM PICKLE FILE ---
all_combined_metrics = utils5.load_selected_subject_results(All_dates, DataProcess_folder)

# Swap and rename metrics for consistency
all_combined_metrics = utils5.swap_and_rename_metrics(all_combined_metrics, All_dates)
# # -------------------------------------------------------------------------------------------------------------------

# Filter all_combined_metrics based on distance and count NaNs
filtered_metrics, total_nan, Nan_counts_per_subject_per_hand, Nan_counts_per_index = utils5.filter_combined_metrics_and_count_nan(all_combined_metrics)

# Update filtered metrics and count NaN replacements based on distance and duration thresholds: distance_threshold=15, duration_threshold=1.6
updated_metrics, Cutoff_counts_per_subject_per_hand, Cutoff_counts_per_index, total_nan_per_subject_hand = utils5.update_filtered_metrics_and_count(filtered_metrics)

# -------------------------------------------------------------------------------------------------------------------
# PART 4: Data Analysis and Visualization

# -------------------------------------------------------------------------------------------------------------------
# Do reach types that are faster on average also tend to be less accurate on average?
result_Check_SAT_in_trials_mean_median_of_reach_indices = utils6.Check_SAT_in_trials_mean_median_of_reach_indices(updated_metrics, '07/22/HW', 'durations', 'distance', stat_type="median")

# Within one reach location, is there still a speed–accuracy trade-off across repetitions?
utils6.scatter_plot_duration_distance_by_choice(updated_metrics, overlay_hands=False, selected_subjects=['07/22/HW'], special_indices=[0], show_hyperbolic_fit=False, color_mode="uniform", show_median_overlay=False)
_, corr_results, result_Check_SAT_in_reach_indices_by_hand_by_subject, heatmap_medians = utils6.Check_SAT_in_reach_indices_by_index_or_subject(updated_metrics, '07/22/HW', grouping="hand_by_subject", hyperbolic=False)

utils6.heatmap_spearman_correlation_reach_indices_signifcant(corr_results, hand="both", simplified=False, return_medians=True, overlay_median=True)
# -------------------------------------------------------------------------------------------------------------------
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import matplotlib.colors as mcolors
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.stats import spearmanr, zscore

from matplotlib.colors import LinearSegmentedColormap
import pickle


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
# -------------------------------------------------------------------------------------------------------------------
# subject = '07/22/HW'
# hand = 'right'
# target_file = '/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT63.csv'

# # Get all file keys in the trial dictionary (results[subject][hand][1])
# file_keys = list(results[subject][hand][1].keys())

# # Find the index of the target file in the file keys list
# target_index = file_keys.index(target_file)
# print("Index of target file:", target_index)

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
# Calculate RMS reprojection error statistics
utils4.compute_rms_reprojection_error_stats()
# -------------------------------------------------------------------------------------------------------------------
# Load Motor Experiences from CSV into a dictionary
MotorExperiences = utils7.load_motor_experiences("/Users/yilinwu/Desktop/Yilin-Honours/MotorExperience.csv")
# Calculate demographic variables from MotorExperiences
utils7.display_motor_experiences_stats(MotorExperiences)

# Update MotorExperiences with weighted scores
utils7.update_overall_h_total_weighted(MotorExperiences)
# # -------------------------------------------------------------------------------------------------------------------

# Load sBBTResult from CSV into a DataFrame and compute right and left hand scores
sBBTResult = utils8.load_and_compute_sbbt_result()
print(sBBTResult[['right_score', 'left_score']])

# Examine correlations between motor experience metrics and sBBT scores by extracting the highest score for each hand
motor_keys = [
    "physical_h_total_weighted",
    "musical_h_total_weighted",
    "digital_h_total_weighted",
    "other_h_total_weighted",
    "overall_h_total_weighted"
]
score_columns = ["right_score", "left_score"]

utils7.analyze_motor_experience_correlations(motor_keys, score_columns, sBBTResult, MotorExperiences)

