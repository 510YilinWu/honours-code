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
utils9.run_analysis_and_plot(results, reach_speed_segments, test_windows_7,
                      subject="07/22/HW", hand="left",
                      file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv',
                      fs=200, target_samples=101, seg_index=3, dis_phases=0.3,
                      video_save_path='trajectory_with_icon_3.mp4')
# -------------------------------------------------------------------------------------------------------------------
all_phase_data = utils9.calculate_phase_indices_all_files(
    results, reach_speed_segments, test_windows_7,
    fs=200, target_samples=101, dis_phases=0.3
)

utils9.plot_3d_trajectory_icon_all_phase_data(results, reach_speed_segments, test_windows_7,
                                       subject="07/22/HW", hand="left",
                                       file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv',
                                       seg_index=3, fs=200, target_samples=101, dis_phases=0.3,
                                       show_icon=False)

utils9.plot_trajectory_all_phase_data(results, reach_speed_segments, test_windows_7, all_phase_data,
                         subject="07/22/HW", hand="left",
                         file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv',
                         seg_index=3)

utils9.plot_3d_trajectory_video_with_icon_all_phase_data(
    results, reach_speed_segments, test_windows_7, all_phase_data,
    subject="07/22/HW", hand="left",
    file_path="/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv",
    seg_index=3,
    fs=200, target_samples=101, dis_phases=0.3,
    video_save_path="trajectory_with_icon_3_new.mp4"
)