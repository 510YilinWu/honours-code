import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

fs=200 # Frame rate in Hz


# --- PROCESS RESULTS TO GET REACH SPEED SEGMENTS ---
def get_reach_speed_segments(results):
    reach_speed_segments = {}

    for date, hands_data in results.items():
        reach_speed_segments[date] = {}
        for hand, (indices, _) in hands_data.items():
            reach_speed_segments[date][hand] = {}
            for trial, trial_indices in indices.items():
                start_indices = [trial_indices[i] for i in range(len(trial_indices)) if i % 2 == 0]
                end_indices = [trial_indices[i] for i in range(len(trial_indices)) if i % 2 == 1]
                reach_speed_segments[date][hand][trial] = list(zip(start_indices, end_indices))

    return reach_speed_segments

# --- CALCULATE REACH METRICS ---
def calculate_reach_metrics(reach_speed_segments, results, fs):
    reach_durations = {}
    reach_cartesian_distances = {}
    reach_path_distances = {}
    reach_v_peaks = {}
    reach_v_peak_indices = {}

    for date, hands_data in reach_speed_segments.items():
        reach_durations[date] = {}
        reach_cartesian_distances[date] = {}
        reach_path_distances[date] = {}
        reach_v_peaks[date] = {}
        reach_v_peak_indices[date] = {}

        for hand, trials_data in hands_data.items():
            reach_durations[date][hand] = {}
            reach_cartesian_distances[date][hand] = {}
            reach_path_distances[date][hand] = {}
            reach_v_peaks[date][hand] = {}
            reach_v_peak_indices[date][hand] = {}

            for trial, segments in trials_data.items():
                # --- Calculate durations for each segment ---
                durations = [(end - start) * (1 / fs) for start, end in segments]
                reach_durations[date][hand][trial] = durations

                # --- Calculate distances for each segment ---
                position = results[date][hand][1][trial]['traj_space']['RFIN'][0] if hand == "right" else results[date][hand][1][trial]['traj_space']['LFIN'][0]
                cartesian_distances = [abs(position[end] - position[start]) for start, end in segments] 
                reach_cartesian_distances[date][hand][trial] = cartesian_distances

                # --- Calculate path distances for each segment ---
                path_distances = [
                    position[start:end].diff().abs().sum()
                    for start, end in segments
                ]
                reach_path_distances[date][hand][trial] = path_distances

                # --- Calculate peak velocities for each segment ---
                speed = results[date][hand][1][trial]['traj_space']['RFIN'][1] if hand == "right" else results[date][hand][1][trial]['traj_space']['LFIN'][1]
                v_peak = [speed[start:end].max() for start, end in segments]
                reach_v_peaks[date][hand][trial] = v_peak
                v_peak_indices = [
                    speed[start:end].idxmax() for start, end in segments
                ]
                reach_v_peak_indices[date][hand][trial] = v_peak_indices

    reach_metrics = {
        "reach_durations": reach_durations,
        "reach_cartesian_distances": reach_cartesian_distances,
        "reach_path_distances": reach_path_distances,
        "reach_v_peaks": reach_v_peaks,
        "reach_v_peak_indices": reach_v_peak_indices,
    }

    return reach_metrics


# --- DEFINE TIME WINDOWS ---
def define_time_windows(reach_speed_segments, reach_metrics, fs, window_size):
    test_windows_1 = reach_speed_segments
    test_windows_2_1 = {
        date: {
            hand: {
                trial: [
                    (segment[0], reach_metrics['reach_v_peak_indices'][date][hand][trial][i]) 
                    for i, segment in enumerate(reach_speed_segments[date][hand][trial])
                ]
                for trial in reach_speed_segments[date][hand]
            }
            for hand in reach_speed_segments[date]
        }
        for date in reach_speed_segments
    }
    test_windows_2_2 = {
        date: {
            hand: {
                trial: [
                    (reach_metrics['reach_v_peak_indices'][date][hand][trial][i], segment[1]) 
                    for i, segment in enumerate(reach_speed_segments[date][hand][trial])
                ]
                for trial in reach_speed_segments[date][hand]
            }
            for hand in reach_speed_segments[date]
        }
        for date in reach_speed_segments
    }
    window_frames = int(fs * window_size)
    test_windows_3 = {
        date: {
            hand: {
                trial: [
                    (reach_metrics['reach_v_peak_indices'][date][hand][trial][i] - window_frames,
                     reach_metrics['reach_v_peak_indices'][date][hand][trial][i] + window_frames)
                    for i in range(len(reach_metrics['reach_v_peak_indices'][date][hand][trial]))
                ]
                for trial in reach_speed_segments[date][hand]
            }
            for hand in reach_speed_segments[date]
        }
        for date in reach_speed_segments
    }
    window_frames_100ms = int(fs * 0.1)
    test_windows_4 = {
        date: {
            hand: {
                trial: [
                    (reach_metrics['reach_v_peak_indices'][date][hand][trial][i] - window_frames_100ms,
                     reach_metrics['reach_v_peak_indices'][date][hand][trial][i])
                    for i in range(len(reach_metrics['reach_v_peak_indices'][date][hand][trial]))
                ]
                for trial in reach_speed_segments[date][hand]
            }
            for hand in reach_speed_segments[date]
        }
        for date in reach_speed_segments
    }
    test_windows_5 = {
        date: {
            hand: {
                trial: [
                    (reach_metrics['reach_v_peak_indices'][date][hand][trial][i],
                     reach_metrics['reach_v_peak_indices'][date][hand][trial][i] + window_frames_100ms)
                    for i in range(len(reach_metrics['reach_v_peak_indices'][date][hand][trial]))
                ]
                for trial in reach_speed_segments[date][hand]
            }
            for hand in reach_speed_segments[date]
        }
        for date in reach_speed_segments
    }
    window_frames_200ms = int(fs * 0.2)
    test_windows_6 = {
        date: {
            hand: {
                trial: [
                    (
                        int((segment[0] + segment[1]) * 0.5 - window_frames_200ms),
                        int((segment[0] + segment[1]) * 0.5 + window_frames_200ms)
                    )
                    for segment in reach_speed_segments[date][hand][trial]
                ]
                for trial in reach_speed_segments[date][hand]
            }
            for hand in reach_speed_segments[date]
        }
        for date in reach_speed_segments
    }
    return test_windows_1, test_windows_2_1, test_windows_2_2, test_windows_3, test_windows_4, test_windows_5, test_windows_6

# # --- CALCULATE REACH METRICS SPECIFIC TO TIME WINDOW ---
# def calculate_reach_metrics_for_time_windows(test_windows, results): 
#     reach_acc_peaks = {}
#     reach_jerk_peaks = {}
#     reach_LDLJ = {}

#     for date, hands_data in test_windows.items():
#         reach_acc_peaks[date] = {}
#         reach_jerk_peaks[date] = {}
#         reach_LDLJ[date] = {}

#         for hand, trials_data in hands_data.items():
#             reach_acc_peaks[date][hand] = {}
#             reach_jerk_peaks[date][hand] = {}
#             reach_LDLJ[date][hand] = {}

#             for trial, segments in trials_data.items():
#                 marker = "RFIN" if hand == "right" else "LFIN"
#                 position, speed, acceleration, jerk = results[date][hand][1][trial]['traj_space'][marker]

#                 # Find the peak acceleration
#                 acc_peak = [acceleration[start:end].max() for start, end in segments]
#                 reach_acc_peaks[date][hand][trial] = acc_peak

#                 # Find the peak jerk
#                 jerk_peak = [jerk[start:end].max() for start, end in segments]
#                 reach_jerk_peaks[date][hand][trial] = jerk_peak

#                 LDLJ = []
#                 for start, end in segments:
#                     jerk_segment = jerk[start:end] # Get the jerk segment for the current time window
#                     duration = (end - start) / fs # Calculate the duration of the segment in seconds
#                     t = np.linspace(0, duration, len(jerk_segment)) # Create a time vector for the segment
#                     jerk_squared_integral = np.trapezoid(jerk_segment**2, t) # Calculate the integral of the squared jerk
#                     vpeak = speed[start:end].max() # Get the peak speed for the segment
#                     dimensionless_jerk = (duration**3 / vpeak**2) * jerk_squared_integral
#                     LDLJ.append(-math.log(abs(dimensionless_jerk), math.e))

#                 reach_LDLJ[date][hand][trial] = LDLJ

#     reach_TW_metrics = {
#         "reach_acc_peaks": reach_acc_peaks,
#         "reach_jerk_peaks": reach_jerk_peaks,
#         "reach_LDLJ": reach_LDLJ
#     }
#     return reach_TW_metrics

# --- CALCULATE REACH METRICS SPECIFIC TO TIME WINDOW ---
def calculate_reach_metrics_for_time_windows_Normalizing(test_windows, results): 
    reach_acc_peaks = {}
    reach_jerk_peaks = {}
    reach_LDLJ = {}
    target_samples = 101  # Standardized number of time points (0-100%)

    for date, hands_data in test_windows.items():
        reach_acc_peaks[date] = {}
        reach_jerk_peaks[date] = {}
        reach_LDLJ[date] = {}

        for hand, trials_data in hands_data.items():
            reach_acc_peaks[date][hand] = {}
            reach_jerk_peaks[date][hand] = {}
            reach_LDLJ[date][hand] = {}

            for trial, segments in trials_data.items():
                marker = "RFIN" if hand == "right" else "LFIN"
                position, speed, acceleration, jerk = results[date][hand][1][trial]['traj_space'][marker]

                # Find the peak acceleration
                acc_peak = [acceleration[start:end].max() for start, end in segments]
                reach_acc_peaks[date][hand][trial] = acc_peak

                # Find the peak jerk
                jerk_peak = [jerk[start:end].max() for start, end in segments]
                reach_jerk_peaks[date][hand][trial] = jerk_peak

                LDLJ = []
                for start, end in segments:
                    # Extract the jerk segment and convert to a numpy array
                    jerk_segment = np.array(jerk[start:end])
                    duration = (end - start) / fs  # Duration in seconds

                    # Create original time vector for the segment
                    t_orig = np.linspace(0, duration, num=len(jerk_segment))
                    # Create standardized time vector (e.g., 0 to 100% with target_samples points)
                    t_std = np.linspace(0, duration, num=target_samples)
                    # Warp the jerk segment to the standardized time base using linear interpolation
                    warped_jerk = np.interp(t_std, t_orig, jerk_segment)

                    # Calculate the integral of the squared, warped jerk segment
                    jerk_squared_integral = np.trapezoid(warped_jerk**2, t_std)

                    # Get the peak speed for the current segment
                    vpeak = speed[start:end].max()
                    dimensionless_jerk = (duration**3 / vpeak**2) * jerk_squared_integral
                    LDLJ.append(-math.log(abs(dimensionless_jerk), math.e))

                reach_LDLJ[date][hand][trial] = LDLJ

    reach_TW_metrics = {
        "reach_acc_peaks": reach_acc_peaks,
        "reach_jerk_peaks": reach_jerk_peaks,
        "reach_LDLJ": reach_LDLJ
    }
    return reach_TW_metrics

# --- SAVE LDLJ VALUES ---
def save_ldlj_values(reach_TW_metrics, DataProcess_folder):
    """
    Save all LDLJ values by subject, hand, and trial to a CSV file.

    Parameters:
        reach_TW_metrics (dict): Metrics for time windows.
        DataProcess_folder (str): Folder path to save the CSV file.
    """
    ldlj_table = []

    for date in reach_TW_metrics['reach_LDLJ']:
        for hand in ['right', 'left']:
            for trial in reach_TW_metrics['reach_LDLJ'][date][hand]:
                ldlj_values = reach_TW_metrics['reach_LDLJ'][date][hand][trial]
                ldlj_table.append({
                    "Date": date,
                    "Hand": hand,
                    "Trial": trial,
                    **{f"Reach {i + 1}": ldlj_value for i, ldlj_value in enumerate(ldlj_values)}
                })

    ldlj_df = pd.DataFrame(ldlj_table)

    # Save as CSV file
    csv_save_path = os.path.join(DataProcess_folder, "ldlj_values.csv")
    ldlj_df.to_csv(csv_save_path, index=False)
    print(f"LDLJ values saved to {csv_save_path}")

# # --- SPARC FUNCTION ---
# def sparc(movement, fs, padlevel=4, fc=20.0, amp_th=0.05):
#     """
#     Calcualtes the smoothness of the given speed profile using the modified
#     spectral arc length metric.

#     Parameters
#     ----------
#     movement : np.array
#                The array containing the movement speed profile.
#     fs       : float
#                The sampling frequency of the data.
#     padlevel : integer, optional
#                Indicates the amount of zero padding to be done to the movement
#                data for estimating the spectral arc length. [default = 4]
#     fc       : float, optional
#                The max. cut off frequency for calculating the spectral arc
#                length metric. [default = 20.]
#     amp_th   : float, optional
#                The amplitude threshold to used for determing the cut off
#                frequency upto which the spectral arc length is to be estimated.
#                [default = 0.05]

#     Returns
#     -------
#     sal      : float
#                The spectral arc length estimate of the given movement's
#                smoothness.
#     (f, Mf)  : tuple of two np.arrays
#                This is the frequency(f) and the magntiude spectrum(Mf) of the
#                given movement data. This spectral is from 0. to fs/2.
#     (f_sel, Mf_sel) : tuple of two np.arrays
#                       This is the portion of the spectrum that is selected for
#                       calculating the spectral arc length.

#     Notes
#     -----
#     This is the modfieid spectral arc length metric, which has been tested only
#     for discrete movements.

#     Examples
#     --------
#     >>> t = np.arange(-1, 1, 0.01)
#     >>> move = np.exp(-5*pow(t, 2))
#     >>> sal, _, _ = sparc(move, fs=100.)
#     >>> '%.5f' % sal
#     '-1.41403'

#     """
#     # Number of zeros to be padded.
#     nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

#     # Frequency
#     f = np.arange(0, fs, fs / nfft)
#     # Normalized magnitude spectrum
#     Mf = abs(np.fft.fft(movement, nfft))
#     Mf = Mf / max(Mf)

#     # Indices to choose only the spectrum within the given cut off frequency
#     # Fc.
#     # NOTE: This is a low pass filtering operation to get rid of high frequency
#     # noise from affecting the next step (amplitude threshold based cut off for
#     # arc length calculation).
#     fc_inx = ((f <= fc) * 1).nonzero()
#     f_sel = f[fc_inx]
#     Mf_sel = Mf[fc_inx]

#     # Choose the amplitude threshold based cut off frequency.
#     # Index of the last point on the magnitude spectrum that is greater than
#     # or equal to the amplitude threshold.
#     inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
#     fc_inx = range(inx[0], inx[-1] + 1)
#     f_sel = f_sel[fc_inx]
#     Mf_sel = Mf_sel[fc_inx]

#     # Calculate arc length
#     new_sal = -sum(np.sqrt(pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) +
#                            pow(np.diff(Mf_sel), 2)))
#     return new_sal, (f, Mf), (f_sel, Mf_sel)

# # --- CALCULATE REACH SPARC ---
# def calculate_reach_sparc(test_windows, results):
#     """
#     Calculate SPARC for each test window for all dates, hands, and trials.

#     Args:
#         test_windows (dict): Dictionary containing test windows for each date, hand, and trial.
#         results (dict): Dictionary containing processed results for each date, hand, and trial.

#     Returns:
#         dict: SPARC values for each date, hand, and trial.
#     """
#     reach_sparc = {}

#     for date, hands_data in test_windows.items():
#         reach_sparc[date] = {}

#         for hand, trials_data in hands_data.items():
#             reach_sparc[date][hand] = {}

#             for trial, segments in trials_data.items():
#                 marker_name = "RFIN" if hand == "right" else "LFIN"
#                 speed_data = results[date][hand][1][trial]["traj_space"][marker_name][1]

#                 sparc_values = []
#                 for start_frame, end_frame in segments:
#                     window_speed = speed_data[start_frame:end_frame]
#                     sal, _, _ = sparc(window_speed, fs=200.)
#                     sparc_values.append(sal)

#                 reach_sparc[date][hand][trial] = sparc_values

#     return reach_sparc


# --- SPARC FUNCTION ---
def sparc_Normalizing(movement, fs, padlevel=4, fc=20.0, amp_th=0.05):
    """
    Calculates the smoothness of the given movement profile using the modified
    spectral arc length metric. Before computing the metric, the trajectory is
    normalized in amplitude (space) so that the start-to-target displacement has unit length.

    Parameters
    ----------
    movement : np.array
               The array containing the movement trajectory. It can be multidimensional.
    fs       : float
               The sampling frequency of the data.
    padlevel : integer, optional
               Amount of zero padding to be done to the movement data for estimating
               the spectral arc length. [default = 4]
    fc       : float, optional
               The cutoff frequency for calculating the spectral arc length metric. [default = 20.]
    amp_th   : float, optional
               The amplitude threshold used for determining the cutoff frequency up to 
               which the spectral arc length is estimated. [default = 0.05]

    Returns
    -------
    sal      : float
               The spectral arc length estimate of the movement's smoothness.
    (f, Mf)  : tuple of two np.arrays
               The frequency (f) and the normalized magnitude spectrum (Mf) computed from the fft.
    (f_sel, Mf_sel) : tuple of two np.arrays
                      The portion of the spectrum that is selected for spectral arc length calculation.

    Notes
    -----
    This modified spectral arc length metric has been tested only for discrete movements.
    
    Examples
    --------
    >>> t = np.linspace(0, 1, 200)
    >>> move = np.exp(-5 * (t - 0.5)**2)
    >>> sal, _, _ = sparc(move, fs=200.)
    >>> '%.5f' % sal
    '-1.41403'
    """
    # Normalize movement: scale trajectory to have unit start-to-end displacement.
    # Ensure movement is a numpy array.
    movement = np.array(movement)
    amp = np.linalg.norm(movement[-1] - movement[0])
    if amp != 0:
        movement = movement / amp

    # Number of zeros to be padded.
    nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

    # Frequency vector.
    f = np.arange(0, fs, fs / nfft)
    # Normalized magnitude spectrum.
    Mf = abs(np.fft.fft(movement, nfft))
    Mf = Mf / max(Mf)

    # Select spectrum within the cutoff frequency fc.
    fc_inx = ((f <= fc) * 1).nonzero()
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]

    # Apply amplitude threshold for further selection.
    inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
    fc_inx = range(inx[0], inx[-1] + 1)
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]

    # Calculate spectral arc length.
    new_sal = -sum(np.sqrt(pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) +
                           pow(np.diff(Mf_sel), 2)))
    return new_sal, (f, Mf), (f_sel, Mf_sel)

# # --- SPARC WITH PLOTS ---
# def sparc_with_plots(movement, fs=200, padlevel=4, fc=20.0, amp_th=0.05):
#     # 1. Time domain plot (original movement)
#     t = np.arange(len(movement)) / fs
#     plt.figure(figsize=(12, 8))
#     plt.subplot(3, 2, 1)
#     plt.plot(t, movement)
#     plt.title("1. Speed Profile (Time Domain)")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Speed")

#     # 2. FFT and Normalization
#     nfft = int(2 ** (np.ceil(np.log2(len(movement))) + padlevel))
#     f = np.arange(0, fs, fs / nfft)
#     Mf = np.abs(np.fft.fft(movement, nfft))
#     Mf = Mf / np.max(Mf)

#     plt.subplot(3, 2, 2)
#     plt.plot(f[:nfft // 2], Mf[:nfft // 2])
#     plt.title("2. Full Frequency Spectrum")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Normalized Magnitude")

#     # 3. Cutoff Frequency Filtering
#     fc_idx = np.where(f <= fc)[0]
#     f_sel = f[fc_idx]
#     Mf_sel = Mf[fc_idx]

#     plt.subplot(3, 2, 3)
#     plt.plot(f_sel, Mf_sel)
#     plt.title("3. Spectrum Below Cutoff (fc = 20 Hz)")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Magnitude")

#     # 4. Amplitude Thresholding with Continuous Range
#     inx = np.where(Mf_sel >= amp_th)[0]
#     fc_inx = np.arange(inx[0], inx[-1] + 1)
#     f_cut = f_sel[fc_inx]
#     Mf_cut = Mf_sel[fc_inx]

#     plt.subplot(3, 2, 4)
#     plt.plot(f_sel, Mf_sel, color='lightgray', label='All under 20Hz')
#     plt.plot(f_cut, Mf_cut, color='blue', label='Above threshold')
#     plt.axhline(y=amp_th, color='red', linestyle='--', label='Threshold')
#     plt.title("4. After Amplitude Thresholding")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Magnitude")
#     plt.legend()

#     # 5. Spectral Arc Length Calculation (matching sparc())
#     df = np.diff(f_cut) / (f_cut[-1] - f_cut[0])
#     dM = np.diff(Mf_cut)
#     arc_length = -np.sum(np.sqrt(df ** 2 + dM ** 2))

#     plt.subplot(3, 2, 5)
#     plt.plot(f_cut, Mf_cut, marker='o')
#     for i in range(len(df)):
#         plt.plot([f_cut[i], f_cut[i+1]], [Mf_cut[i], Mf_cut[i+1]], 'k--')
#     plt.title("5. Arc Segments Used in SPARC")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Magnitude")

#     # 6. Display SPARC Value
#     plt.subplot(3, 2, 6)
#     plt.text(0.1, 0.5, f"SPARC Value:\n{arc_length:.4f}", fontsize=20)
#     plt.axis('off')

#     plt.tight_layout()
#     plt.show()

#     return arc_length

# --- CALCULATE REACH SPARC ---
def calculate_reach_sparc_Normalizing(test_windows, results):
    """
    Calculate SPARC for each test window for all dates, hands, and trials.

    Args:
        test_windows (dict): Dictionary containing test windows for each date, hand, and trial.
        results (dict): Dictionary containing processed results for each date, hand, and trial.

    Returns:
        dict: SPARC values for each date, hand, and trial.
    """
    reach_sparc = {}

    for date, hands_data in test_windows.items():
        reach_sparc[date] = {}

        for hand, trials_data in hands_data.items():
            reach_sparc[date][hand] = {}

            for trial, segments in trials_data.items():
                marker_name = "RFIN" if hand == "right" else "LFIN"
                speed_data = results[date][hand][1][trial]["traj_space"][marker_name][1]

                sparc_values = []
                for start_frame, end_frame in segments:
                    window_speed = speed_data[start_frame:end_frame]
                    sal, _, _ = sparc_Normalizing(window_speed, fs=200.)
                    sparc_values.append(sal)

                reach_sparc[date][hand][trial] = sparc_values

    return reach_sparc

# --- SAVE SPARC VALUES ---
def save_sparc_values(reach_sparc, DataProcess_folder):
    """
    Save all SPARC values by subject, hand, and trial into a CSV file.

    Parameters:
        reach_sparc (dict): Dictionary containing SPARC values for each date, hand, and trial.
        DataProcess_folder (str): Path to the folder where the CSV file will be saved.
    """
    sparc_table = []

    for date in reach_sparc:
        for hand in reach_sparc[date]:
            for trial in reach_sparc[date][hand]:
                sparc_values = reach_sparc[date][hand][trial]
                sparc_table.append({
                    "Date": date,
                    "Hand": hand,
                    "Trial": trial,
                    **{f"Window {i + 1}": sparc_value for i, sparc_value in enumerate(sparc_values)}
                })

    sparc_df = pd.DataFrame(sparc_table)

    # Save as CSV file
    csv_save_path = os.path.join(DataProcess_folder, "sparc_values.csv")
    sparc_df.to_csv(csv_save_path, index=False)
    print(f"SPARC values saved to {csv_save_path}")
