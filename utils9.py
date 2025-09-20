
import numpy as np
import matplotlib.pyplot as plt
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
                        0.05 * reach_metrics['reach_v_peaks'][date][hand][trial][i]
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




# example single placement

# def calculate_phase_indices(results, reach_speed_segments_1, reach_speed_segments_2, 
#                             subject="06/19/CZ", hand="right",
#                             file_path='/Users/yilinwu/Desktop/Honours-Honours/Subject/Traj/2025/06/19/CZ/CZ_tBBT01.csv',
#                             fs=200, target_samples=101, seg_index=0, dis_phases=0.3):
#     """
#     Calculates phase indices and related segmentation data.
#     Returns a dictionary with kinematic data, phase indices,
#     segmentation indices, LDLJ values, and filtered peak indices.
#     """
#     # Select marker based on hand.
#     marker = 'RFIN' if hand == 'right' else 'LFIN'
    
#     # Extract trajectory arrays.
#     traj_data = results[subject][hand][1][file_path]['traj_space'][marker]
#     position_full = traj_data[0]
#     velocity_full = traj_data[1]
#     acceleration_full = traj_data[2]
#     jerk_full = traj_data[3]
    
#     # Extract coordinate arrays.
#     traj_data_full = results[subject][hand][1][file_path]['traj_data']
#     coord_prefix = "RFIN_" if hand == "right" else "LFIN_"
#     coord_x_full = np.array(traj_data_full[coord_prefix + "X"])
#     coord_y_full = np.array(traj_data_full[coord_prefix + "Y"])
#     coord_z_full = np.array(traj_data_full[coord_prefix + "Z"])
    
#     # Use reach_speed_segments_2 to get main segment's start and end.
#     seg_range = reach_speed_segments_2[subject][hand][file_path][seg_index]
#     start_seg, end_seg = seg_range

#     # Compute Latency phase based on x-coordinate cumulative differences.
#     coord_x_seg = coord_x_full[start_seg:end_seg]
#     cum_dist = np.insert(np.cumsum(np.abs(np.diff(coord_x_seg))), 0, 0)
#     if np.any(cum_dist > dis_phases):
#         latency_end_idx = start_seg + np.argmax(cum_dist > dis_phases)
#     else:
#         latency_end_idx = start_seg

#     # Compute Verification phase from the end.
#     rev_cum = np.insert(np.cumsum(np.abs(np.diff(coord_x_seg[::-1]))), 0, 0)
#     if np.any(rev_cum > dis_phases):
#         verification_start_idx = end_seg - np.argmax(rev_cum > dis_phases)
#     else:
#         verification_start_idx = end_seg - 1

#     # Compute Ballistic phase using cumulative differences.
#     if len(coord_x_seg) > 1:
#         ballistic_diffs = np.abs(np.diff(coord_x_seg))
#         ballistic_total = np.sum(ballistic_diffs)
#         ballistic_threshold = 0.25 * ballistic_total
#         ballistic_cum = np.insert(np.cumsum(ballistic_diffs), 0, 0)
#         if np.any(ballistic_cum > ballistic_threshold):
#             ballistic_end = start_seg + np.argmax(ballistic_cum > ballistic_threshold)
#         else:
#             ballistic_end = verification_start_idx
#     else:
#         ballistic_end = start_seg

#     # Retrieve segmentation ranges for the selected markers.
#     seg_ranges1 = reach_speed_segments_1[subject][hand][file_path]
#     seg_ranges2 = reach_speed_segments_2[subject][hand][file_path]
#     if seg_index < 0 or seg_index >= len(seg_ranges1) or seg_index >= len(seg_ranges2):
#         raise ValueError("Invalid seg_index provided.")

#     # Retrieve indices for the specific segment.
#     start_idx1, end_idx1 = seg_ranges1[seg_index]
#     start_idx2, end_idx2 = seg_ranges2[seg_index]

#     # Extract trajectory segments.
#     pos_seg1 = position_full[start_idx1:end_idx1]
#     vel_seg1 = velocity_full[start_idx1:end_idx1]
#     acc_seg1 = acceleration_full[start_idx1:end_idx1]
#     jerk_seg1 = np.array(jerk_full[start_idx1:end_idx1])

#     pos_seg2 = position_full[start_idx2:end_idx2]
#     vel_seg2 = velocity_full[start_idx2:end_idx2]
#     acc_seg2 = acceleration_full[start_idx2:end_idx2]
#     jerk_seg2 = np.array(jerk_full[start_idx2:end_idx2])

#     # Process Segment 1 for normalized jerk.
#     duration1 = len(jerk_seg1) / fs
#     t_orig1 = np.linspace(0, duration1, num=len(jerk_seg1))
#     t_std1 = np.linspace(0, duration1, num=target_samples)
#     warped_jerk1 = np.interp(t_std1, t_orig1, jerk_seg1)
#     jerk_squared_integral1 = np.trapezoid(warped_jerk1**2, t_std1)
#     vpeak1 = vel_seg1.max()
#     dimensionless_jerk1 = (duration1**3 / vpeak1**2) * jerk_squared_integral1
#     LDLJ1 = -math.log(abs(dimensionless_jerk1), math.e)

#     # Process Segment 2 for normalized jerk.
#     duration2 = len(jerk_seg2) / fs
#     t_orig2 = np.linspace(0, duration2, num=len(jerk_seg2))
#     t_std2 = np.linspace(0, duration2, num=target_samples)
#     warped_jerk2 = np.interp(t_std2, t_orig2, jerk_seg2)
#     jerk_squared_integral2 = np.trapezoid(warped_jerk2**2, t_std2)
#     vpeak2 = vel_seg2.max()
#     dimensionless_jerk2 = (duration2**3 / vpeak2**2) * jerk_squared_integral2
#     LDLJ2 = -math.log(abs(dimensionless_jerk2), math.e)

#     # Use real frame indices for plotting.
#     x_vals1 = range(start_idx1, end_idx1)
#     x_vals2 = range(start_idx2, end_idx2)

#     # Find filtered local minima of jerk in segment 1.
#     if len(jerk_seg1) > 1:
#         drops1 = np.array([jerk_seg1[i-1] - jerk_seg1[i] for i in range(1, len(jerk_seg1))])
#         max_drop1 = drops1.max() if drops1.size > 0 else 0
#         derivative_threshold1 = 0.01 * max_drop1
#     else:
#         derivative_threshold1 = 0

#     peaks1_jerk, _ = find_peaks(-jerk_seg1)
#     filtered_peaks1 = []
#     for p in peaks1_jerk:
#         if p > 0:
#             drop = jerk_seg1[p-1] - jerk_seg1[p]
#             if drop >= derivative_threshold1:
#                 candidate = p
#                 candidate_abs = start_idx1 + candidate
#                 if not filtered_peaks1:
#                     filtered_peaks1.append(candidate)
#                 else:
#                     current_best = filtered_peaks1[0]
#                     current_best_abs = start_idx1 + current_best
#                     if abs(candidate_abs - ballistic_end) < abs(current_best_abs - ballistic_end) or \
#                        (abs(candidate_abs - ballistic_end) == abs(current_best_abs - ballistic_end) and candidate_abs > ballistic_end):
#                         filtered_peaks1[0] = candidate

#     # Adjust ballistic_end if a filtered peak exists.
#     if filtered_peaks1:
#         ballistic_end = filtered_peaks1[0] + start_idx1

#     return {
#         "marker": marker,
#         "position_full": position_full,
#         "velocity_full": velocity_full,
#         "acceleration_full": acceleration_full,
#         "jerk_full": jerk_full,
#         "coord_x_full": coord_x_full,
#         "coord_y_full": coord_y_full,
#         "coord_z_full": coord_z_full,
#         "start_seg": start_seg,
#         "end_seg": end_seg,
#         "latency_end_idx": latency_end_idx,
#         "ballistic_end": ballistic_end,
#         "verification_start_idx": verification_start_idx,
#         "start_idx1": start_idx1,
#         "end_idx1": end_idx1,
#         "start_idx2": start_idx2,
#         "end_idx2": end_idx2,
#         "pos_seg1": pos_seg1,
#         "vel_seg1": vel_seg1,
#         "acc_seg1": acc_seg1,
#         "jerk_seg1": jerk_seg1,
#         "pos_seg2": pos_seg2,
#         "vel_seg2": vel_seg2,
#         "acc_seg2": acc_seg2,
#         "jerk_seg2": jerk_seg2,
#         "x_vals1": x_vals1,
#         "x_vals2": x_vals2,
#         "LDLJ1": LDLJ1,
#         "LDLJ2": LDLJ2,
#         "filtered_peaks1": filtered_peaks1
#     }


# def calculate_phase_indices_for_file(results, reach_speed_segments_2, subject, hand, file_path):
#     """
#     Calculates two phase segmentation for one reach file for the given subject, hand, and file.
#     Phase1: from reach_speed_segments_2 start to ballistic_end (determined via peak detection on the velocity time window).
#     Phase2: from ballistic_end to reach_speed_segments_2 end.
#     Returns a dictionary structured as:
#       phase_data[segment] = {
#           "phase1": (start, ballistic_end),
#           "phase2": (ballistic_end, end)
#       }
#     """
#     phase_data = {}
#     marker = 'RFIN' if hand == 'right' else 'LFIN'
#     traj_data = results[subject][hand][1][file_path]['traj_space'][marker]
#     velocity_full = traj_data[1]  # time window used for phase detection

#     seg_ranges2 = reach_speed_segments_2[subject][hand][file_path]
    
#     # Loop over each reach segment.
#     for seg_index, seg_range in enumerate(seg_ranges2):
#         start_seg, end_seg = seg_range
        
#         # Determine ballistic_end as the first peak in the negative time window.
#         time_window = velocity_full[start_seg:end_seg]
#         peaks1, _ = find_peaks(-np.array(time_window), prominence=100)
#         if peaks1.size > 0:
#             if peaks1.size > 1:
#                 candidate = peaks1[np.argmax(np.array(time_window)[peaks1])]
#                 ballistic_end = start_seg + candidate
#             else:
#                 ballistic_end = start_seg + peaks1[0]
#         else:
#             ballistic_end = start_seg
        
#         phase_data[seg_index] = ballistic_end
    
#     return phase_data




def plot_trajectory(data):
    """
    Plots 2D subplots overlaying Position, Velocity, Acceleration,
    Original Jerk and Trajectory (X, Y, Z) for two segmentations.
    Vertical lines mark phase transitions.
    """
    fig, axs = plt.subplots(7, 1, figsize=(12, 20))
    
    # Overlay Position.
    axs[0].plot(data["x_vals1"], data["pos_seg1"], color='blue', linewidth=2, label='Segment 1')
    axs[0].plot(data["x_vals2"], data["pos_seg2"], color='lime', linestyle='--', linewidth=2, label='Segment 2')
    axs[0].set_title('Position')
    
    # Overlay Velocity.
    axs[1].plot(data["x_vals1"], data["vel_seg1"], color='blue', linewidth=2, label='Segment 1')
    axs[1].plot(data["x_vals2"], data["vel_seg2"], color='lime', linestyle='--', linewidth=2, label='Segment 2')
    axs[1].set_title('Velocity')
    
    # Overlay Acceleration.
    axs[2].plot(data["x_vals1"], data["acc_seg1"], color='blue', linewidth=2, label='Segment 1')
    axs[2].plot(data["x_vals2"], data["acc_seg2"], color='lime', linestyle='--', linewidth=2, label='Segment 2')
    axs[2].set_title('Acceleration')
    
    # Overlay Original Jerk.
    axs[3].plot(data["x_vals1"], data["jerk_seg1"], color='blue', linewidth=2, label='Segment 1')
    axs[3].plot(data["x_vals2"], data["jerk_seg2"], color='lime', linestyle='--', linewidth=2, label='Segment 2')
    axs[3].set_title('Original Jerk')
    axs[3].set_xlabel("Frame Index", fontsize=12)
    axs[3].legend(loc='upper right')
    
    # Overlay Trajectory X.
    traj_x_seg1 = data["coord_x_full"][data["start_idx1"]:data["end_idx1"]]
    traj_x_seg2 = data["coord_x_full"][data["start_idx2"]:data["end_idx2"]]
    axs[4].plot(data["x_vals1"], traj_x_seg1, color='blue', linewidth=2, label='Segment 1')
    axs[4].plot(data["x_vals2"], traj_x_seg2, color='lime', linestyle='--', linewidth=2, label='Segment 2')
    axs[4].set_title('Trajectory X')
    axs[4].set_xlabel("Frame Index", fontsize=12)
    axs[4].legend(loc='upper right')
    
    # Overlay Trajectory Y.
    traj_y_seg1 = data["coord_y_full"][data["start_idx1"]:data["end_idx1"]]
    traj_y_seg2 = data["coord_y_full"][data["start_idx2"]:data["end_idx2"]]
    axs[5].plot(data["x_vals1"], traj_y_seg1, color='blue', linewidth=2, label='Segment 1')
    axs[5].plot(data["x_vals2"], traj_y_seg2, color='lime', linestyle='--', linewidth=2, label='Segment 2')
    axs[5].set_title('Trajectory Y')
    axs[5].set_xlabel("Frame Index", fontsize=12)
    axs[5].legend(loc='upper right')
    
    # Overlay Trajectory Z.
    traj_z_seg1 = data["coord_z_full"][data["start_idx1"]:data["end_idx1"]]
    traj_z_seg2 = data["coord_z_full"][data["start_idx2"]:data["end_idx2"]]
    axs[6].plot(data["x_vals1"], traj_z_seg1, color='blue', linewidth=2, label='Segment 1')
    axs[6].plot(data["x_vals2"], traj_z_seg2, color='lime', linestyle='--', linewidth=2, label='Segment 2')
    axs[6].set_title('Trajectory Z')
    axs[6].set_xlabel("Frame Index", fontsize=12)
    axs[6].legend(loc='upper right')
    
    # Overlay phase markers on each subplot.
    for ax in axs:
        ax.axvline(data["latency_end_idx"], color='magenta', linestyle=':', label='Latency End')
        ax.axvline(data["ballistic_end"], color='cyan', linestyle=':', label='Ballistic End')
        ax.axvline(data["verification_start_idx"], color='orange', linestyle=':', label='Verification Start')
        for p in data["filtered_peaks1"]:
            ax.axvline(data["start_idx1"] + p, color='purple', linestyle=':', label='Jerk Minima')
    axs[0].legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_3d_trajectory(data):
    """
    Plots a 3D trajectory with phase colors.
    Phases: Latency (magenta), Ballistic (cyan), Correction (green), Verification (orange).
    Also marks start and end points.
    """
    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    
    # Scatter for each phase.
    ax3d.scatter(
        data["coord_x_full"][data["start_seg"]:data["latency_end_idx"]],
        data["coord_y_full"][data["start_seg"]:data["latency_end_idx"]],
        data["coord_z_full"][data["start_seg"]:data["latency_end_idx"]],
        color='magenta', s=20, label='Latency Phase'
    )
    ax3d.scatter(
        data["coord_x_full"][data["latency_end_idx"]:data["ballistic_end"]],
        data["coord_y_full"][data["latency_end_idx"]:data["ballistic_end"]],
        data["coord_z_full"][data["latency_end_idx"]:data["ballistic_end"]],
        color='cyan', s=20, label='Ballistic Phase'
    )
    ax3d.scatter(
        data["coord_x_full"][data["ballistic_end"]:data["verification_start_idx"]],
        data["coord_y_full"][data["ballistic_end"]:data["verification_start_idx"]],
        data["coord_z_full"][data["ballistic_end"]:data["verification_start_idx"]],
        color='green', s=20, label='Correction Phase'
    )
    ax3d.scatter(
        data["coord_x_full"][data["verification_start_idx"]:data["end_seg"]],
        data["coord_y_full"][data["verification_start_idx"]:data["end_seg"]],
        data["coord_z_full"][data["verification_start_idx"]:data["end_seg"]],
        color='orange', s=20, label='Verification Phase'
    )
    
    # Mark start and end points.
    ax3d.scatter(
        data["coord_x_full"][data["start_seg"]],
        data["coord_y_full"][data["start_seg"]],
        data["coord_z_full"][data["start_seg"]],
        color='black', s=50, marker='o', label='Start'
    )
    ax3d.text(
        data["coord_x_full"][data["start_seg"]],
        data["coord_y_full"][data["start_seg"]],
        data["coord_z_full"][data["start_seg"]],
        'Start', color='black'
    )
    ax3d.scatter(
        data["coord_x_full"][data["end_seg"]-1],
        data["coord_y_full"][data["end_seg"]-1],
        data["coord_z_full"][data["end_seg"]-1],
        color='red', s=50, marker='o', label='End'
    )
    ax3d.text(
        data["coord_x_full"][data["end_seg"]-1],
        data["coord_y_full"][data["end_seg"]-1],
        data["coord_z_full"][data["end_seg"]-1],
        'End', color='red'
    )
    
    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")
    ax3d.legend()
    plt.tight_layout()
    plt.show()

def plot_3d_trajectory_icon(data):
    """
    Plots a 3D trajectory with phase colors.
    Phases: Latency (magenta), Ballistic (cyan), Correction (green), Verification (orange).
    Also marks start and end points with an image.
    """
    fig3d = plt.figure(figsize=(10, 8))

    ax3d = fig3d.add_subplot(111, projection='3d')

    
    # Line for each phase.
    ax3d.plot(
        data["coord_x_full"][data["start_seg"]:data["latency_end_idx"]],
        data["coord_y_full"][data["start_seg"]:data["latency_end_idx"]],
        data["coord_z_full"][data["start_seg"]:data["latency_end_idx"]],
        color='magenta', linewidth=4, label='Latency Phase'
    )
    ax3d.plot(
        data["coord_x_full"][data["latency_end_idx"]:data["ballistic_end"]],
        data["coord_y_full"][data["latency_end_idx"]:data["ballistic_end"]],
        data["coord_z_full"][data["latency_end_idx"]:data["ballistic_end"]],
        color='cyan', linewidth=4, label='Ballistic Phase'
    )
    ax3d.plot(
        data["coord_x_full"][data["ballistic_end"]:data["verification_start_idx"]],
        data["coord_y_full"][data["ballistic_end"]:data["verification_start_idx"]],
        data["coord_z_full"][data["ballistic_end"]:data["verification_start_idx"]],
        color='green', linewidth=4, label='Correction Phase'
    )
    ax3d.plot(
        data["coord_x_full"][data["verification_start_idx"]:data["end_seg"]],
        data["coord_y_full"][data["verification_start_idx"]:data["end_seg"]],
        data["coord_z_full"][data["verification_start_idx"]:data["end_seg"]],
        color='orange', linewidth=4, label='Verification Phase'
    )
    # Function to add an image at a 3D point.
    def add_image(ax, xs, ys, zs, img, zoom=0.1):
        x2, y2, _ = proj_transform(xs, ys, zs, ax.get_proj())
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, (x2, y2), frameon=False, xycoords='data')
        ax.add_artist(ab)
    
    # Load PNG image.
    img = mpimg.imread("/Users/yilinwu/Desktop/HandHoldBlock.png")  # Replace with your file path
    
    # Add image at start point.
    add_image(ax3d,
              data["coord_x_full"][data["start_seg"]],
              data["coord_y_full"][data["start_seg"]],
              data["coord_z_full"][data["start_seg"]]+4,
              img, zoom=0.08)
    # Add image at end point.
    add_image(ax3d,
              data["coord_x_full"][data["end_seg"]],
              data["coord_y_full"][data["end_seg"]],
              data["coord_z_full"][data["end_seg"]]+4,
              img, zoom=0.08)

    # Mark start and end points with scatter markers.
    ax3d.scatter(
        data["coord_x_full"][data["start_seg"]],
        data["coord_y_full"][data["start_seg"]],
        data["coord_z_full"][data["start_seg"]],
        color='black', s=50, marker='o', label='Start'
    )
    ax3d.scatter(
        data["coord_x_full"][data["end_seg"]],
        data["coord_y_full"][data["end_seg"]],
        data["coord_z_full"][data["end_seg"]],
        color='red', s=50, marker='o', label='End'
    )    

    offset = 10  # adjust the offset as needed
    ax3d.text(
        data["coord_x_full"][data["start_seg"]] + offset,
        data["coord_y_full"][data["start_seg"]],
        data["coord_z_full"][data["start_seg"]],
        'Start', color='black',
        fontdict={'fontsize': 20, 'fontweight': 'bold'}
    )

    ax3d.text(
        data["coord_x_full"][data["end_seg"]] + offset,
        data["coord_y_full"][data["end_seg"]],
        data["coord_z_full"][data["end_seg"]],
        'End', color='black',
        fontdict={'fontsize': 20, 'fontweight': 'bold'}
    )

    
    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")
    ax3d.legend()
    
    # Configure a nicer 3D view.
    # ax3d.view_init(elev=30, azim=90)
    ax3d.grid(True)

    plt.tight_layout()
    plt.show()

def plot_3d_trajectory_video_with_icon(data, test_window, save_path='trajectory_with_icon.mp4'):
    start_seg, end_seg = test_window

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set fixed axis limits
    ax.set_xlim([np.min(data["coord_x_full"]), np.max(data["coord_x_full"])])
    ax.set_ylim([np.min(data["coord_y_full"]), np.max(data["coord_y_full"])])
    ax.set_zlim([np.min(data["coord_z_full"]), np.max(data["coord_z_full"])])

    # Load PNG image
    img = mpimg.imread("/Users/yilinwu/Desktop/HandHoldBlock.png")  # replace with your path

    # Add PNG image at start point
    start_proj = proj3d.proj_transform(data["coord_x_full"][start_seg],
                                       data["coord_y_full"][start_seg],
                                       data["coord_z_full"][start_seg] + 4, ax.get_proj())
    start_ab = AnnotationBbox(OffsetImage(img, zoom=0.03),
                              (start_proj[0], start_proj[1]),
                              frameon=False, xycoords='data')
    ax.add_artist(start_ab)

    # Add PNG image at end point
    end_proj = proj3d.proj_transform(data["coord_x_full"][end_seg],
                                     data["coord_y_full"][end_seg],
                                     data["coord_z_full"][end_seg] + 4, ax.get_proj())
    end_ab = AnnotationBbox(OffsetImage(img, zoom=0.03),
                            (end_proj[0], end_proj[1]),
                            frameon=False, xycoords='data')
    ax.add_artist(end_ab)

    # Scatter start and end points
    ax.scatter(data["coord_x_full"][start_seg],
               data["coord_y_full"][start_seg],
               data["coord_z_full"][start_seg],
               color='black', s=50, marker='o', label='Start')
    ax.scatter(data["coord_x_full"][end_seg],
               data["coord_y_full"][end_seg],
               data["coord_z_full"][end_seg],
               color='red', s=50, marker='o', label='End')

    # Text labels
    offset = 10
    ax.text(data["coord_x_full"][start_seg] + offset,
            data["coord_y_full"][start_seg],
            data["coord_z_full"][start_seg],
            'Start', color='black', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    ax.text(data["coord_x_full"][end_seg] + offset,
            data["coord_y_full"][end_seg],
            data["coord_z_full"][end_seg],
            'End', color='black', fontdict={'fontsize': 20, 'fontweight': 'bold'})

    # Set labels
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.legend()
    ax.grid(True)

    # Lines for phases
    latency_line, = ax.plot([], [], [], color='magenta', linewidth=4, label='Latency Phase')
    ballistic_line, = ax.plot([], [], [], color='cyan', linewidth=4, label='Ballistic Phase')
    correction_line, = ax.plot([], [], [], color='green', linewidth=4, label='Correction Phase')
    verification_line, = ax.plot([], [], [], color='orange', linewidth=4, label='Verification Phase')

    # Animation function without dynamic icon update
    def update(frame):
        if frame < data["latency_end_idx"]:
            latency_line.set_data(data["coord_x_full"][start_seg:frame],
                                  data["coord_y_full"][start_seg:frame])
            latency_line.set_3d_properties(data["coord_z_full"][start_seg:frame])
        elif frame < data["ballistic_end"]:
            latency_line.set_data(data["coord_x_full"][start_seg:data["latency_end_idx"]],
                                  data["coord_y_full"][start_seg:data["latency_end_idx"]])
            latency_line.set_3d_properties(data["coord_z_full"][start_seg:data["latency_end_idx"]])
            ballistic_line.set_data(data["coord_x_full"][data["latency_end_idx"]:frame],
                                    data["coord_y_full"][data["latency_end_idx"]:frame])
            ballistic_line.set_3d_properties(data["coord_z_full"][data["latency_end_idx"]:frame])
        elif frame < data["verification_start_idx"]:
            ballistic_line.set_data(data["coord_x_full"][data["latency_end_idx"]:data["ballistic_end"]],
                                    data["coord_y_full"][data["latency_end_idx"]:data["ballistic_end"]])
            ballistic_line.set_3d_properties(data["coord_z_full"][data["latency_end_idx"]:data["ballistic_end"]])
            correction_line.set_data(data["coord_x_full"][data["ballistic_end"]:frame],
                                     data["coord_y_full"][data["ballistic_end"]:frame])
            correction_line.set_3d_properties(data["coord_z_full"][data["ballistic_end"]:frame])
        else:
            correction_line.set_data(data["coord_x_full"][data["ballistic_end"]:data["verification_start_idx"]],
                                     data["coord_y_full"][data["ballistic_end"]:data["verification_start_idx"]])
            correction_line.set_3d_properties(data["coord_z_full"][data["ballistic_end"]:data["verification_start_idx"]])
            verification_line.set_data(data["coord_x_full"][data["verification_start_idx"]:frame],
                                       data["coord_y_full"][data["verification_start_idx"]:frame])
            verification_line.set_3d_properties(data["coord_z_full"][data["verification_start_idx"]:frame])

        return latency_line, ballistic_line, correction_line, verification_line

    # Animate only the test window segment
    ani = FuncAnimation(fig, update, frames=range(start_seg, end_seg), interval=30, blit=False)

    # Save animation
    writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(save_path, writer=writer)
    plt.close(fig)

def run_analysis_and_plot(results, reach_speed_segments, test_windows,
                          subject, hand, file_path,
                          fs=200, target_samples=101, seg_index=3,
                          dis_phases=0.3, video_save_path='trajectory_with_icon_3.mp4'):
    data = calculate_phase_indices(results, reach_speed_segments, test_windows,
                                   subject=subject, hand=hand, file_path=file_path,
                                   fs=fs, target_samples=target_samples,
                                   seg_index=seg_index, dis_phases=dis_phases)
    plot_trajectory(data)
    plot_3d_trajectory(data)
    plot_3d_trajectory_icon(data)
    test_window = test_windows[subject][hand][file_path][seg_index]
    plot_3d_trajectory_video_with_icon(data, test_window, save_path=video_save_path)
    
# # Example usage:
# run_analysis_and_plot(results, reach_speed_segments, test_windows_7,
#                       subject="07/22/HW", hand="left",
#                       file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv',
#                       fs=200, target_samples=101, seg_index=3, dis_phases=0.3,
#                       video_save_path='trajectory_with_icon_3.mp4')

# def calculate_phase_indices_all_files(results, reach_speed_segments_1, reach_speed_segments_2, fs=200, target_samples=101, dis_phases=0.3):
#     """
#     Calculates phase indices and related segmentation data for all reach segments
#     across all files for all subjects and hands.
#     Returns a dictionary structured as:
#       all_phase_data[subject][hand][file][segment] = {
#           "latency_end_idx": ...,
#           "ballistic_end": ...,
#           "verification_start_idx": ...
#       }
#     """
#     all_phase_data = {}

#     # Iterate over all subjects.
#     for subject in results:
#         all_phase_data[subject] = {}
#         # Iterate over all hands for the subject.
#         for hand in results[subject]:
#             all_phase_data[subject][hand] = {}
#             # Loop over every file for the specified subject and hand.
#             # Assuming that results[subject][hand] is a dict and that trial 1 holds the file paths.
#             for file_path in results[subject][hand][1]:
#                 # Extract trajectory arrays.
#                 marker = 'RFIN' if hand == 'right' else 'LFIN'
#                 traj_data = results[subject][hand][1][file_path]['traj_space'][marker]
#                 position_full = traj_data[0]
#                 velocity_full = traj_data[1]
#                 acceleration_full = traj_data[2]
#                 jerk_full = traj_data[3]

                
#                 # Extract coordinate arrays.
#                 traj_data_full = results[subject][hand][1][file_path]['traj_data']
#                 coord_prefix = "RFIN_" if hand == 'right' else "LFIN_"
#                 coord_x_full = np.array(traj_data_full[coord_prefix + "X"])
#                 coord_y_full = np.array(traj_data_full[coord_prefix + "Y"])
#                 coord_z_full = np.array(traj_data_full[coord_prefix + "Z"])

#                 coord_ax_full = np.array(traj_data_full[coord_prefix + "AX"])
#                 coord_jx_full = traj_data_full[coord_prefix + "AX"].diff() * 200  


                
#                 # Retrieve segmentation ranges for the selected file.
#                 seg_ranges1 = reach_speed_segments_1[subject][hand][file_path]
#                 seg_ranges2 = reach_speed_segments_2[subject][hand][file_path]
                
#                 file_phase_data = {}
                
#                 # Loop over every reach segment.
#                 for seg_index in range(len(seg_ranges2)):
#                     # Use the segmentation from reach_speed_segments_2 to get segment start and end.
#                     seg_range = seg_ranges2[seg_index]
#                     start_seg, end_seg = seg_range
                    
#                     # Compute Latency phase based on cumulative differences of the x-coordinate.
#                     coord_x_seg = coord_x_full[start_seg:end_seg]
#                     cum_dist = np.insert(np.cumsum(np.abs(np.diff(coord_x_seg))), 0, 0)
#                     if np.any(cum_dist > dis_phases):
#                         latency_end_idx = start_seg + np.argmax(cum_dist > dis_phases)
#                     else:
#                         latency_end_idx = start_seg
                    
#                     # Compute Verification phase from the end.
#                     rev_cum = np.insert(np.cumsum(np.abs(np.diff(coord_x_seg[::-1]))), 0, 0)
#                     if np.any(rev_cum > dis_phases):
#                         verification_start_idx = end_seg - np.argmax(rev_cum > dis_phases)
#                     else:
#                         verification_start_idx = end_seg - 1
#                     # Compute Ballistic phase using cumulative differences.
#                     if len(coord_x_seg) > 1:
#                         ballistic_diffs = np.abs(np.diff(coord_x_seg))
#                         ballistic_total = np.sum(ballistic_diffs)
#                         ballistic_threshold = 0.25 * ballistic_total
#                         ballistic_cum = np.insert(np.cumsum(ballistic_diffs), 0, 0)
#                         indices = np.where(ballistic_cum > ballistic_threshold)[0]
#                         if indices.size > 0:
#                             ballistic_end = start_seg + indices[0]
#                         else:
#                             raise ValueError("Ballistic phase could not be determined for segment index {}".format(seg_index))
                    
#                     # Retrieve indices for the specific segment.
#                     start_idx1, end_idx1 = seg_ranges1[seg_index]
#                     start_idx2, end_idx2 = seg_ranges2[seg_index]
                    
#                     # Extract trajectory segments.
#                     pos_seg1 = position_full[start_idx1:end_idx1]
#                     vel_seg1 = velocity_full[start_idx1:end_idx1]
#                     acc_seg1 = acceleration_full[start_idx1:end_idx1]
#                     jerk_seg1 = np.array(jerk_full[start_idx1:end_idx1])
                    
#                     pos_seg2 = position_full[start_idx2:end_idx2]
#                     vel_seg2 = velocity_full[start_idx2:end_idx2]
#                     acc_seg2 = acceleration_full[start_idx2:end_idx2]
#                     jerk_seg2 = np.array(jerk_full[start_idx2:end_idx2])
                    
#                     # Process Segment 1 for normalized jerk.
#                     duration1 = len(jerk_seg1) / fs
#                     t_orig1 = np.linspace(0, duration1, num=len(jerk_seg1))
#                     t_std1 = np.linspace(0, duration1, num=target_samples)
#                     warped_jerk1 = np.interp(t_std1, t_orig1, jerk_seg1)
#                     jerk_squared_integral1 = np.trapezoid(warped_jerk1**2, t_std1)
#                     vpeak1 = vel_seg1.max()
#                     dimensionless_jerk1 = (duration1**3 / vpeak1**2) * jerk_squared_integral1
#                     LDLJ1 = -math.log(abs(dimensionless_jerk1), math.e)
                    
#                     # Process Segment 2 for normalized jerk.
#                     duration2 = len(jerk_seg2) / fs
#                     t_orig2 = np.linspace(0, duration2, num=len(jerk_seg2))
#                     t_std2 = np.linspace(0, duration2, num=target_samples)
#                     warped_jerk2 = np.interp(t_std2, t_orig2, jerk_seg2)
#                     jerk_squared_integral2 = np.trapezoid(warped_jerk2**2, t_std2)
#                     vpeak2 = vel_seg2.max()
#                     dimensionless_jerk2 = (duration2**3 / vpeak2**2) * jerk_squared_integral2
#                     LDLJ2 = -math.log(abs(dimensionless_jerk2), math.e)
                    
#                     # Use real frame indices for plotting.
#                     x_vals1 = list(range(start_idx1, end_idx1))
#                     x_vals2 = list(range(start_idx2, end_idx2))

#                     # coord_jx_seg = coord_jx_full[start_idx2:end_idx2]
#                     # # Find zero crossing indices in the coordinate jerk segment
#                     # zero_crossings = np.where(np.diff(np.sign(coord_jx_seg)))[0]
#                     # # Filter zero crossings to be within the ballistic phase
#                     # filtered_peaks1 = [zc for zc in zero_crossings if start_seg + zc > ballistic_end]

                
#                     # # Find filtered local minima of jerk in segment 1.
#                     # if len(jerk_seg2) > 1:
#                     #     drops1 = np.array([jerk_seg2[i-1] - jerk_seg2[i] for i in range(1, len(jerk_seg2))])
#                     #     max_drop1 = drops1.max() if drops1.size > 0 else 0
#                     #     derivative_threshold1 = 0.01 * max_drop1
#                     # else:
#                     #     derivative_threshold1 = 0
                    
#                     # peaks1_jerk, _ = find_peaks(-jerk_seg2)
#                     # filtered_peaks1 = []
#                     # for p in peaks1_jerk:
#                     #     if p > 0:
#                     #         drop = jerk_seg2[p-1] - jerk_seg2[p]
#                     #         if drop >= derivative_threshold1:
#                     #             candidate = p
#                     #             candidate_abs = start_idx2 + candidate
#                     #             if not filtered_peaks1:
#                     #                 filtered_peaks1.append(candidate)
#                     #             else:
#                     #                 current_best = filtered_peaks1[0]
#                     #                 current_best_abs = start_idx2 + current_best
#                     #                 if (abs(candidate_abs - ballistic_end) < abs(current_best_abs - ballistic_end)) or \
#                     #                    (abs(candidate_abs - ballistic_end) == abs(current_best_abs - ballistic_end) and candidate_abs > ballistic_end):
#                     #                     filtered_peaks1[0] = candidate




#                     # --- Smooth jerk to reduce noise ---
#                     if len(jerk_seg2) > 5:  # need at least window_length points
#                         jerk_smooth = savgol_filter(jerk_seg2, window_length=11, polyorder=3)
#                     else:
#                         jerk_smooth = jerk_seg2

#                     # --- Find local minima (peaks of -jerk) with prominence filtering ---
#                     minima, props = find_peaks(-jerk_smooth, prominence=0.05 * np.max(np.abs(jerk_smooth)))
#                     minima = np.array([p for p in minima if (start_idx2 + p) > (ballistic_end + latency_end_idx)])



#                     filtered_peaks1 = []
#                     best_peak = None

#                     if len(minima) > 0:
#                         scores = []
#                         max_vel = np.max(vel_seg2)  # velocity in the same segment
#                         vel_thresh = 0.6 * max_vel       # top 60% of velocity

#                         for p in minima:
#                             if p > 0:
#                                 candidate_abs = start_idx2 + p

#                                 # --- Apply constraints ---
#                                 if not (latency_end_idx < candidate_abs < verification_start_idx):
#                                     continue  # must be between latency_end and verification_start
#                                 if vel_seg2.iloc[p] < vel_thresh:
#                                     continue  # must be in top 60% velocity

#                                 # --- Scoring ---
#                                 drop = jerk_smooth[p-1] - jerk_smooth[p]   # depth of the minimum
#                                 dist = abs(candidate_abs - ballistic_end)  # distance to ballistic_end

#                                 score = drop / (dist + 1e-6)  # deeper + closer = better
#                                 scores.append((score, p))

#                         if scores:
#                             best_peak = max(scores, key=lambda x: x[0])[1]
#                             filtered_peaks1 = [best_peak]

#                     # Adjust ballistic_end if a filtered peak exists.
#                     if filtered_peaks1:
#                         ballistic_end = filtered_peaks1[0] + latency_end_idx
                    
#                     # file_phase_data[seg_index] = {
#                     #     "latency_end_idx": latency_end_idx,
#                     #     "ballistic_end": ballistic_end,
#                     #     "verification_start_idx": verification_start_idx
#                     # }
#                     file_phase_data[seg_index] = ballistic_end

#                 all_phase_data[subject][hand][file_path] = file_phase_data

#     return all_phase_data






def plot_3d_trajectory_icon_all_phase_data(results, reach_speed_segments, test_windows,
                                           subject, hand, file_path, seg_index=3,
                                           fs=200, target_samples=101, dis_phases=0.3,
                                           show_icon=True):
    """
    Computes phase indices directly from results and plots a 3D trajectory with phase colors.
    This modified version calls calculate_phase_indices internally.
    Optionally shows icons if show_icon is True.
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
    data = calculate_phase_indices(results, reach_speed_segments, test_windows,
                                   subject=subject, hand=hand, file_path=file_path,
                                   fs=fs, target_samples=target_samples,
                                   seg_index=seg_index, dis_phases=dis_phases)

    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')

    # Plot phase lines using indices computed from data.
    ax3d.plot(
        data["coord_x_full"][data["start_seg"]:data["latency_end_idx"]],
        data["coord_y_full"][data["start_seg"]:data["latency_end_idx"]],
        data["coord_z_full"][data["start_seg"]:data["latency_end_idx"]],
        color='magenta', linewidth=4, label='Latency Phase'
    )
    ax3d.plot(
        data["coord_x_full"][data["latency_end_idx"]:data["ballistic_end"]],
        data["coord_y_full"][data["latency_end_idx"]:data["ballistic_end"]],
        data["coord_z_full"][data["latency_end_idx"]:data["ballistic_end"]],
        color='cyan', linewidth=4, label='Ballistic Phase'
    )
    ax3d.plot(
        data["coord_x_full"][data["ballistic_end"]:data["verification_start_idx"]],
        data["coord_y_full"][data["ballistic_end"]:data["verification_start_idx"]],
        data["coord_z_full"][data["ballistic_end"]:data["verification_start_idx"]],
        color='green', linewidth=4, label='Correction Phase'
    )
    ax3d.plot(
        data["coord_x_full"][data["verification_start_idx"]:data["end_seg"]],
        data["coord_y_full"][data["verification_start_idx"]:data["end_seg"]],
        data["coord_z_full"][data["verification_start_idx"]:data["end_seg"]],
        color='orange', linewidth=4, label='Verification Phase'
    )

    # Helper function to add an image at a 3D point.
    def add_image(ax, xs, ys, zs, img, zoom=0.1):
        x2, y2, _ = proj_transform(xs, ys, zs, ax.get_proj())
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, (x2, y2), frameon=False, xycoords='data')
        ax.add_artist(ab)

    # If the option is enabled, load the PNG image and add icons.
    if show_icon:
        img = mpimg.imread("/Users/yilinwu/Desktop/HandHoldBlock.png")
        add_image(ax3d,
                  data["coord_x_full"][data["start_seg"]],
                  data["coord_y_full"][data["start_seg"]],
                  data["coord_z_full"][data["start_seg"]] + 4,
                  img, zoom=0.08)
        add_image(ax3d,
                  data["coord_x_full"][data["end_seg"]],
                  data["coord_y_full"][data["end_seg"]],
                  data["coord_z_full"][data["end_seg"]] + 4,
                  img, zoom=0.08)

    # Mark start and end points with scatter markers.
    ax3d.scatter(
        data["coord_x_full"][data["start_seg"]],
        data["coord_y_full"][data["start_seg"]],
        data["coord_z_full"][data["start_seg"]],
        color='black', s=50, marker='o', label='Start'
    )
    ax3d.scatter(
        data["coord_x_full"][data["end_seg"]],
        data["coord_y_full"][data["end_seg"]],
        data["coord_z_full"][data["end_seg"]],
        color='red', s=50, marker='o', label='End'
    )

    offset = 10  # adjust as needed
    ax3d.text(
        data["coord_x_full"][data["start_seg"]] + offset,
        data["coord_y_full"][data["start_seg"]],
        data["coord_z_full"][data["start_seg"]],
        'Start', color='black',
        fontdict={'fontsize': 20, 'fontweight': 'bold'}
    )
    ax3d.text(
        data["coord_x_full"][data["end_seg"]] + offset,
        data["coord_y_full"][data["end_seg"]],
        data["coord_z_full"][data["end_seg"]],
        'End', color='black',
        fontdict={'fontsize': 20, 'fontweight': 'bold'}
    )

    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")
    ax3d.legend()
    ax3d.grid(True)

    plt.tight_layout()
    plt.show()

def plot_trajectory_all_phase_data(results, reach_speed_segments, test_windows_7, all_phase_data,
                             subject, hand, file_path, seg_index):
    """
    Plots 2D subplots overlaying Position, Velocity, Acceleration,
    Original Jerk and Trajectory (X, Y, Z) for two segmentations.
    For the first segmentation, indices come from reach_speed_segments;
    for the second segmentation, indices come from test_windows_7.
    The phase markers (Latency End, Ballistic End, Verification Start) are taken from all_phase_data.
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
    
    # Plotting the subplots.
    fig, axs = plt.subplots(7, 1, figsize=(6, 10))
    
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
    axs[1].set_title('Velocity', fontsize=tittle_fontsize)
    
    # Overlay Acceleration.
    axs[2].plot(x_vals1, acc_seg1, color='blue', linewidth=2, label='Segment 1')
    axs[2].plot(x_vals2, acc_seg2, color='lime', linestyle='--', linewidth=2, label='Segment 2')
    axs[2].set_title('Acceleration', fontsize=tittle_fontsize)
    
    # Overlay Original Jerk.
    axs[3].plot(x_vals1, jerk_seg1, color='blue', linewidth=2, label='Segment 1')
    axs[3].plot(x_vals2, jerk_seg2, color='lime', linestyle='--', linewidth=2, label='Segment 2')
    axs[3].set_title('Original Jerk', fontsize=tittle_fontsize)
    
    # Overlay Trajectory X.
    traj_x_seg1 = coord_x_full[start_idx1:end_idx1]
    traj_x_seg2 = coord_x_full[start_idx2:end_idx2]
    axs[4].plot(x_vals1, traj_x_seg1, color='blue', linewidth=2, label='Segment 1')
    axs[4].plot(x_vals2, traj_x_seg2, color='lime', linestyle='--', linewidth=2, label='Segment 2')
    axs[4].set_title('Trajectory X', fontsize=tittle_fontsize)
    
    # Overlay Trajectory Y.
    traj_y_seg1 = coord_y_full[start_idx1:end_idx1]
    traj_y_seg2 = coord_y_full[start_idx2:end_idx2]
    axs[5].plot(x_vals1, traj_y_seg1, color='blue', linewidth=2, label='Segment 1')
    axs[5].plot(x_vals2, traj_y_seg2, color='lime', linestyle='--', linewidth=2, label='Segment 2')
    axs[5].set_title('Trajectory Y', fontsize=tittle_fontsize)
    
    # Overlay Trajectory Z.
    traj_z_seg1 = coord_z_full[start_idx1:end_idx1]
    traj_z_seg2 = coord_z_full[start_idx2:end_idx2]
    axs[6].plot(x_vals1, traj_z_seg1, color='blue', linewidth=2, label='Segment 1')
    axs[6].plot(x_vals2, traj_z_seg2, color='lime', linestyle='--', linewidth=2, label='Segment 2')
    axs[6].set_title('Trajectory Z', fontsize=tittle_fontsize)
    axs[6].set_xlabel("Frame Index", fontsize=label_fontsize)

    
    # Overlay phase markers on each subplot.
    for ax in axs:
        ax.axvline(latency_end_idx, color='magenta', linestyle=':', label='Latency End')
        ax.axvline(ballistic_end, color='cyan', linestyle=':', label='Ballistic End')
        ax.axvline(verification_start_idx, color='orange', linestyle=':', label='Verification Start')
        if ax == axs[6]:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), fontsize=legend_fontsize, ncol=len(ax.get_legend_handles_labels()[0]))
    plt.tight_layout()
    plt.show()

def plot_3d_trajectory_video_with_icon_all_phase_data(results, reach_speed_segments, test_windows_7, all_phase_data,
                                                subject="07/22/HW", hand="left",
                                                file_path="/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv",
                                                seg_index=3,
                                                fs=200, target_samples=101, dis_phases=0.3,
                                                video_save_path="trajectory_with_icon_3.mp4"):
    # Compute reconstruction data using the existing segmentation functions.
    data = calculate_phase_indices(results, reach_speed_segments, test_windows_7,
                                   subject=subject, hand=hand, file_path=file_path,
                                   fs=fs, target_samples=target_samples,
                                   seg_index=seg_index, dis_phases=dis_phases)
    
    # Get the test window boundaries from test_windows_7.
    test_window = test_windows_7[subject][hand][file_path][seg_index]
    start_seg, end_seg = test_window
    
    # Additionally, obtain phase markers from all_phase_data.
    phase = all_phase_data[subject][hand][file_path][seg_index]
    # (phase contains keys: 'latency_end_idx', 'ballistic_end', 'verification_start_idx')
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # Set fixed axis limits.
    ax.set_xlim([np.min(data["coord_x_full"]), np.max(data["coord_x_full"])])
    ax.set_ylim([np.min(data["coord_y_full"]), np.max(data["coord_y_full"])])
    ax.set_zlim([np.min(data["coord_z_full"]), np.max(data["coord_z_full"])])
    
    # Load PNG image.
    img = mpimg.imread("/Users/yilinwu/Desktop/HandHoldBlock.png")  # replace with your actual path
    
    # Add PNG image at start point.
    start_proj = proj3d.proj_transform(data["coord_x_full"][start_seg],
                                       data["coord_y_full"][start_seg],
                                       data["coord_z_full"][start_seg] + 4, ax.get_proj())
    start_ab = AnnotationBbox(OffsetImage(img, zoom=0.03),
                              (start_proj[0], start_proj[1]),
                              frameon=False, xycoords="data")
    ax.add_artist(start_ab)
    
    # Add PNG image at end point.
    end_proj = proj3d.proj_transform(data["coord_x_full"][end_seg],
                                     data["coord_y_full"][end_seg],
                                     data["coord_z_full"][end_seg] + 4, ax.get_proj())
    end_ab = AnnotationBbox(OffsetImage(img, zoom=0.03),
                            (end_proj[0], end_proj[1]),
                            frameon=False, xycoords="data")
    ax.add_artist(end_ab)
    
    # Scatter start and end points.
    ax.scatter(data["coord_x_full"][start_seg],
               data["coord_y_full"][start_seg],
               data["coord_z_full"][start_seg],
               color="black", s=50, marker="o", label="Start")
    ax.scatter(data["coord_x_full"][end_seg],
               data["coord_y_full"][end_seg],
               data["coord_z_full"][end_seg],
               color="red", s=50, marker="o", label="End")
    
    # Text labels.
    offset = 10
    ax.text(data["coord_x_full"][start_seg] + offset,
            data["coord_y_full"][start_seg],
            data["coord_z_full"][start_seg],
            "Start", color="black", fontdict={"fontsize": 20, "fontweight": "bold"})
    ax.text(data["coord_x_full"][end_seg] + offset,
            data["coord_y_full"][end_seg],
            data["coord_z_full"][end_seg],
            "End", color="black", fontdict={"fontsize": 20, "fontweight": "bold"})
    
    # Set labels.
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.legend()
    ax.grid(True)
    
    # Define lines for phases.
    latency_line, = ax.plot([], [], [], color="magenta", linewidth=4, label="Latency Phase")
    ballistic_line, = ax.plot([], [], [], color="cyan", linewidth=4, label="Ballistic Phase")
    correction_line, = ax.plot([], [], [], color="green", linewidth=4, label="Correction Phase")
    verification_line, = ax.plot([], [], [], color="orange", linewidth=4, label="Verification Phase")
    
    # Animation update function.
    def update(frame):
        if frame < data["latency_end_idx"]:
            latency_line.set_data(data["coord_x_full"][start_seg:frame],
                                  data["coord_y_full"][start_seg:frame])
            latency_line.set_3d_properties(data["coord_z_full"][start_seg:frame])
        elif frame < data["ballistic_end"]:
            latency_line.set_data(data["coord_x_full"][start_seg:data["latency_end_idx"]],
                                  data["coord_y_full"][start_seg:data["latency_end_idx"]])
            latency_line.set_3d_properties(data["coord_z_full"][start_seg:data["latency_end_idx"]])
            ballistic_line.set_data(data["coord_x_full"][data["latency_end_idx"]:frame],
                                    data["coord_y_full"][data["latency_end_idx"]:frame])
            ballistic_line.set_3d_properties(data["coord_z_full"][data["latency_end_idx"]:frame])
        elif frame < data["verification_start_idx"]:
            ballistic_line.set_data(data["coord_x_full"][data["latency_end_idx"]:data["ballistic_end"]],
                                    data["coord_y_full"][data["latency_end_idx"]:data["ballistic_end"]])
            ballistic_line.set_3d_properties(data["coord_z_full"][data["latency_end_idx"]:data["ballistic_end"]])
            correction_line.set_data(data["coord_x_full"][data["ballistic_end"]:frame],
                                     data["coord_y_full"][data["ballistic_end"]:frame])
            correction_line.set_3d_properties(data["coord_z_full"][data["ballistic_end"]:frame])
        else:
            correction_line.set_data(data["coord_x_full"][data["ballistic_end"]:data["verification_start_idx"]],
                                     data["coord_y_full"][data["ballistic_end"]:data["verification_start_idx"]])
            correction_line.set_3d_properties(data["coord_z_full"][data["ballistic_end"]:data["verification_start_idx"]])
            verification_line.set_data(data["coord_x_full"][data["verification_start_idx"]:frame],
                                       data["coord_y_full"][data["verification_start_idx"]:frame])
            verification_line.set_3d_properties(data["coord_z_full"][data["verification_start_idx"]:frame])
        return latency_line, ballistic_line, correction_line, verification_line
    
    # Animate only over the test window segment.
    ani = FuncAnimation(fig, update, frames=range(start_seg, end_seg), interval=30, blit=False)
    
    # Save animation.
    writer = FFMpegWriter(fps=30, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(video_save_path, writer=writer)
    plt.close(fig)


import numpy as np
from scipy.signal import find_peaks, savgol_filter

def calculate_ballistic_end_all_files(results, reach_speed_segments_1, reach_speed_segments_2):
    """
    Detects ballistic_end indices (with jerk-based refinement) for all reach segments
    across all files for all subjects and hands.
    
    Returns:
      all_ballistic_data[subject][hand][file][segment] = ballistic_end
    """
    all_ballistic_data = {}

    for subject in results:
        all_ballistic_data[subject] = {}
        for hand in results[subject]:
            all_ballistic_data[subject][hand] = {}
            
            for file_path in results[subject][hand][1]:
                # Extract trajectory arrays
                marker = 'RFIN' if hand == 'right' else 'LFIN'
                traj_data = results[subject][hand][1][file_path]['traj_space'][marker]
                position_full = traj_data[0]
                velocity_full = traj_data[1]
                jerk_full = traj_data[3]

                seg_ranges1 = reach_speed_segments_1[subject][hand][file_path]
                seg_ranges2 = reach_speed_segments_2[subject][hand][file_path]

                file_ballistic_data = {}
                
                for seg_index, seg_range in enumerate(seg_ranges2):
                    start_seg, end_seg = seg_range

                    # --- Initial ballistic_end estimate (displacement-based) ---
                    coord_x_seg = np.array(position_full[start_seg:end_seg])
                    if len(coord_x_seg) > 1:
                        ballistic_diffs = np.abs(np.diff(coord_x_seg))
                        ballistic_total = np.sum(ballistic_diffs)
                        ballistic_threshold = 0.25 * ballistic_total
                        ballistic_cum = np.insert(np.cumsum(ballistic_diffs), 0, 0)
                        indices = np.where(ballistic_cum > ballistic_threshold)[0]
                        if indices.size > 0:
                            ballistic_end = start_seg + indices[0]
                        else:
                            ballistic_end = start_seg
                    else:
                        ballistic_end = start_seg


                    # --- Refinement step using jerk ---
                    start_idx2, end_idx2 = seg_ranges2[seg_index]
                    jerk_seg2 = np.array(jerk_full[start_idx2:end_idx2])
                    vel_seg2 = np.array(velocity_full[start_idx2:end_idx2])  # <--- FIX HERE

                    if len(jerk_seg2) > 5:
                        jerk_smooth = savgol_filter(jerk_seg2, window_length=11, polyorder=3)
                    else:
                        jerk_smooth = jerk_seg2

                    minima, _ = find_peaks(-jerk_smooth, prominence=0.05 * np.max(np.abs(jerk_smooth)))

                    filtered_peaks = []
                    best_peak = None

                    if len(minima) > 0:
                        scores = []
                        max_vel = np.max(vel_seg2)
                        vel_thresh = 0.6 * max_vel  # only consider top 60% velocity

                        for p in minima:
                            candidate_abs = start_idx2 + p

                            # Require high velocity
                            if vel_seg2[p] < vel_thresh:
                                continue

                            # Score candidate: deeper minima + closer to ballistic_end is better
                            drop = jerk_smooth[p-1] - jerk_smooth[p]
                            dist = abs(candidate_abs - ballistic_end)
                            score = drop / (dist + 1e-6)

                            scores.append((score, p))

                        if scores:
                            best_peak = max(scores, key=lambda x: x[0])[1]
                            filtered_peaks = [best_peak]

                    if filtered_peaks:
                        ballistic_end = start_idx2 + filtered_peaks[0]

                    file_ballistic_data[seg_index] = ballistic_end

                all_ballistic_data[subject][hand][file_path] = file_ballistic_data

    return all_ballistic_data
