import utils
import pprint
import numpy as np

# sBBT_files = utils.find_bbt_files(Traj_folder, date, file_type = "sBBT")
# print("sBBT files:", sBBT_files)

# iBBT_files = utils.find_bbt_files(Traj_folder, date, file_type = "iBBT")
# print("iBBT files:", iBBT_files)


date = "06/19/1" 
time_window_method = 1  # Choose 1, 2, or 3
window_size = 0.05  # 50 ms
prominence_threshold_speed = 500  # Adjust as needed
speed_threshold = 500  # Adjust as needed

Traj_folder = "/Volumes/MNHS-MoCap/Yilin-Honours/Subject/Traj/2025/"
Box_Traj_folder = "/Volumes/MNHS-MoCap/Yilin-Honours/Box/Traj/2025/"
save_path = "/Users/yilinwu/Desktop/honours/Thesis/figure"  # Adjust the save path as needed

tBBT_files = utils.find_bbt_files(Traj_folder, date, file_type = "tBBT")
print(f"tBBT files: {tBBT_files}\nNumber of tBBT files: {len(tBBT_files)}")
if len(tBBT_files) != 64:
    raise ValueError(f"Expected 64 tBBT files, but found {len(tBBT_files)}")

Box_Traj_file = utils.find_bbt_files(Box_Traj_folder, date, file_type = "BOX_Cali")
print("Box Trajectory files:", Box_Traj_file)
print("Number of Box Trajectory files:", len(Box_Traj_file))
if len(Box_Traj_file) != 1:
    raise ValueError(f"Expected 1 Box Trajectory file, but found {len(Box_Traj_file)}")

# Select only files at odd indices (right hand)
file_paths = [file for i, file in enumerate(tBBT_files) if i % 2 == 0] 

marker_name = "RFIN"  # Adjust marker name as needed

# # Select only files at even indices (left hand)
# file_paths = [file for i, file in enumerate(tBBT_files) if i % 2 == 1]  
# marker_name = "LFIN"  # Adjust marker name as needed

results = utils.process_files(file_paths, Box_Traj_file, marker_name, prominence_threshold_speed, speed_threshold, window_size, time_window_method, save_path)

# utils.plot_files(results, file_paths, marker_name, prominence_threshold_speed, speed_threshold, save_path)


# import os
# import matplotlib.pyplot as plt

# # Create one figure and three subplots (one per metric)
# fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
# colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']  # Add more if needed

# for idx, file_path in enumerate(file_paths):
#     # Access data
#     X = results[file_path]['traj_data'][f"{marker_name}_X"]
#     position, speed, _, _ = results[file_path]['Traj_Space_data'][marker_name]
#     time = results[file_path]['time']
    
#     label = os.path.basename(file_path)

#     # Plot all three on their respective axes
#     axs[0].plot(time, X, label=label, color=colors[idx % len(colors)])
#     axs[1].plot(time, position, label=label, color=colors[idx % len(colors)])
#     axs[2].plot(time, speed, label=label, color=colors[idx % len(colors)])

# # Set titles, labels, etc.
# titles = [f"{marker_name} X Coordinate", f"{marker_name} Position", f"{marker_name} Speed"]
# ylabels = ["X (mm)", "Position (mm)", "Speed (mm/s)"]

# for ax, title, ylabel in zip(axs, titles, ylabels):
#     ax.set_title(title)
#     ax.set_ylabel(ylabel)
#     ax.grid()
#     ax.legend()

# axs[-1].set_xlabel("Time (s)")
# plt.tight_layout()
# plt.show()
