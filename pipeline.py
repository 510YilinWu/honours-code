import utils
import pprint

Traj_folder = "/Users/yilinwu/Desktop/honours data/Yilin-Honours/Subject/Traj/"
date = "06/16"
# sBBT_files = find_bbt_files(Traj_folder, date, file_type = "sBBT")
# print("sBBT files:", sBBT_files)

# iBBT_files = find_bbt_files(Traj_folder, date, file_type = "iBBT")
# print("iBBT files:", iBBT_files)

tBBT_files = utils.find_bbt_files(Traj_folder, date, file_type = "tBBT")
print("tBBT files:", tBBT_files)
print("Number of tBBT files:", len(tBBT_files))

# tBBT_files = [
#     "/Users/yilinwu/Desktop/honours data/Yilin-Honours/Subject/Traj/06/12/Test04.csv",  # right hand
#     "/Users/yilinwu/Desktop/honours data/Yilin-Honours/Subject/Traj/06/12/Test05.csv",  # left hand
#     "/Users/yilinwu/Desktop/honours data/Yilin-Honours/Subject/Traj/06/12/Test06.csv",  # right hand
#     "/Users/yilinwu/Desktop/honours data/Yilin-Honours/Subject/Traj/06/12/Test07.csv",  # left hand
# ]


BoxTrajfile_path = "/Users/yilinwu/Desktop/honours data/Yilin-Honours/Box/Traj/06/16/BOX.csv"  

# file_paths = [file for i, file in enumerate(tBBT_files) if i % 2 == 0]  # Select only files at odd indices
# marker_name = "RFIN"  # Adjust marker name as needed

file_paths = [file for i, file in enumerate(tBBT_files) if i % 2 == 1]  # Select only files at even indices
marker_name = "LFIN"  # Adjust marker name as needed

prominence_threshold_speed = 350  # Adjust as needed
speed_threshold = 300  # Adjust as needed
window_size = 0.05  # 50 ms
time_window_method = 1  # Choose 1, 2, or 3
save_path = "/Users/yilinwu/Desktop/honours/Thesis/figure"  # Adjust the save path as needed

results = utils.process_files(file_paths, BoxTrajfile_path, marker_name, prominence_threshold_speed, speed_threshold, window_size, time_window_method)

# ''' Example usage for printing results '''
# # Print results in a more readable way
# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(results)

# utils.plot_files(results, file_paths, marker_name, prominence_threshold_speed, speed_threshold, save_path)
