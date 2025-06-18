import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
import numpy as np

# This script reads a CSV file containing trajectory data, processes it, and extracts specific columns based on predefined indices.

# Read the file, skipping the first 5 rows. df is pandas.DataFrame, a 2D table (rows × columns)
df = pd.read_csv(
    "/Users/yilinwu/Desktop/honours data/05/13/YW_tBBT01_Trajectories.csv",
    skiprows=4,
    sep=r"\s+|,",            # Split on whitespace or commas
    engine="python"
)

# size of the DataFrame
rows, columns = df.shape

# Extract Frame from column 1 (second column)
Frame = df.iloc[:, 0]

# Calculate the time for each frame based on a 200 Hz frame capture rate
time = Frame / 200  # Time in seconds

# Define a function to extract X, Y, Z, VX, VY, VZ, AX, AY, AZ, MX, MVX, MAX for a given prefix and column indices
# The function takes a prefix (e.g., "C7") and a list of indices corresponding to the columns in the DataFrame.
# Returns a dict[str, pandas.Series]
def extract_columns(prefix, indices):
    return {
        f"{prefix}_X": df.iloc[:, indices[0]],
        f"{prefix}_Y": df.iloc[:, indices[1]],
        f"{prefix}_Z": df.iloc[:, indices[2]],
        f"{prefix}_VX": df.iloc[:, indices[3]],
        f"{prefix}_VY": df.iloc[:, indices[4]],
        f"{prefix}_VZ": df.iloc[:, indices[5]],
        f"{prefix}_AX": df.iloc[:, indices[6]],
        f"{prefix}_AY": df.iloc[:, indices[7]],
        f"{prefix}_AZ": df.iloc[:, indices[8]],
        f"{prefix}_MX": df.iloc[:, indices[9]],
        f"{prefix}_MVX": df.iloc[:, indices[10]],
        f"{prefix}_MAX": df.iloc[:, indices[11]],
    }

# Define the column indices for each prefix
column_indices = {
    "C7": [2, 3, 4, 74, 75, 76, 146, 147, 148, 218, 242, 266],
    "T10": [5, 6, 7, 77, 78, 79, 149, 150, 151, 219, 243, 267],
    "CLAV": [8, 9, 10, 80, 81, 82, 152, 153, 154, 220, 244, 268],
    "STRN": [11, 12, 13, 83, 84, 85, 155, 156, 157, 221, 245, 269],
    "LSHO": [14, 15, 16, 86, 87, 88, 158, 159, 160, 222, 246, 270],
    "LUPA": [17, 18, 19, 89, 90, 91, 161, 162, 163, 223, 247, 271],
    "LUPB": [20, 21, 22, 92, 93, 94, 164, 165, 166, 224, 248, 272],
    "LUPC": [23, 24, 25, 95, 96, 97, 167, 168, 169, 225, 249, 273],
    "LELB": [26, 27, 28, 98, 99, 100, 170, 171, 172, 226, 250, 274],
    "LMEP": [29, 30, 31, 101, 102, 103, 173, 174, 175, 227, 251, 275],
    "LWRA": [32, 33, 34, 104, 105, 106, 176, 177, 178, 228, 252, 276],
    "LWRB": [35, 36, 37, 107, 108, 109, 179, 180, 181, 229, 253, 277],
    "LFRA": [38, 39, 40, 110, 111, 112, 182, 183, 184, 230, 254, 278],
    "LFIN": [41, 42, 43, 113, 114, 115, 185, 186, 187, 231, 255, 279],
    "RSHO": [44, 45, 46, 116, 117, 118, 188, 189, 190, 232, 256, 280],
    "RUPA": [47, 48, 49, 119, 120, 121, 191, 192, 193, 233, 257, 281],
    "RUPB": [50, 51, 52, 122, 123, 124, 194, 195, 196, 234, 258, 282],
    "RUPC": [53, 54, 55, 125, 126, 127, 197, 198, 199, 235, 259, 283],
    "RELB": [56, 57, 58, 128, 129, 130, 200, 201, 202, 236, 260, 284],
    "RMEP": [59, 60, 61, 131, 132, 133, 203, 204, 205, 237, 261, 285],
    "RWRA": [62, 63, 64, 134, 135, 136, 206, 207, 208, 238, 262, 286],
    "RWRB": [65, 66, 67, 137, 138, 139, 209, 210, 211, 239, 263, 287],
    "RFRA": [68, 69, 70, 140, 141, 142, 212, 213, 214, 240, 264, 288],
    "RFIN": [71, 72, 73, 143, 144, 145, 215, 216, 217, 241, 265, 289],
}

# Extract the data for each prefix
data = {}
for prefix, indices in column_indices.items():
    data.update(extract_columns(prefix, indices))

# Example: Accessing the data for C7
# C7_X = data["C7_X"]
# C7_Y = data["C7_Y"]
# C7_Z = data["C7_Z"]
# C7_VX = data["C7_VX"]
# C7_VY = data["C7_VY"]
# C7_VZ = data["C7_VZ"]
# C7_AX = data["C7_AX"]
# C7_AY = data["C7_AY"]
# C7_AZ = data["C7_AZ"]
# C7_MX = data["C7_MX"]
# C7_MVX = data["C7_MVX"]
# C7_MAX = data["C7_MAX"]



import matplotlib.pyplot as plt

# Extract RFIN x, y, z coordinates
RFIN_X = data["RFIN_X"]
RFIN_Y = data["RFIN_Y"]
RFIN_Z = data["RFIN_Z"]
RFIN_Position = (RFIN_X**2 + RFIN_Y**2 + RFIN_Z**2)**0.5

# Calculate the speed in space for RFIN using VX, VY, VZ
RFIN_VX = data["RFIN_VX"]
RFIN_VY = data["RFIN_VY"]
RFIN_VZ = data["RFIN_VZ"]
RFIN_Speed = (RFIN_VX**2 + RFIN_VY**2 + RFIN_VZ**2)**0.5

# Calculate the acceleration in space for RFIN using AX, AY, AZ
RFIN_AX = data["RFIN_AX"]
RFIN_AY = data["RFIN_AY"]
RFIN_AZ = data["RFIN_AZ"]
RFIN_Acceleration = (RFIN_AX**2 + RFIN_AY**2 + RFIN_AZ**2)**0.5

# Calculate the change in acceleration over time (jerk) for RFIN
RFIN_Jerk = RFIN_Acceleration.diff()



# # Plot the velocity components (VX, VY, VZ) and position components (X, Y, Z) of RFIN as subplots
# fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)

# # Plot velocity components
# for ax, vel, label, color in zip(
#     axs[:3], [RFIN_VX, RFIN_VY, RFIN_VZ], ["VX", "VY", "VZ"], ["blue", "green", "red"]
# ):
#     ax.plot(time, vel, color=color)
#     ax.set_title(f"RFIN {label} Velocity")
#     ax.set_ylabel(f"{label} (mm/s)")
#     ax.grid()

# # Plot position components
# for ax, pos, label, color in zip(
#     axs[3:], [RFIN_X, RFIN_Y, RFIN_Z], ["X", "Y", "Z"], ["blue", "green", "red"]
# ):
#     ax.plot(time, pos, color=color)
#     ax.set_title(f"RFIN {label} Position")
#     ax.set_ylabel(f"{label} (mm)")
#     ax.grid()

# axs[-1].set_xlabel("Time (s)")
# plt.tight_layout()
# plt.show()






# # Plot RFIN Position, Speed, and Acceleration
# fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
# for ax, data, title, ylabel, color in zip(
#     axs, [RFIN_Position, RFIN_Speed, RFIN_Acceleration],
#     ["RFIN Position", "RFIN Speed", "RFIN Acceleration"],
#     ["Position (mm)", "Speed (mm/s)", "Acceleration (mm/s²)"],
#     ["blue", "green", "red"]
# ):
#     ax.plot(time, data, color=color)
#     ax.set_title(title)
#     ax.set_ylabel(ylabel)
#     ax.grid()
# axs[-1].set_xlabel("Time (s)")
# plt.tight_layout()
# plt.show()






prominence_threshold_position = 10  # Adjust as needed
prominence_threshold_speed = 350  # Adjust as needed

# Find local minima in RFIN Position and RFIN Speed
position_minima = RFIN_Position[find_peaks(-RFIN_Position, prominence=10)[0]]
speed_minima = RFIN_Speed[find_peaks(-RFIN_Speed, prominence=350)[0]]


# fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# for ax, data, title, ylabel, color in zip(
#     axs, [RFIN_Position, RFIN_Speed],
#     ["RFIN Position with Speed Minima", "RFIN Speed with Local Minima"],
#     ["Position (mm)", "Speed (mm/s)"], ["blue", "green"]
# ):
#     ax.plot(time, data, color=color, label=title.split()[1])
#     ax.plot(time[speed_minima.index], data[speed_minima.index], "ro", label="Speed Minima")
#     ax.set_title(title)
#     ax.set_ylabel(ylabel)
#     ax.legend()
#     ax.grid()

# axs[-1].set_xlabel("Time (s)")
# plt.tight_layout()
# plt.show()


# # Define the speed threshold
# speed_threshold = 100  # Adjust as needed

# # Find peaks in speed
# speed_peaks, _ = find_peaks(RFIN_Speed, prominence=prominence_threshold_speed)

# # Initialize a list to store the start and end times for each peak
# speed_segments = []

# # Iterate through each speed peak
# for peak in speed_peaks:
#     # Find the earliest time before the peak where speed drops below the threshold
#     start_index = peak
#     while start_index > 0 and RFIN_Speed[start_index] > speed_threshold:
#         start_index -= 1

#     # Find the latest time after the peak where speed drops below the threshold
#     end_index = peak
#     while end_index < len(RFIN_Speed) - 1 and RFIN_Speed[end_index] > speed_threshold:
#         end_index += 1

#     # Store the start and end times
#     speed_segments.append((time[start_index], time[end_index]))

# # Print the segments
# for i, (start, end) in enumerate(speed_segments):
#     print(f"Segment {i + 1}: Start = {start:.2f}s, End = {end:.2f}s, Duration = {end - start:.2f}s")

# # Create a figure with subplots for Speed and Position
# fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# # Plot RFIN Speed with start and end points of segments
# axs[0].plot(time, RFIN_Speed, label="RFIN Speed", color="blue")
# for start, end in speed_segments:
#     axs[0].plot(start, RFIN_Speed[time == start].values[0], "go", label="Start" if "Start" not in axs[0].get_legend_handles_labels()[1] else "")
#     axs[0].plot(end, RFIN_Speed[time == end].values[0], "ro", label="End" if "End" not in axs[0].get_legend_handles_labels()[1] else "")
# axs[0].set_title("RFIN Speed with Start and End Points of Segments")
# axs[0].set_ylabel("Speed (mm/s)")
# axs[0].legend()
# axs[0].grid()

# # Plot RFIN Position with start and end points of segments
# axs[1].plot(time, RFIN_Position, label="RFIN Position", color="green")
# for start, end in speed_segments:
#     axs[1].plot(start, RFIN_Position[time == start].values[0], "go", label="Start" if "Start" not in axs[1].get_legend_handles_labels()[1] else "")
#     axs[1].plot(end, RFIN_Position[time == end].values[0], "ro", label="End" if "End" not in axs[1].get_legend_handles_labels()[1] else "")
# axs[1].set_title("RFIN Position with Start and End Points of Segments")
# axs[1].set_xlabel("Time (s)")
# axs[1].set_ylabel("Position (mm)")
# axs[1].legend()
# axs[1].grid()

# # Adjust layout and show the plot
# plt.tight_layout()
# plt.show()




# Define the minimum prominence for peaks/troughs to be considered significant
prominence_threshold = 30

# Find local maxima in RFIN_X with the specified prominence
peaks, _ = find_peaks(RFIN_X, prominence=prominence_threshold)

# Find local minima in RFIN_X by inverting the values with the specified prominence
troughs, _ = find_peaks(-RFIN_X, prominence=prominence_threshold)



# # Plot RFIN_X, RFIN_Y, and RFIN_Z with peaks and troughs
# fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
# coords = [RFIN_X, RFIN_Y, RFIN_Z]
# labels = ["RFIN_X", "RFIN_Y", "RFIN_Z"]
# colors = ["blue", "green", "red"]

# for ax, coord, label, color in zip(axs, coords, labels, colors):
#     ax.plot(time, coord, label=label, color=color)
#     ax.plot(time[peaks], coord[peaks], "ro", label="Peaks")
#     ax.plot(time[troughs], coord[troughs], "go", label="Troughs")
#     ax.set_title(f"{label} with Peaks and Troughs")
#     ax.set_ylabel(f"{label} (mm)")
#     ax.legend()
#     ax.grid()

# axs[-1].set_xlabel("Time (s)")
# plt.tight_layout()
# plt.show()






# # Create subplots for RFIN coordinates, speed, acceleration, and jerk
# fig, axs = plt.subplots(6, 1, figsize=(12, 18), sharex=True)
# titles = ["RFIN X", "RFIN Y", "RFIN Z", "RFIN Speed", "RFIN Acceleration", "RFIN Jerk"]
# data = [RFIN_X, RFIN_Y, RFIN_Z, RFIN_Speed, RFIN_Acceleration, RFIN_Jerk]
# colors = ["blue", "green", "red", "purple", "orange", "cyan"]
# ylabels = ["mm", "mm", "mm", "mm/s", "mm/s²", "mm/s³"]

# for ax, title, d, color, ylabel in zip(axs, titles, data, colors, ylabels):
#     ax.plot(time, d, color=color)
#     ax.set_title(title)
#     ax.set_ylabel(ylabel)
#     ax.grid()

# axs[-1].set_xlabel("Time (s)")
# plt.tight_layout()
# plt.show()


# # Create a 3D plot for the trajectory of the marker (first 1000 frames)
# fig2 = plt.figure(figsize=(10, 8))
# ax = fig2.add_subplot(111, projection='3d')

# # Plot the trajectory of RFIN in 3D space for the first 1000 frames
# ax.plot(RFIN_X[:1000], RFIN_Y[:1000], RFIN_Z[:1000], label="RFIN Trajectory (First 1000 Frames)", color="blue")

# # Set labels and title
# ax.set_title("3D Trajectory of RFIN Marker (First 1000 Frames)")
# ax.set_xlabel("X (mm)")
# ax.set_ylabel("Y (mm)")
# ax.set_zlabel("Z (mm)")

# # Add a legend
# ax.legend()

# # Show the plot
# plt.show()

# # Plot the magnitude of MX, MVX, and MAX for RFIN
# fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
# titles = ["RFIN MX Magnitude", "RFIN MVX Magnitude Velocity", "RFIN MAX Magnitude Acceleration"]
# ylabels = ["MX (mm)", "MVX (mm/s)", "MAX (mm/s²)"]
# colors = ["blue", "green", "red"]
# keys = ["RFIN_MX", "RFIN_MVX", "RFIN_MAX"]

# for ax, title, ylabel, color, key in zip(axs, titles, ylabels, colors, keys):
#     ax.plot(time, data[key], label=title, color=color)
#     ax.set_title(title)
#     ax.set_ylabel(ylabel)
#     ax.grid()

# axs[-1].set_xlabel("Time (s)")
# plt.tight_layout()
# plt.show()

# Thresholds for movement start and end
z_range, speed_range, accel_range,RFIN_X_threshold = (830, 860), (0, 150), (0, 3500),200

# Detect tStart and tEnd
tStart_mask = (RFIN_X < RFIN_X_threshold) & RFIN_Z.between(*z_range) & RFIN_Speed.between(*speed_range) & RFIN_Acceleration.between(*accel_range)
tEnd_mask = (RFIN_X > RFIN_X_threshold) & RFIN_Z.between(*z_range) & RFIN_Speed.between(*speed_range) & RFIN_Acceleration.between(*accel_range)

tStart_indices, tEnd_indices = time[tStart_mask].to_numpy(), time[tEnd_mask].to_numpy()


if not len(tStart_indices) or not len(tEnd_indices):
    print("Error: tStart_indices or tEnd_indices is empty.")
    exit()




# Calculate differences for tStart_indices and tEnd_indices
tStart_diffs = np.diff(tStart_indices)
tEnd_diffs = np.diff(tEnd_indices)

# Find the top 16 indices and sort them in ascending order
tStart_indices_cluster = np.sort(np.argsort(tStart_diffs)[-15:])
tEnd_indices_cluster = np.sort(np.argsort(tEnd_diffs)[-15:])

# Define clusters for tStart_indices based on tStart_indices_cluster
tStart_indices_clusters = []
start = 0
for end in tStart_indices_cluster:
    tStart_indices_clusters.append(tStart_indices[start:end + 1])
    start = end + 1
tStart_indices_clusters.append(tStart_indices[tStart_indices_cluster[-1]:])

# Define clusters for tEnd_indices based on tEnd_indices_cluster
tEnd_indices_clusters = []
start = 0
for end in tEnd_indices_cluster:
    tEnd_indices_clusters.append(tEnd_indices[start:end + 1])
    start = end + 1
tEnd_indices_clusters.append(tEnd_indices[tEnd_indices_cluster[-1]:])

# Find the representative value for each cluster in tStart_indices_clusters and tEnd_indices_clusters
tStart_representatives = [np.median(cluster) if len(cluster) % 2 != 0 else sorted(cluster)[len(cluster) // 2 - 1] for cluster in tStart_indices_clusters]
tEnd_representatives = [np.median(cluster) if len(cluster) % 2 != 0 else sorted(cluster)[len(cluster) // 2 - 1] for cluster in tEnd_indices_clusters]


print("tStart_representatives:", tStart_representatives)
print("tEnd_representatives:", tEnd_representatives)


# Plot RFIN_X, RFIN_Z, RFIN_Speed, and RFIN_Acceleration with tStart_indices, tEnd_indices, tStart_representatives, and tEnd_representatives
fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
titles = ["RFIN X", "RFIN Z", "RFIN Speed", "RFIN Acceleration"]
data = [RFIN_X, RFIN_Z, RFIN_Speed, RFIN_Acceleration]
colors = ["blue", "green", "purple", "orange"]
ylabels = ["X (mm)", "Z (mm)", "Speed (mm/s)", "Acceleration (mm/s²)"]

for ax, title, d, color, ylabel in zip(axs, titles, data, colors, ylabels):
    ax.plot(time, d, color=color, label=title)
    ax.plot(tStart_indices, d[tStart_mask], "go", label="tStart")
    ax.plot(tEnd_indices, d[tEnd_mask], "ro", label="tEnd")
    ax.plot(tStart_representatives, d[np.isin(time, tStart_representatives)], "bs", label="tStart Representative")
    ax.plot(tEnd_representatives, d[np.isin(time, tEnd_representatives)], "ms", label="tEnd Representative")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid()

axs[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()

"""
tStart_representatives/tEnd_representatives: list[float]
    A list of median of each cluster of tStart_indices/tEnd_indices for the start/end of movement clusters. 
    where clusters are formed based on significant differences in consecutive time indices. top 16 indices are used to form clusters.
    tStart_indices/tEnd_indices are caculated based on the threshold of RFIN_X, RFIN_Z, RFIN_Speed, and RFIN_Acceleration.
"""







# # Find clusters where differences exceed the threshold
# tStart_clusters = np.split(tStart_indices, np.where(tStart_diffs > threshold)[0] + 1)
# tEnd_clusters = np.split(tEnd_indices, np.where(tEnd_diffs > threshold)[0] + 1)

# # Filter out empty clusters
# tStart_clusters = [cluster for cluster in tStart_clusters if len(cluster) > 0]
# tEnd_clusters = [cluster for cluster in tEnd_clusters if len(cluster) > 0]

# # Print the clusters
# print("tStart_clusters:")
# for i, cluster in enumerate(tStart_clusters):
#     print(f"Cluster {i + 1}: {cluster}")

# print("\ntEnd_clusters:")
# for i, cluster in enumerate(tEnd_clusters):
#     print(f"Cluster {i + 1}: {cluster}")







# print(f"tStart_indices: {tStart_indices}")
# print(f"tEnd_indices: {tEnd_indices}")

# # Plot RFIN_X, RFIN_Z, RFIN_Speed, and RFIN_Acceleration with tStart_indices and tEnd_indices as green and red dots
# fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
# titles = ["RFIN X", "RFIN Z", "RFIN Speed", "RFIN Acceleration"]
# data = [RFIN_X, RFIN_Z, RFIN_Speed, RFIN_Acceleration]
# colors = ["blue", "green", "purple", "orange"]
# ylabels = ["X (mm)", "Z (mm)", "Speed (mm/s)", "Acceleration (mm/s²)"]

# for ax, title, d, color, ylabel in zip(axs, titles, data, colors, ylabels):
#     ax.plot(time, d, color=color, label=title)
#     ax.plot(time[tStart_indices], d[tStart_indices], "go", label="tStart")
#     ax.plot(time[tEnd_indices], d[tEnd_indices], "ro", label="tEnd")
#     ax.set_title(title)
#     ax.set_ylabel(ylabel)
#     ax.legend()
#     ax.grid()

# axs[-1].set_xlabel("Time (s)")
# plt.tight_layout()
# plt.show()


