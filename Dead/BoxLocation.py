import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np

# File path
file_path = '/Users/yilinwu/Desktop/honours data/Yilin-Honours/Box/Traj/06/12/Test04.csv'

# Read CSV
df = pd.read_csv(file_path, skiprows=4, sep=r"\s+|,", engine="python")

# Column indices
column_indices = {
    "BOX:RTC": [2, 3, 4],
    "BOX:LTC": [5, 6, 7],
    "BOX:LBC": [8, 9, 10],
    "BOX:PT": [11, 12, 13],
    "BOX:PB": [14, 15, 16],
}

# Extract data without filtering
filtered_data = {}

for prefix, (ix, iy, iz) in column_indices.items():
    x = df.iloc[:, ix]
    y = df.iloc[:, iy]
    z = df.iloc[:, iz]
    filtered_data[prefix] = (x, y, z)



# Find the maximum values of the 4 markers on each axis
max_values = {}

for prefix in column_indices.keys():
    x, y, z = filtered_data[prefix]
    max_x = x.max()
    max_y = y.max()
    max_z = z.max()
    max_values[prefix] = (max_x, max_y, max_z)

# Print the maximum values for each marker
for prefix, (max_x, max_y, max_z) in max_values.items():
    print(f"Max values for {prefix}: X={max_x}, Y={max_y}, Z={max_z}")

# Calculate rfin_x_range and lfin_x_range based on the given logic
rfin_x_range_max = filtered_data["Box:LTC"][0].max()
rfin_x_range_min = filtered_data["Box:PT"][0].min()

lfin_x_range_max = filtered_data["Box:PT"][0].max()
lfin_x_range_min = filtered_data["Box:RTC"][0].min()






# def calculate_rangesByBoxTraj(BoxTrajfile_path):
#     # Read CSV
#     df = pd.read_csv(BoxTrajfile_path, skiprows=4, sep=r"\s+|,", engine="python")

#     # Column indices
#     column_indices = {
#         "Box:LCorner": [2, 3, 4],
#         "Box:LEdge": [5, 6, 7],
#         "Box:Partition": [8, 9, 10],
#         "Box:RCorner": [11, 12, 13],
#     }

#     # Extract data without filtering
#     filtered_data = {}
#     for prefix, (ix, iy, iz) in column_indices.items():
#         x = df.iloc[:, ix]
#         y = df.iloc[:, iy]
#         z = df.iloc[:, iz]
#         filtered_data[prefix] = (x, y, z)

#     # Calculate rfin_x_range and lfin_x_range based on the given logic
#     # for end position of the reach
#     rfin_x_range_max = filtered_data["Box:LCorner"][0].max()
#     rfin_x_range_min = filtered_data["Box:Partition"][0].min()

#     lfin_x_range_max = filtered_data["Box:Partition"][0].max()
#     lfin_x_range_min = filtered_data["Box:RCorner"][0].min()

#     rfin_x_range = (rfin_x_range_min, rfin_x_range_max)
#     lfin_x_range = (lfin_x_range_min, lfin_x_range_max)

#     return lfin_x_range, rfin_x_range

# # Example usage
# BoxTrajfile_path = '/Users/yilinwu/Desktop/honours data/Yilin-Honours/Box/Traj/06/12/Test04.csv'
# lfin_x_range, rfin_x_range = calculate_rangesByBoxTraj(BoxTrajfile_path)
# print("LFin X Range:", lfin_x_range)
# print("RFin X Range:", rfin_x_range)









# fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# view_pairs = [('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]
# colors = cm.tab10(np.linspace(0, 1, len(column_indices)))
# radius = 6.5 / 2
# theta = np.linspace(0, 2 * np.pi, 100)

# for ax, (dim1, dim2) in zip(axes, view_pairs):
#     for (prefix, color) in zip(column_indices.keys(), colors):
#         coords = dict(zip(['X', 'Y', 'Z'], filtered_data[prefix]))
#         x1, x2 = coords[dim1], coords[dim2]
#         ax.scatter(x1, x2, label=prefix, s=10, color=color)
#         center1, center2 = x1.mean(), x2.mean()
#         circle1 = center1 + radius * np.cos(theta)
#         circle2 = center2 + radius * np.sin(theta)
#         ax.plot(circle1, circle2, color=color, linestyle='--', alpha=0.7)
#         # Display mean values on the plot, except for subplot 3
#         if dim1 != 'Y' or dim2 != 'Z':
#             ax.text(center1, center2, f"Mean\n({center1:.2f}, {center2:.2f})", 
#             color='black', fontsize=8, ha='center', va='center')

#     ax.set(xlabel=dim1, ylabel=dim2, title=f"View: {dim1} vs {dim2}")
#     ax.legend(loc='upper right', fontsize='small')

# plt.tight_layout()
# plt.savefig('/Users/yilinwu/Desktop/honours code/BoxLocation_Centered_2D_with_Circles_and_Means.png', dpi=300)
# plt.show()

# # 3D plot with circles indicating markers and equal aspect ratio
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# colors = cm.tab10(np.linspace(0, 1, len(column_indices)))

# for prefix, color in zip(column_indices.keys(), colors):
#     x, y, z = filtered_data[prefix]
#     ax.scatter(x, y, z, label=prefix, s=10, color=color)

#     # Calculate marker center and plot a circle
#     center_x = x.mean()
#     center_y = y.mean()
#     center_z = z.mean()
#     radius = 6.5 / 2  # marker radius in mm

#     # Plot circle in XY plane
#     theta = np.linspace(0, 2 * np.pi, 100)
#     circle_x = center_x + radius * np.cos(theta)
#     circle_y = center_y + radius * np.sin(theta)
#     circle_z = np.full_like(circle_x, center_z)
#     ax.plot(circle_x, circle_y, circle_z, color=color, linestyle='--', alpha=0.7)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Marker Locations with Circles')
# ax.legend(loc='upper right', fontsize='small')

# plt.tight_layout()
# plt.savefig('/Users/yilinwu/Desktop/honours code/BoxLocation_Centered_3D.png', dpi=300)
# plt.show()