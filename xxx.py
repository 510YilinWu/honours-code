
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

Box_Traj_file='/Volumes/MNHS-MoCap/Yilin-Honours/Box/Traj/2025/06/26/2/BOX_Cali.csv'

# Read CSV
df = pd.read_csv(Box_Traj_file, skiprows=4, sep=r"\s+|,", engine="python")

# Column indices
column_indices = {
    "BOX:RTC": [2, 3, 4],
    "BOX:LTC": [5, 6, 7],
    "BOX:LBC": [8, 9, 10],
    "BOX:PT": [11, 12, 13],
    "BOX:PB": [14, 15, 16],
}

# Extract data without filtering
Box_data = {}
for prefix, (ix, iy, iz) in column_indices.items():
    x = df.iloc[:, ix]
    y = df.iloc[:, iy]
    z = df.iloc[:, iz]
    Box_data[prefix] = (x, y, z)


import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each set of points
for prefix, (x, y, z) in Box_data.items():
    ax.scatter(x, y, z, label=prefix)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()