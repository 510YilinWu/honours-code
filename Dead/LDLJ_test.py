
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import math
import pandas as pd

# Time and waypoints
t = np.linspace(0, 10, 1000)
waypoint_times = np.linspace(0, 10, 10)
smooth_positions = np.sin(waypoint_times)
unsmooth_positions = smooth_positions + 3 * np.sin(20 * waypoint_times) + 2 * np.sin(100 * waypoint_times)

# Cubic splines
smooth_spline = CubicSpline(waypoint_times, smooth_positions, bc_type="natural")
unsmooth_spline = CubicSpline(waypoint_times, unsmooth_positions, bc_type="natural")

# Evaluate position, speed, acceleration, jerk
smooth_data = [smooth_spline(t, i) for i in range(4)]
unsmooth_data = [unsmooth_spline(t, i) for i in range(4)]

# Plot position, speed, acceleration, jerk
labels = ["Position", "Speed", "Acceleration", "Jerk"]
colors = ["blue", "orange", "green", "red"]
plt.figure(figsize=(12, 8))
for i, label in enumerate(labels):
    plt.subplot(4, 1, i + 1)
    plt.plot(t, smooth_data[i], label=f"Smooth {label}", color=colors[i])
    plt.plot(t, unsmooth_data[i], label=f"Unsmooth {label}", color=colors[i], linestyle="--")
    plt.ylabel(label)
    plt.legend()
    plt.grid(True)
plt.xlabel("Time")
plt.tight_layout()
plt.show()

# LDLJ calculation
segments = [(int(i * 100), int((i + 1) * 100)) for i in range(10)]
smooth_LDLJ, unsmooth_LDLJ = [], []
for start, end in segments:
    for data, LDLJ in zip([smooth_data, unsmooth_data], [smooth_LDLJ, unsmooth_LDLJ]):
        jerk_segment, duration = data[3][start:end], (end - start) / 200
        t_segment = np.linspace(0, duration, len(jerk_segment))
        jerk_integral = np.trapezoid(jerk_segment**2, t_segment)
        vpeak = data[1][start:end].max()
        LDLJ.append(-math.log(abs((duration**3 / vpeak**2) * jerk_integral), math.e))
        print(f"Segment {start // 100 + 1}: LDLJ = {LDLJ[-1]}", 
              f"Duration = {duration:.2f}, Vpeak = {vpeak:.2f}, Jerk Integral = {jerk_integral:.2f}")

# Plot LDLJ comparison
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), smooth_LDLJ, label="Smooth LDLJ", marker="o")
plt.plot(range(1, 11), unsmooth_LDLJ, label="Unsmooth LDLJ", marker="o", linestyle="--")
plt.xlabel("Segment")
plt.ylabel("LDLJ Value")
plt.title("LDLJ Comparison Across Segments")
plt.legend()
plt.grid(True)
plt.show()
# Create a DataFrame for LDLJ values
ldlj_comparison_df = pd.DataFrame({
    "Segment": range(1, 11),
    "Smooth LDLJ": smooth_LDLJ,
    "Unsmooth LDLJ": unsmooth_LDLJ
})

# Print the table
print(ldlj_comparison_df)