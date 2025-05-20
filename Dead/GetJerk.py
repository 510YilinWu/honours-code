import LoadExtractedData
import numpy as np

extracted_data_path = "/Users/yilinwu/Desktop/honours data/Extracted data/Yilin-YW Trial 6.py"
extractedData=LoadExtractedData.main(extracted_data_path)# Run the main function from LoadExtractedData


def calculate_ldj(acc_x, acc_y, acc_z, frame_rate):
    # Compute Jerk (numerical derivative of acceleration)
    # Jerk: First, the function calculates how fast the acceleration is changing in the x, y, and z directions. 
    # This is done by finding the difference in acceleration between each point in time and dividing it by the time difference (frame rate). 
    # This gives us jerk values.
    # same length as acceleration

    jerk_x = np.gradient(acc_x, 1/frame_rate)
    jerk_y = np.gradient(acc_y, 1/frame_rate)
    jerk_z = np.gradient(acc_z, 1/frame_rate)
    
    # Compute squared magnitude of jerk
    # Squared Jerk Magnitude: Next, we calculate the "magnitude" of the jerk at each point in time. 
    # This is like finding the overall strength of the jerk by combining the jerks in the x, y, and z directions and squaring them.
    # same length as acceleration
    jerk_magnitude_squared = jerk_x**2 + jerk_y**2 + jerk_z**2
    
    # Compute integral of squared jerk magnitude
    # Total Jerk over Time: We then add up all these squared jerk values across the entire time period to get the total jerk over time. 
    # This is done using a method called the trapezoidal rule, which estimates the area under the curve of jerk magnitude.
    jerk_integral = np.trapezoid(jerk_magnitude_squared, dx=1/frame_rate)
    
    # Compute total duration
    T = len(acc_x) / frame_rate
    
    # Compute average speed (approximation using magnitude of velocity)
    avg_velocity = np.mean(np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2))
    print("avg_velocity:", avg_velocity)

    # Compute LDJ
    LDJ = np.log10(jerk_integral / (T * avg_velocity**2))
    
    return LDJ

# Extract acceleration data and ensure they are numpy arrays of floats
acc_x = np.array([float(i) for i in extractedData.get('Trajectories', {}).get('variables', {}).get('LFIN', {}).get('Accelerations', {}).get("X''", []) if i], dtype=float)
acc_y = np.array([float(i) for i in extractedData.get('Trajectories', {}).get('variables', {}).get('LFIN', {}).get('Accelerations', {}).get("Y''", []) if i], dtype=float)
acc_z = np.array([float(i) for i in extractedData.get('Trajectories', {}).get('variables', {}).get('LFIN', {}).get('Accelerations', {}).get("Z''", []) if i], dtype=float)

# Set frame rate
frame_rate = 100  # Hz

# Compute LDJ
ldj_value = calculate_ldj(acc_x, acc_y, acc_z, frame_rate)
print("LDJ a:", ldj_value)


import matplotlib.pyplot as plt

# Calculate total duration
T = len(acc_x) / frame_rate  # Total duration in seconds
time = np.linspace(0, T, len(acc_x))  # Time array from 0 to total duration

# Create a plot for position data
plt.figure(figsize=(10, 5))
plt.subplot(3, 1, 1)
plt.plot(time, acc_x, label="X", color="blue")
plt.title("Acceleration X Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (units/s^2)")

plt.subplot(3, 1, 2)
plt.plot(time, acc_y, label="Y", color="green")
plt.title("Acceleration Y Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (units/s^2)")

plt.subplot(3, 1, 3)
plt.plot(time, acc_z, label="Z", color="red")
plt.title("Acceleration Z Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (units/s^2)")

plt.tight_layout()
plt.title("Acceleration Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("acc (units)")
plt.legend()
plt.show()



# Extract position data and ensure they are numpy arrays of floats
pos_x = extractedData['Trajectories']['variables']['LFIN']['position']['X']
pos_y = extractedData['Trajectories']['variables']['LFIN']['position']['Y']
pos_z = extractedData['Trajectories']['variables']['LFIN']['position']['Z']

# Calculate total duration
T = len(pos_x) / frame_rate  # Total duration in seconds
time = np.linspace(0, T, len(pos_x))  # Time array from 0 to total duration

import matplotlib.pyplot as plt

# Create a plot for position data
plt.figure(figsize=(10, 5))
plt.plot(time, pos_x, label="X", color="blue")
plt.title("Position X Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Position (units)")
plt.legend()
plt.show()
