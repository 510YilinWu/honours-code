import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import math
from numpy.fft import rfft, rfftfreq

# --- Synthetic Trajectories ---
def straight_slow(t):
    return np.vstack((t, np.zeros_like(t))).T

def straight_fast(t):
    return np.vstack((t**2, np.zeros_like(t))).T  # faster acceleration

def curved_slow(t):
    return np.vstack((t, np.sin(2*np.pi*t))).T

def curved_fast(t):
    return np.vstack((t**2, np.sin(6*np.pi*t))).T

# --- LDLJ computation ---
def compute_ldlj(traj, dt):
    vel = np.gradient(traj, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    jerk_mag = np.linalg.norm(jerk, axis=1)
    duration = len(traj) * dt
    amp = np.linalg.norm(traj[-1] - traj[0])
    dimless_jerk = np.trapz(jerk_mag**2, dx=dt) * (duration**5) / (amp**2)
    return -np.log(dimless_jerk)

# --- SPARC computation (simplified spectral arc length) ---
def sparc(movement, fs, padlevel=4, fc=20.0, amp_th=0.05):
    """
    Calcualtes the smoothness of the given speed profile using the modified
    spectral arc length metric.

    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    padlevel : integer, optional
               Indicates the amount of zero padding to be done to the movement
               data for estimating the spectral arc length. [default = 4]
    fc       : float, optional
               The max. cut off frequency for calculating the spectral arc
               length metric. [default = 10.]
    amp_th   : float, optional
               The amplitude threshold to used for determing the cut off
               frequency upto which the spectral arc length is to be estimated.
               [default = 0.05]

    Returns
    -------
    sal      : float
               The spectral arc length estimate of the given movement's
               smoothness.
    (f, Mf)  : tuple of two np.arrays
               This is the frequency (f) and the magnitude spectrum (Mf) of the
               given movement data. The spectrum spans from 0 to fs/2.
    (f_sel, Mf_sel) : tuple of two np.arrays
                      This is the portion of the spectrum that is selected for
                      calculating the spectral arc length.

    Notes
    -----
    This is the modified spectral arc length metric, which has been tested only
    for discrete movements.

    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5 * t**2)
    >>> sal, _, _ = sparc(move, fs=100.)
    >>> '%.5f' % sal
    '-1.41403'
    """
    # Number of zeros to be padded.
    nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

    # Frequency
    f = np.arange(0, fs, fs / nfft)
    # Normalized magnitude spectrum
    Mf = np.abs(np.fft.fft(movement, nfft))
    Mf = Mf / max(Mf)

    # Indices to choose only the spectrum within the given cutoff frequency (fc).
    fc_inx = ((f <= fc) * 1).nonzero()
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]

    # Choose the amplitude threshold based cutoff frequency.
    inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
    fc_inx = range(inx[0], inx[-1] + 1)
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]

    # Calculate arc length
    new_sal = -np.sum(np.sqrt((np.diff(f_sel) / (f_sel[-1] - f_sel[0]))**2 + np.diff(Mf_sel)**2))
    return new_sal, (f, Mf), (f_sel, Mf_sel)

# --- Generate & Compare ---
dt = 0.01
t = np.linspace(0, 1, 100)

cases = {
    "Straight Slow": straight_slow(t),
    "Straight Fast": straight_fast(t),
    "Curved Slow": curved_slow(t),
    "Curved Fast": curved_fast(t)
}

results = {}
for name, traj in cases.items():
    ldlj = compute_ldlj(traj, dt)
    vel = np.gradient(traj, dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    sparc_val, _, _ = sparc(speed, 1/dt)
    results[name] = (ldlj, sparc_val)

# --- Visualize ---
# Create a GridSpec layout with 3 rows:
# Left column: three rows (top: trajectories, mid: velocities, bottom: accelerations)
# Right column: scatter metrics spanning all three rows.
fig = plt.figure(figsize=(12,10))
gs_master = fig.add_gridspec(nrows=3, ncols=2, width_ratios=[2, 1])
ax_traj = fig.add_subplot(gs_master[0, 0])
ax_vel  = fig.add_subplot(gs_master[1, 0])
ax_acc  = fig.add_subplot(gs_master[2, 0])
ax_scatter = fig.add_subplot(gs_master[:, 1])

# Plot trajectories in the top-left subplot
for name, traj in cases.items():
    ax_traj.plot(traj[:,0], traj[:,1], label=name)
ax_traj.set_title("Synthetic Trajectories")
ax_traj.legend()

# Plot velocities in the middle-left subplot
for name, traj in cases.items():
    vel = np.gradient(traj, dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    ax_vel.plot(t, speed, label=name)
ax_vel.set_xlabel("Time (s)")
ax_vel.set_ylabel("Speed")
ax_vel.set_title("Velocity Profiles")
ax_vel.legend()

# Plot accelerations in the bottom-left subplot
for name, traj in cases.items():
    vel = np.gradient(traj, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    acc_mag = np.linalg.norm(acc, axis=1)
    ax_acc.plot(t, acc_mag, label=name)
ax_acc.set_xlabel("Time (s)")
ax_acc.set_ylabel("Acceleration")
ax_acc.set_title("Acceleration Profiles")
ax_acc.legend()

# Scatter plot of metrics in the right subplot
for name, (ldlj, sparc_val) in results.items():
    ax_scatter.scatter(ldlj, sparc_val, label=name, s=100)
ax_scatter.set_xlabel("LDLJ (higher = smoother)")
ax_scatter.set_ylabel("SPARC (higher = smoother)")
ax_scatter.set_title("Smoothness Metrics Comparison")
ax_scatter.legend()
ax_scatter.set_xlim(-20, 55)
ax_scatter.set_ylim(-4, -1)

plt.tight_layout()
plt.show()

# Print numeric results
print("Results (LDLJ, SPARC):")
for name, vals in results.items():
    print(f"{name}: LDLJ={vals[0]:.3f}, SPARC={vals[1]:.3f}")

# --- SPARC WITH PLOTS ---
def sparc_with_plots(movement, fs=200, padlevel=4, fc=20.0, amp_th=0.05):
    # 1. Time domain plot (original movement)
    t = np.arange(len(movement)) / fs
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 2, 1)
    plt.plot(t, movement)
    plt.title("1. Speed Profile (Time Domain)")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed")

    # 2. FFT and Normalization
    nfft = int(2 ** (np.ceil(np.log2(len(movement))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    Mf = np.abs(np.fft.fft(movement, nfft))
    Mf = Mf / np.max(Mf)

    plt.subplot(3, 2, 2)
    plt.plot(f[:nfft // 2], Mf[:nfft // 2])
    plt.title("2. Full Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Magnitude")

    # 3. Cutoff Frequency Filtering
    fc_idx = np.where(f <= fc)[0]
    f_sel = f[fc_idx]
    Mf_sel = Mf[fc_idx]

    plt.subplot(3, 2, 3)
    plt.plot(f_sel, Mf_sel)
    plt.title("3. Spectrum Below Cutoff (fc = 10 Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    # 4. Amplitude Thresholding with Continuous Range
    inx = np.where(Mf_sel >= amp_th)[0]
    fc_inx = np.arange(inx[0], inx[-1] + 1)
    f_cut = f_sel[fc_inx]
    Mf_cut = Mf_sel[fc_inx]

    plt.subplot(3, 2, 4)
    plt.plot(f_sel, Mf_sel, color='lightgray', label='All under 20Hz')
    plt.plot(f_cut, Mf_cut, color='blue', label='Above threshold')
    plt.axhline(y=amp_th, color='red', linestyle='--', label='Threshold')
    plt.title("4. After Amplitude Thresholding")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()

    # 5. Spectral Arc Length Calculation (matching sparc())
    df = np.diff(f_cut) / (f_cut[-1] - f_cut[0])
    dM = np.diff(Mf_cut)
    arc_length = -np.sum(np.sqrt(df ** 2 + dM ** 2))

    plt.subplot(3, 2, 5)
    plt.plot(f_cut, Mf_cut, marker='o')
    for i in range(len(df)):
        plt.plot([f_cut[i], f_cut[i+1]], [Mf_cut[i], Mf_cut[i+1]], 'k--')
    plt.title("5. Arc Segments Used in SPARC")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    # 6. Display SPARC Value
    plt.subplot(3, 2, 6)
    plt.text(0.1, 0.5, f"SPARC Value:\n{arc_length:.4f}", fontsize=20)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return arc_length

# Plot SPARC for curved_slow trajectory:
dt = t[1] - t[0]
traj_curved_slow = curved_slow(t)
vel_curved_slow = np.gradient(traj_curved_slow, dt, axis=0)
speed_curved_slow = np.linalg.norm(vel_curved_slow, axis=1)

print("SPARC with plots for curved_slow trajectory")
sparc_val_curved_slow = sparc_with_plots(speed_curved_slow, fs=1/dt)

# Plot SPARC for curved_fast trajectory:
traj_curved_fast = curved_fast(t)
vel_curved_fast = np.gradient(traj_curved_fast, dt, axis=0)
speed_curved_fast = np.linalg.norm(vel_curved_fast, axis=1)

print("SPARC with plots for curved_fast trajectory")
sparc_val_curved_fast = sparc_with_plots(speed_curved_fast, fs=1/dt)

# Plot SPARC for straight_slow trajectory:
dt = t[1] - t[0]
traj_straight_slow = straight_slow(t)
vel_straight_slow = np.gradient(traj_straight_slow, dt, axis=0)
speed_straight_slow = np.linalg.norm(vel_straight_slow, axis=1)

print("SPARC with plots for straight_slow trajectory")
sparc_val_straight_slow = sparc_with_plots(speed_straight_slow, fs=1/dt)

# Plot SPARC for straight_fast trajectory:
traj_straight_fast = straight_fast(t)
vel_straight_fast = np.gradient(traj_straight_fast, dt, axis=0)
speed_straight_fast = np.linalg.norm(vel_straight_fast, axis=1)

print("SPARC with plots for straight_fast trajectory")
sparc_val_straight_fast = sparc_with_plots(speed_straight_fast, fs=1/dt)



## -----------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf

# -----------------------------
# 1. Create synthetic dataset
# -----------------------------
np.random.seed(42)

n_subjects = 10
n_locations = 16  # 4x4 grid
trials_per_loc = 5

rows = []
for subj in range(n_subjects):
    for loc in range(n_locations):
        amplitude = np.random.uniform(10, 30)  # cm
        for t in range(trials_per_loc):
            duration = np.random.uniform(0.4, 1.2)  # seconds
            # LDLJ depends on duration (shorter = higher LDLJ), amplitude, and some noise
            ldlj = 3 - 1.5 * duration + 0.02 * amplitude + np.random.normal(0, 0.2)
            rows.append([subj, loc, amplitude, duration, ldlj])


data = pd.DataFrame(rows, columns=["Subject", "Location", "Amplitude", "Duration", "LDLJ"])

# -----------------------------
# 2. Simple correlation
# -----------------------------
r, pval = pearsonr(data["Duration"], data["LDLJ"])
print("Correlation LDLJ vs Duration: r=%.3f, p=%.4f" % (r, pval))

# -----------------------------
# 3. ANOVA (treating Location as factor)
# -----------------------------
model_aov = ols("LDLJ ~ Duration + Amplitude + C(Location)", data=data).fit()
aov_table = sm.stats.anova_lm(model_aov, typ=2)
print("\nANOVA results:\n", aov_table)

# -----------------------------
# 4. Mixed-effects model (Subject as random factor)
# -----------------------------
model_mixed = smf.mixedlm("LDLJ ~ Duration + Amplitude", data, groups=data["Subject"])
result_mixed = model_mixed.fit()
print("\nMixed-effects results:\n", result_mixed.summary())


## -----------------------------------------------------------


import numpy as np
import math

# Fabricated example data generator
def make_fake_traj(jerky=False, n_points=500, fs=200):
    t = np.linspace(0, 2, n_points)  # 2 seconds
    if not jerky:
        # Smooth: sinusoidal path with even lower noise for increased smoothness
        position = np.sin(2 * np.pi * t / t[-1])
        position += 0.001 * np.random.randn(n_points)  # Reduced noise
    else:
        # Jerky: add higher frequency noise + abrupt jumps
        position = np.sin(2 * np.pi * t / t[-1])
        position += 0.3 * np.sin(20 * np.pi * t / t[-1])
        position += 0.05 * np.random.randn(n_points)

    # Differentiate to get velocity, acceleration, jerk
    velocity = np.gradient(position, 1/fs)
    acceleration = np.gradient(velocity, 1/fs)
    jerk = np.gradient(acceleration, 1/fs)
    
    return position, velocity, acceleration, jerk

# Build a fake `results` dict in the expected structure
results = {
    "07/22/HW": {
        "left": {
            1: {
                "/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv": {
                    'traj_space': {
                        'LFIN': make_fake_traj(jerky=False)
                    }
                }
            },
            0: {  # index 0 holds the start/end indices
                "/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv": [None,None,None,None,None,None,0,500]
            }
        },
        "right": {
            1: {
                "/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT01.csv": {
                    'traj_space': {
                        'RFIN': make_fake_traj(jerky=True)
                    }
                }
            },
            0: {
                "/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT01.csv": [None,None,None,None,None,None,0,500]
            }
        }
    }
}


def plot_normalized_jerk(results, subject="06/19/CZ", hand="right", trial=1,
                         file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/06/19/CZ/CZ_tBBT01.csv',
                         fs=200, target_samples=101):
    """
    Extracts trajectory segments (position, velocity, acceleration, jerk) for a given trial,
    normalizes the jerk signal (within a selected range) to a standardized time base via linear interpolation,
    and plots Position, Velocity, Acceleration, Original Jerk, and Normalized Jerk as subplots.
    
    Parameters:
        results (dict): The results dictionary containing trajectory data.
        subject (str): Subject identifier.
        hand (str): Hand identifier.
        trial (int): Trial index.
        file_path (str): Path to the trial CSV file.
        fs (int): Sampling rate (Hz). Default is 200.
        target_samples (int): Number of samples for the normalized jerk signal.
    """
    if hand == 'right':
        marker = 'RFIN' 
    else:
        marker = 'LFIN'    # Extract trajectory segments
    traj_data = results[subject][hand][trial][file_path]['traj_space'][marker]
    position = traj_data[0]
    velocity = traj_data[1]
    acceleration = traj_data[2]
    jerk = traj_data[3]
    
    # Retrieve the selected range from the full data
    plotRange = results[subject][hand][0][file_path]
    start_idx = plotRange[6]
    end_idx = plotRange[7]
    # end_idx = plotRange[-1]

    # Select only the range specified
    position = position[start_idx:end_idx]
    velocity = velocity[start_idx:end_idx]
    acceleration = acceleration[start_idx:end_idx]
    jerk = jerk[start_idx:end_idx]
    
    # Parameters for time normalization based on the selected range
    duration = len(jerk) / fs  # total duration in seconds for the selected segment
    # Create original and standardized time vectors for the selected segment
    t_orig = np.linspace(0, duration, num=len(jerk))
    t_std = np.linspace(0, duration, num=target_samples)
    
    # Warp the jerk segment to the standardized time base using linear interpolation
    warped_jerk = np.interp(t_std, t_orig, jerk)

    # Calculate the integral of the squared, warped jerk segment
    jerk_squared_integral = np.trapezoid(warped_jerk**2, t_std)

    # Get the peak speed for the current segment
    vpeak = velocity.max()
    dimensionless_jerk = (duration**3 / vpeak**2) * jerk_squared_integral
    LDLJ = -math.log(abs(dimensionless_jerk), math.e)
    
    # Plot Position, Velocity, Acceleration, Original Jerk and Normalized Jerk as 5 subplots
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(5, 1, figsize=(12, 8))
    
    axs[0].plot(position, color='blue')
    axs[0].set_title('Position')
    
    axs[1].plot(velocity, color='green')
    axs[1].set_title('Velocity')
    
    axs[2].plot(acceleration, color='red')
    axs[2].set_title('Acceleration')
    
    axs[3].plot(jerk, color='purple')
    axs[3].set_title('Original Jerk')
    
    axs[4].plot(t_std, warped_jerk, color='orange', linestyle='--')
    # Convert x-axis tick labels to percentage of the selected trial duration
    percent_ticks = np.linspace(0, 100, 6)
    time_ticks = np.linspace(0, duration, 6)
    axs[4].set_xticks(time_ticks)
    axs[4].set_xticklabels([f"{int(p)}%" for p in percent_ticks])
    axs[4].set_title(f'Normalized Jerk (LDLJ: {LDLJ:.2f})')
    
    plt.tight_layout()
    plt.show()

# Now you can call your function with either hand to see smooth vs. jerky
plot_normalized_jerk(results, subject="07/22/HW", hand="left", trial=1,
                     file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT02.csv')

plot_normalized_jerk(results, subject="07/22/HW", hand="right", trial=1,
                     file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT01.csv')
