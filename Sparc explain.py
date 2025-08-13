import numpy as np
import matplotlib.pyplot as plt

def sparc_with_plots(movement, fs=100, padlevel=4, fc=10.0, amp_th=0.05):
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
    f = np.linspace(0, fs, int(nfft), endpoint=False)
    Mf = np.abs(np.fft.fft(movement, int(nfft)))
    Mf = Mf / np.max(Mf)

    plt.subplot(3, 2, 2)
    plt.plot(f[:nfft // 2], Mf[:nfft // 2])
    plt.title("2. Full Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Magnitude")

    # 3. Cutoff Frequency
    fc_idx = np.where(f <= fc)[0]
    f_sel = f[fc_idx]
    Mf_sel = Mf[fc_idx]

    plt.subplot(3, 2, 3)
    plt.plot(f_sel, Mf_sel)
    plt.title("3. Spectrum Below Cutoff (fc = 10 Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    # 4. Amplitude Threshold Filtering
    amp_mask = Mf_sel >= amp_th
    f_cut = f_sel[amp_mask]
    Mf_cut = Mf_sel[amp_mask]

    plt.subplot(3, 2, 4)
    plt.plot(f_sel, Mf_sel, color='lightgray', label='All under 10Hz')
    plt.plot(f_cut, Mf_cut, color='blue', label='Above threshold')
    plt.axhline(y=amp_th, color='red', linestyle='--', label='Threshold')
    plt.title("4. After Amplitude Thresholding")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()

    # 5. Spectral Arc Length Calculation
    df = np.diff(f_cut) / (f_cut[-1] - f_cut[0])  # Normalize frequency
    dM = np.diff(Mf_cut)
    arc_length = -np.sum(np.sqrt(df ** 2 + dM ** 2))

    plt.subplot(3, 2, 5)
    plt.plot(f_cut, Mf_cut, marker='o')
    for i in range(len(df)):
        plt.plot([f_cut[i], f_cut[i+1]], [Mf_cut[i], Mf_cut[i+1]], 'k--')
    plt.title("5. Arc Segments Used in SPARC")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.subplot(3, 2, 6)
    plt.text(0.1, 0.5, f"SPARC Value:\n{arc_length:.4f}", fontsize=20)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return arc_length

# Example: A smooth Gaussian speed profile
if __name__ == "__main__":
    t = np.arange(0, 2, 0.01)
    speed = np.exp(-5 * (t - 1)**2)
    sparc_value = sparc_with_plots(speed, fs=100)


# import numpy as np
# import matplotlib.pyplot as plt

# def sparc_with_plots(movement, fs=100, padlevel=4, fc=10.0, amp_th=0.05, title_prefix=""):
#     nfft = int(2 ** (np.ceil(np.log2(len(movement))) + padlevel))
#     f = np.linspace(0, fs, int(nfft), endpoint=False)
#     Mf = np.abs(np.fft.fft(movement, int(nfft)))
#     Mf = Mf / np.max(Mf)

#     # Low-pass filter
#     fc_idx = np.where(f <= fc)[0]
#     f_sel = f[fc_idx]
#     Mf_sel = Mf[fc_idx]

#     # Amplitude threshold
#     amp_mask = Mf_sel >= amp_th
#     if not np.any(amp_mask):
#         print("No frequencies above threshold.")
#         return None

#     f_cut = f_sel[amp_mask]
#     Mf_cut = Mf_sel[amp_mask]

#     # Arc length calculation
#     df = np.diff(f_cut) / (f_cut[-1] - f_cut[0])
#     dM = np.diff(Mf_cut)
#     arc_length = -np.sum(np.sqrt(df ** 2 + dM ** 2))

#     # Plotting
#     plt.figure(figsize=(14, 6))

#     # Original signal
#     t = np.arange(len(movement)) / fs
#     plt.subplot(1, 3, 1)
#     plt.plot(t, movement)
#     plt.title(f"{title_prefix}1. Time Domain")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Speed")

#     # Frequency spectrum
#     plt.subplot(1, 3, 2)
#     plt.plot(f[:nfft//2], Mf[:nfft//2])
#     plt.axhline(y=amp_th, color='red', linestyle='--', label="Amp Threshold")
#     plt.axvline(x=fc, color='green', linestyle='--', label="Cutoff Frequency")
#     plt.title(f"{title_prefix}2. Frequency Spectrum")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Magnitude")
#     plt.legend()

#     # Arc length segments
#     plt.subplot(1, 3, 3)
#     plt.plot(f_cut, Mf_cut, marker='o')
#     for i in range(len(df)):
#         plt.plot([f_cut[i], f_cut[i+1]], [Mf_cut[i], Mf_cut[i+1]], 'k--')
#     plt.title(f"{title_prefix}3. SPARC Arc Segments\nSPARC = {arc_length:.4f}")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Magnitude")

#     plt.tight_layout()
#     plt.show()
    
#     return arc_length

# # Generate signals
# fs = 100
# t = np.arange(0, 2, 1/fs)

# # Smooth signal (Gaussian)
# smooth = np.exp(-5 * (t - 1)**2)

# # Jerky signal (same + noise)
# np.random.seed(42)
# jerky = smooth + 0.1 * np.random.randn(len(t))

# # Run and compare
# print("🔵 Smooth Movement")
# smooth_sparc = sparc_with_plots(smooth, fs=fs, title_prefix="Smooth - ")

# print("🔴 Jerky/Noisy Movement")
# jerky_sparc = sparc_with_plots(jerky, fs=fs, title_prefix="Jerky - ")

# print(f"\nSmooth SPARC: {smooth_sparc:.4f}")
# print(f"Jerky SPARC:  {jerky_sparc:.4f}")



# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# fs = 100  # Sampling frequency (samples per second)
# t = np.arange(0, 2, 1/fs)  # Time vector: 2 seconds
# padlevel = 4  # Zero padding level

# # Example smooth movement: Gaussian speed profile
# movement = np.exp(-5 * (t - 1)**2)

# # Calculate zero padded FFT length
# nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

# # Frequency array
# f = np.arange(0, fs, fs / nfft)

# # FFT magnitude and normalization
# Mf = np.abs(np.fft.fft(movement, nfft))
# Mf = Mf / max(Mf)

# # Plot time-domain signal
# plt.figure(figsize=(14,5))

# plt.subplot(1,2,1)
# plt.plot(t, movement)
# plt.title("Time Domain: Speed Profile")
# plt.xlabel("Time (seconds)")
# plt.ylabel("Speed")

# # Plot frequency spectrum (only up to Nyquist fs/2)
# plt.subplot(1,2,2)
# plt.plot(f[:nfft//2], Mf[:nfft//2])
# plt.title("Frequency Domain: Normalized Magnitude Spectrum")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Normalized Magnitude")
# plt.tight_layout()
# plt.show()


# --- CALCULATE SPARC FOR EACH TEST WINDOW ---
sparc_values = {}

# Iterate through each test window for the specified trial
trial_windows_t = test_windows['06/19/CZ']['right']['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/06/19/CZ/CZ_tBBT01.csv']
speed_t = results['06/19/CZ']['right'][1]['/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/06/19/CZ/CZ_tBBT01.csv']["traj_space"]["RFIN"][1]

for i, window in enumerate(trial_windows_t):
    start_frame, end_frame = window
    window_speed = speed_t[start_frame:end_frame]
    sal_2,_,_ = utils.sparc(window_speed, fs=200.)

    sparc_values[f"Window {i + 1}"] = sal

# Save SPARC values to a dictionary or file
sparc_save_path = os.path.join(DataProcess_folder, "sparc_values.csv")
pd.DataFrame([sparc_values]).to_csv(sparc_save_path, index=False)
print(f"SPARC values saved to {sparc_save_path}")
