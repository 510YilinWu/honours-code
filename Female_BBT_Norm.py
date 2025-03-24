import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file (assuming it's named 'data.xlsx')
file_path = '/Users/yilinwu/Desktop/box_and_block_test_norm.xlsx'  # Update with your actual file path
df = pd.read_excel(file_path, sheet_name='Females')

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(8, 18), sharex=True)

# Right hand data
right_hand_df = df[df['Hand'] == 'R']
x_r = right_hand_df['Age (yr)']
Mean_r = right_hand_df['Mean']
SE_r = right_hand_df['SE']
SD_r = right_hand_df['SD']
Low_r = right_hand_df['Low']
High_r = right_hand_df['High']

axes[0].errorbar(x_r, Mean_r, yerr=SE_r, fmt='o', label='Mean ± SE', capsize=5, color='blue')
axes[0].errorbar(x_r, Mean_r, yerr=SD_r, fmt='o', label='Mean ± SD', capsize=5, color='purple')
axes[0].scatter(x_r, Low_r, color='green', label='Low', zorder=5)
axes[0].scatter(x_r, High_r, color='orange', label='High', zorder=5)
axes[0].fill_between(x_r, Low_r, High_r, color='lightblue', alpha=0.5, label='Range (Low to High)')
axes[0].scatter(x_r, Mean_r, color='red', label='Mean', zorder=5)
axes[0].set_title('Right Hand')
axes[0].set_ylabel('Values')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.6)

# Left hand data
left_hand_df = df[df['Hand'] == 'L']
x_l = left_hand_df['Age (yr)']
Mean_l = left_hand_df['Mean']
SE_l = left_hand_df['SE']
SD_l = left_hand_df['SD']
Low_l = left_hand_df['Low']
High_l = left_hand_df['High']

axes[1].errorbar(x_l, Mean_l, yerr=SE_l, fmt='o', label='Mean ± SE', capsize=5, color='blue')
axes[1].errorbar(x_l, Mean_l, yerr=SD_l, fmt='o', label='Mean ± SD', capsize=5, color='purple')
axes[1].scatter(x_l, Low_l, color='green', label='Low', zorder=5)
axes[1].scatter(x_l, High_l, color='orange', label='High', zorder=5)
axes[1].fill_between(x_l, Low_l, High_l, color='lightblue', alpha=0.5, label='Range (Low to High)')
axes[1].scatter(x_l, Mean_l, color='red', label='Mean', zorder=5)
axes[1].set_title('Left Hand')
axes[1].set_ylabel('Values')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)

# Combined plot for both hands
axes[2].errorbar(x_r, Mean_r, yerr=SE_r, fmt='o', label='Right Hand Mean ± SE', capsize=5, color='red')
axes[2].errorbar(x_l, Mean_l, yerr=SE_l, fmt='o', label='Left Hand Mean ± SE', capsize=5, color='green')
axes[2].scatter(x_r, Mean_r, color='red', label='Right Hand Mean', zorder=5)
axes[2].scatter(x_l, Mean_l, color='green', label='Left Hand Mean', zorder=5)
axes[2].set_title('Both Hands')
axes[2].set_xlabel('Age (yr)')
axes[2].set_ylabel('Values')
axes[2].legend()
axes[2].grid(True, linestyle='--', alpha=0.6)

# Adjust layout
plt.tight_layout()
plt.show()
