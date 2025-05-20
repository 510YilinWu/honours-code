import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Load the Excel file (assuming it's named 'data.xlsx')
file_path = '/Users/yilinwu/Desktop/box_and_block_test_norm.xlsx'  # Update with your actual file path
df = pd.read_excel(file_path, sheet_name='Hand Dominance')

# Separate data by hand dominance and sex
right_dominant = df[df['Subject Dominance'] == 'Right dominant']
left_dominant = df[df['Subject Dominance'] == 'Left dominant']

# Right dominant hand
# in right dominant hand, is male left and right mean different?
male_right_dominant = right_dominant[right_dominant['Sex'] == 'Males']
m_rd_r=male_right_dominant[male_right_dominant['Hand'] == 'Right']
m_rd_l=male_right_dominant[male_right_dominant['Hand'] == 'Left']

# in right dominant hand, is Female left and right mean different?
female_right_dominant = right_dominant[right_dominant['Sex'] == 'Females']
f_rd_r=female_right_dominant[female_right_dominant['Hand'] == 'Right']
f_rd_l=female_right_dominant[female_right_dominant['Hand'] == 'Left']

# Left dominant hand
# in left dominant hand, is male left and right mean different?
male_left_dominant = left_dominant[left_dominant['Sex'] == 'Males']
m_ld_r=male_left_dominant[male_left_dominant['Hand'] == 'Right']
m_ld_l=male_left_dominant[male_left_dominant['Hand'] == 'Left']

# in left dominant hand, is Female left and right mean different?
female_left_dominant = left_dominant[left_dominant['Sex'] == 'Females']
f_ld_r=female_left_dominant[female_left_dominant['Hand'] == 'Right']
f_ld_l=female_left_dominant[female_left_dominant['Hand'] == 'Left']





# Function to perform t-test and print results
def perform_ttest(group1, group2, label):
    stat, p_value = ttest_ind(group1['Mean'], group2['Mean'], nan_policy='omit')
    print(f"{label}: t-statistic = {stat:.3f}, p-value = {p_value:.3e}")
    if p_value < 0.05:
        print(f"  -> Significant difference (p < 0.05)")
    else:
        print(f"  -> No significant difference (p >= 0.05)")

# Perform t-tests
perform_ttest(m_rd_r, m_rd_l, "Male Right Dominant: Right vs Left Hand")
perform_ttest(f_rd_r, f_rd_l, "Female Right Dominant: Right vs Left Hand")
perform_ttest(m_ld_r, m_ld_l, "Male Left Dominant: Right vs Left Hand")
perform_ttest(f_ld_r, f_ld_l, "Female Left Dominant: Right vs Left Hand")