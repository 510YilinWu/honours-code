
import csv
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns

# Load Motor Experiences from CSV into a dictionary
def load_motor_experiences(csv_filepath):
    MotorExperiences = {}
    with open(csv_filepath, mode='r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        rows = list(reader)

    if rows:
        headers = rows[0]  # List of all headers
        for row in rows[1:]:
            subject_name = row[0]
            MotorExperiences[subject_name] = {header: value for header, value in zip(headers[1:], row[1:])}
    
    print(f"Imported data for {len(MotorExperiences)} subjects.")
    return MotorExperiences

# Calculate demographic variables from MotorExperiences
def display_motor_experiences_stats(MotorExperiences):
    """
    Takes a MotorExperiences dictionary, converts it to a DataFrame,
    and prints summary statistics for Age, Gender, Handedness, and domain characteristics.
    """

    # Convert MotorExperiences dictionary to a DataFrame (rows: subjects, columns: demographic fields)
    df_me = pd.DataFrame.from_dict(MotorExperiences, orient='index')

    # ---------------------------
    # Age (mean ± sd)
    df_me['Age'] = pd.to_numeric(df_me['Age'], errors='coerce')
    age_mean = df_me['Age'].mean()
    age_std = df_me['Age'].std()
    print(f"Age: {age_mean:.2f} ± {age_std:.2f}")

    # ---------------------------
    # Gender counts and percentages
    print("\nGender:")
    gender_counts = df_me['Gender'].value_counts(dropna=False)
    total_gender = gender_counts.sum()
    for gender, count in gender_counts.items():
        pct = count / total_gender * 100
        print(f"  {gender}: {count} ({pct:.1f}%)")

    # ---------------------------
    # Handedness counts and percentages
    print("\nHandedness:")
    handedness_counts = df_me['handedness'].value_counts(dropna=False)
    total_hand = handedness_counts.sum()
    for hand, count in handedness_counts.items():
        pct = count / total_hand * 100
        print(f"  {hand}: {count} ({pct:.1f}%)")

    # ---------------------------
    # Domain characteristics for each domain
    domains = ['physical', 'musical', 'digital', 'other']
    for domain in domains:
        print(f"\nDomain: {domain.capitalize()}")
        
        # Level: e.g., beginner, intermediate, advanced
        level_col = f"{domain}_level"
        if level_col in df_me.columns:
            level_counts = df_me[level_col].value_counts(dropna=False)
            print("  Level counts:")
            for level, count in level_counts.items():
                print(f"    {level}: {count}")
        
        # Hours per week
        h_week_col = f"{domain}_h_week"
        if h_week_col in df_me.columns:
            df_me[h_week_col] = pd.to_numeric(df_me[h_week_col], errors='coerce')
            h_week_mean = df_me[h_week_col].mean()
            h_week_std = df_me[h_week_col].std()
            print(f"  Hours per week: {h_week_mean:.2f} ± {h_week_std:.2f}")
        
        # Number of years
        n_year_col = f"{domain}_n_year"
        if n_year_col in df_me.columns:
            df_me[n_year_col] = pd.to_numeric(df_me[n_year_col], errors='coerce')
            n_year_mean = df_me[n_year_col].mean()
            n_year_std = df_me[n_year_col].std()
            print(f"  Number of years: {n_year_mean:.2f} ± {n_year_std:.2f}")
        
        # Total hours
        h_total_col = f"{domain}_h_total"
        if h_total_col in df_me.columns:
            df_me[h_total_col] = pd.to_numeric(df_me[h_total_col], errors='coerce')
            h_total_mean = df_me[h_total_col].mean()
            h_total_std = df_me[h_total_col].std()
            print(f"  Total hours: {h_total_mean:.2f} ± {h_total_std:.2f}")
        
        # Weighted total hours
        h_total_weighted_col = f"{domain}_h_total_weighted"
        if h_total_weighted_col in df_me.columns:
            df_me[h_total_weighted_col] = pd.to_numeric(df_me[h_total_weighted_col], errors='coerce')
            h_total_weighted_mean = df_me[h_total_weighted_col].mean()
            h_total_weighted_std = df_me[h_total_weighted_col].std()
            print(f"  Weighted total hours: {h_total_weighted_mean:.2f} ± {h_total_weighted_std:.2f}")

def update_overall_h_total_weighted(MotorExperiences):
    """
    Calculate overall_h_total_weighted as the sum of 
    physical_h_total_weighted, musical_h_total_weighted, and digital_h_total_weighted
    for each subject, and update the MotorExperiences dictionary.
    """
    for subject, metrics in MotorExperiences.items():
        try:
            physical = float(metrics.get('physical_h_total_weighted', 0))
        except (ValueError, TypeError):
            physical = 0
        try:
            musical = float(metrics.get('musical_h_total_weighted', 0))
        except (ValueError, TypeError):
            musical = 0
        try:
            digital = float(metrics.get('digital_h_total_weighted', 0))
        except (ValueError, TypeError):
            digital = 0

        overall = physical + musical + digital
        MotorExperiences[subject]['overall_h_total_weighted'] = overall

    print("Updated MotorExperiences with overall_h_total_weighted:")
    for subject, metrics in MotorExperiences.items():
        print(f"{subject}: {metrics.get('overall_h_total_weighted')}")

# Examine correlations between motor experience metrics and sBBT scores by extracting the highest score for each hand
def analyze_motor_experience_correlations(motor_keys, score_columns, sbbt_df, MotorExperiences):
    """
    For each motor experience metric (motor_keys) and each score column in sbbt_df (e.g., right_score, left_score),
    this function computes the Pearson correlation between the two after merging the MotorExperiences data with sbbt_df.
    It then creates a heatmap showing all correlation values and their corresponding p-values in one plot.
    
    Each cell in the heatmap is annotated with the correlation coefficient and p-value on a new line.

    Parameters:
        motor_keys (list): List of keys in MotorExperiences to analyze.
        score_columns (list): List of score column names in sbbt_df.
        sbbt_df (DataFrame): DataFrame containing score data with a "Subject" column.
        MotorExperiences (dict): Dictionary where each key is a subject and each value is a dict of motor experience metrics.
    """
    import matplotlib.pyplot as plt

    # Prepare empty DataFrames to store correlation coefficients and p-values
    corr_table = pd.DataFrame(index=motor_keys, columns=score_columns)
    p_table = pd.DataFrame(index=motor_keys, columns=score_columns)

    # Loop over all motor experience keys and score columns
    for motor_key in motor_keys:
        # Create a DataFrame from MotorExperiences for the given motor_key
        motor_values = {subject: data.get(motor_key) for subject, data in MotorExperiences.items()}
        df_motor = pd.DataFrame(list(motor_values.items()), columns=['Subject', motor_key])
        
        # Merge with sbbt_df on the "Subject" column
        merged_df = pd.merge(sbbt_df, df_motor, on='Subject', how='inner')
        
        # Convert the motor experience values and score columns to numeric
        merged_df[motor_key] = pd.to_numeric(merged_df[motor_key], errors='coerce')
        for score_column in score_columns:
            merged_df[score_column] = pd.to_numeric(merged_df[score_column], errors='coerce')
            # Drop rows with NaN values in either column
            merged_clean = merged_df.dropna(subset=[motor_key, score_column])
            if merged_clean.empty:
                corr_table.loc[motor_key, score_column] = np.nan
                p_table.loc[motor_key, score_column] = np.nan
            else:
                corr, p_value = pearsonr(merged_clean[motor_key], merged_clean[score_column])
                corr_table.loc[motor_key, score_column] = corr
                p_table.loc[motor_key, score_column] = p_value
                print("Pearson correlation between {} and {}: {:.3f}".format(motor_key, score_column, corr))
                print("P-value: {:.3f}".format(p_value))

    # Create annotations combining correlation and p-value
    annotations = corr_table.copy()
    for motor_key in motor_keys:
        for score_column in score_columns:
            if pd.isna(corr_table.loc[motor_key, score_column]):
                annotations.loc[motor_key, score_column] = ""
            else:
                annotations.loc[motor_key, score_column] = f"{corr_table.loc[motor_key, score_column]:.3f}\np:{p_table.loc[motor_key, score_column]:.3f}"

    # Plot the heatmap of the correlation table with annotations
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_table.astype(float), annot=annotations, fmt='', cmap='coolwarm', center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
