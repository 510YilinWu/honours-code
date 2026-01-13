import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_compute_sbbt_result(csv_filepath="/Users/yilinwu/Desktop/Yilin-Honours/sBBTResult.csv"):
    """
    Load sBBTResult from CSV into a DataFrame and compute right and left hand scores.
    
    Parameters:
        csv_filepath (str): Path to the CSV file containing sBBTResult.

    Returns:
        DataFrame: The resulting DataFrame with new 'right_score' and 'left_score' columns.
    """

    sBBTResult = pd.read_csv(csv_filepath)

    # # Compute right hand values using columns 2 and 4 (0-indexed columns 1 and 3)
    # sBBTResult['right_score'] = np.maximum(sBBTResult.iloc[:, 1], sBBTResult.iloc[:, 3])

    # # Compute left hand values using columns 3 and 5 (0-indexed columns 2 and 4)
    # sBBTResult['left_score'] = np.maximum(sBBTResult.iloc[:, 2], sBBTResult.iloc[:, 4])

    sBBTResult['right_score'] = sBBTResult.iloc[:, 1]
    sBBTResult['left_score'] = sBBTResult.iloc[:, 2]

    return sBBTResult



def swap_and_rename_sbbt_result(sBBTResult):
    """
    For rows in sBBTResult corresponding to subjects 'PY' or 'MC', 
    swap the 'left_score' and 'right_score' values, then rename 
    'left_score' to 'non_dominant' and 'right_score' to 'dominant'.
    
    Parameters:
        sBBTResult (DataFrame): The DataFrame containing sBBTResult data. 
    Returns:
        DataFrame: The modified DataFrame with swapped and renamed columns.
    """
    # Identify rows where Subject is 'PY' or 'MC'
    mask = sBBTResult['Subject'].isin(['PY', 'MC'])

    # Swap left_score and right_score for these rows
    sBBTResult.loc[mask, ['left_score', 'right_score']] = sBBTResult.loc[mask, ['right_score', 'left_score']].values

    # Rename columns
    sBBTResult.rename(columns={'left_score': 'non_dominant', 'right_score': 'dominant'}, inplace=True)

    return sBBTResult

# Calculate sBBTResult score statistics for dominant and non-dominant columns

def compute_sbbt_result_stats(sBBTResult):
    # Compute statistics for 'dominant' scores
    dom_min = sBBTResult['dominant'].min()
    dom_max = sBBTResult['dominant'].max()
    dom_mean = sBBTResult['dominant'].mean()
    dom_std = sBBTResult['dominant'].std()
    
    # Compute statistics for 'non_dominant' scores
    non_min = sBBTResult['non_dominant'].min()
    non_max = sBBTResult['non_dominant'].max()
    non_mean = sBBTResult['non_dominant'].mean()
    non_std = sBBTResult['non_dominant'].std()
    
    print("sBBT Result Score Statistics:")
    print("Dominant Score -> Minimum: {}, Maximum: {}, Mean: {}, Standard Deviation: {}".format(
        dom_min, dom_max, dom_mean, dom_std))
    print("Non-Dominant Score -> Minimum: {}, Maximum: {}, Mean: {}, Standard Deviation: {}".format(
        non_min, non_max, non_mean, non_std))
    
    # Perform paired Wilcoxon signed-rank test comparing dominant and non_dominant scores
    statistic, p_value = wilcoxon(sBBTResult["dominant"], sBBTResult["non_dominant"])
    print("Paired Wilcoxon signed-rank test result: Statistic = {:.4f}, p-value = {:.4f}".format(statistic, p_value))

    dominant_stats = (dom_min, dom_max, dom_mean, dom_std)
    non_dominant_stats = (non_min, non_max, non_mean, non_std)
    
    sBBTResult_stats = {
        'dominant': dominant_stats,
        'non_dominant': non_dominant_stats
    }

    return sBBTResult_stats


def plot_sbbt_boxplot(sBBTResult):
    """
    Plot a boxplot of sBBT scores by hand (Non-dominant vs Dominant) on a given axis.
    It overlays a swarmplot and draws dashed lines connecting each subject's pair of scores.
    
    Parameters:
        ax: matplotlib.axes.Axes - the axis on which to plot.
        sBBTResult: DataFrame - indexed by subjects with columns "non_dominant" and "dominant".
    """
    # Create a figure and axis for the boxplot
    fig, ax = plt.subplots(figsize=(6, 5))
    # Prepare a DataFrame for sBBT scores per subject.
    df_scores = pd.DataFrame({
        "Subject": sBBTResult.index,
        "Non-dominant": sBBTResult["non_dominant"],
        "Dominant": sBBTResult["dominant"]
    })
    
    # Melt the DataFrame to long format.
    df_melt = df_scores.melt(id_vars="Subject", var_name="Hand", value_name="sBBT Score")
    
    # Define the order for hands.
    order = ["Non-dominant", "Dominant"]
    
    # Plot the boxplot and swarmplot.
    sns.boxplot(x="Hand", y="sBBT Score", data=df_melt, palette="Set2", order=order, ax=ax)
    sns.swarmplot(x="Hand", y="sBBT Score", data=df_melt, color="black", size=5, alpha=0.8, order=order, ax=ax)
    
    # Set title and labels.
    # ax.set_title("Box Plot of sBBT Scores by Hand")
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel("Hand", fontsize=20)
    ax.set_ylabel("sBBT Score", fontsize=20)
    
    # # Draw dashed lines connecting each subject's non-dominant and dominant scores.
    # for subject in df_scores["Subject"]:
    #     nd_val = df_scores.loc[df_scores["Subject"] == subject, "Non-dominant"].values[0]
    #     d_val = df_scores.loc[df_scores["Subject"] == subject, "Dominant"].values[0]
    #     ax.plot([0, 1], [nd_val, d_val], color="gray", linestyle="--", linewidth=1, alpha=0.7)

def analyze_sbbt_results(sBBTResult):
    """
    Analyze sBBT results by computing ICC, Spearman correlations, and plotting Bland-Altman plots
    for Left and Right hand scores, and save all calculations in a dictionary.

    Parameters:
        sBBTResult (DataFrame): DataFrame containing columns 'Right', 'Right.1', 'Left' and 'Left.1'.
        show_value_in_legend (bool): If True, include numeric values in plot legends/labels.
    
    Returns:
        dict: Dictionary containing all the computed statistics.
    """
    import scipy.stats as stats
    import matplotlib.pyplot as plt

    # Prepare long-format DataFrame using sBBTResult values
    df = pd.DataFrame({
        'Subject': list(range(1, len(sBBTResult) + 1)) * 2,
        'Test': ['Test1'] * len(sBBTResult) + ['Test2'] * len(sBBTResult),
        'Right': list(sBBTResult['Right']) + list(sBBTResult['Right.1']),
        'Left': list(sBBTResult['Left']) + list(sBBTResult['Left.1'])
    })

    # Compute ICC for Right hand scores
    icc_right = pg.intraclass_corr(data=df, targets='Subject', raters='Test', ratings='Right')
    print("ICC for Right hand:")
    print(icc_right[['Type', 'ICC', 'F', 'pval', 'CI95%']])

    # Compute ICC for Left hand scores
    icc_left = pg.intraclass_corr(data=df, targets='Subject', raters='Test', ratings='Left')
    print("ICC for Left hand:")
    print(icc_left[['Type', 'ICC', 'F', 'pval', 'CI95%']])

    # Calculate Spearman correlation for the Right hand scores (Test1 vs Test2)
    right_corr, right_p = stats.spearmanr(sBBTResult['Right'], sBBTResult['Right.1'])
    print("Spearman correlation for Right hand scores (Test1 vs Test2):\n", right_corr, "p-value:", right_p)

    # Calculate Spearman correlation for the Left hand scores (Test1 vs Test2)
    left_corr, left_p = stats.spearmanr(sBBTResult['Left'], sBBTResult['Left.1'])
    print("Spearman correlation for Left hand scores (Test1 vs Test2):\n", left_corr, "p-value:", left_p)

    # Calculate Pearson correlation for the Right hand scores (Test1 vs Test2)
    right_corr, right_p = stats.pearsonr(sBBTResult['Right'], sBBTResult['Right.1'])
    print("Pearson correlation for Right hand scores (Test1 vs Test2):\n", right_corr, "p-value:", right_p)

    # Calculate Pearson correlation for the Left hand scores (Test1 vs Test2)
    left_corr, left_p = stats.pearsonr(sBBTResult['Left'], sBBTResult['Left.1'])
    print("Pearson correlation for Left hand scores (Test1 vs Test2):\n", left_corr, "p-value:", left_p)

    # Bland-Altman calculations for Left hand scores
    left1 = pd.Series(sBBTResult['Left'])
    left2 = pd.Series(sBBTResult['Left.1'])
    mean_left = (left1 + left2) / 2
    diff_left = left1 - left2
    md_left = diff_left.mean()
    sd_left = diff_left.std()

    # Bland-Altman calculations for Right hand scores
    right1 = pd.Series(sBBTResult['Right'])
    right2 = pd.Series(sBBTResult['Right.1'])
    mean_right = (right1 + right2) / 2
    diff_right = right1 - right2
    md_right = diff_right.mean()
    sd_right = diff_right.std()

    print(
        f"Left Hand - Mean Difference: {md_left:.2f} ± {sd_left:.2f} "
        f"(95% CI: {md_left - 1.96 * sd_left:.2f} to {md_left + 1.96 * sd_left:.2f})"
    )
    print(
        f"Right Hand - Mean Difference: {md_right:.2f} ± {sd_right:.2f} "
        f"(95% CI: {md_right - 1.96 * sd_right:.2f} to {md_right + 1.96 * sd_right:.2f})"
    )

    # Save all calculations in a dictionary.
    results = {
        # "icc_right": icc_right[['Type', 'ICC', 'F', 'pval', 'CI95%']].to_dict(orient='records'),
        # "icc_left": icc_left[['Type', 'ICC', 'F', 'pval', 'CI95%']].to_dict(orient='records'),
        "spearman_right": {"correlation": right_corr, "p_value": right_p},
        "spearman_left": {"correlation": left_corr, "p_value": left_p},
        "bland_altman_left": {
            "mean_diff": md_left,
            "sd": sd_left,
            "upper_limit": md_left + 1.96 * sd_left,
            "lower_limit": md_left - 1.96 * sd_left
        },
        "bland_altman_right": {
            "mean_diff": md_right,
            "sd": sd_right,
            "upper_limit": md_right + 1.96 * sd_right,
            "lower_limit": md_right - 1.96 * sd_right
        }
    }

    return results
