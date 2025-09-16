import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

def load_and_compute_sbbt_result(csv_filepath="/Users/yilinwu/Desktop/Yilin-Honours/sBBTResult.csv"):
    """
    Load sBBTResult from CSV into a DataFrame and compute right and left hand scores.
    
    Parameters:
        csv_filepath (str): Path to the CSV file containing sBBTResult.

    Returns:
        DataFrame: The resulting DataFrame with new 'right_score' and 'left_score' columns.
    """

    sBBTResult = pd.read_csv(csv_filepath)

    # Compute right hand values using columns 2 and 4 (0-indexed columns 1 and 3)
    sBBTResult['right_score'] = np.maximum(sBBTResult.iloc[:, 1], sBBTResult.iloc[:, 3])

    # Compute left hand values using columns 3 and 5 (0-indexed columns 2 and 4)
    sBBTResult['left_score'] = np.maximum(sBBTResult.iloc[:, 2], sBBTResult.iloc[:, 4])

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
import matplotlib.pyplot as plt

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

    # Add box plot for non_dominant and dominant scores, with non_dominant on left
    plt.figure(figsize=(8, 6))
    plt.boxplot([sBBTResult['non_dominant'], sBBTResult['dominant']], labels=['Non-Dominant', 'Dominant'])
    
    # Overlay each value as dots with slight horizontal jitter for non_dominant and dominant scores
    non_dominant_values = sBBTResult['non_dominant']
    dominant_values = sBBTResult['dominant']
    
    # Create jitter for x positions
    x_nondom = np.random.normal(1, 0.04, size=len(non_dominant_values))
    x_dom = np.random.normal(2, 0.04, size=len(dominant_values))
    
    plt.scatter(x_nondom, non_dominant_values, alpha=0.6, color='orange', label='Non-Dominant Data Points')
    plt.scatter(x_dom, dominant_values, alpha=0.6, color='blue', label='Dominant Data Points')
    
    plt.ylabel("sBBT Scores")
    plt.show()

    dominant_stats = (dom_min, dom_max, dom_mean, dom_std)
    non_dominant_stats = (non_min, non_max, non_mean, non_std)
    
    sBBTResult_stats = {
        'dominant': dominant_stats,
        'non_dominant': non_dominant_stats
    }

    return sBBTResult_stats