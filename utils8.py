import pandas as pd
import numpy as np

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
