import pandas as pd

# Define the path to the CSV file
file_path = '/Users/yilinwu/Desktop/honours data/YW Trial 7.csv'

# Read the CSV file into a DataFrame
RawData = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(RawData.head())