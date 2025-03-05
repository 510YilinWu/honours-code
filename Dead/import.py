import pandas as pd

# Define file path
csvPath = '/Users/yilinwu/Desktop/honours data/ModelOutput.csv'

# Read the CSV file, treating empty fields as NaN
df = pd.read_csv(csvPath, na_values=[''])

print(df.head())
print(f"Dataset shape: {df.shape}")




# Extract unique subjects dynamically from column 3 and later for row one, excluding 'Unnamed' columns
subjects = set(col.split(":")[0] for col in df.columns[2:] 
               if ":" in col and not pd.isna(df.at[0, col]) 
               and "Unnamed" not in col)
print(f"Subjects found in row one: {subjects}")

# Define variable names 
variables = [
    "LWJC", "RWJC", "Thorax1", "Thorax2", "Thorax3", "ThoraxO", 
    "leftHand1", "leftHand2", "leftHand3", "leftHandO", 
    "rightHand1", "rightHand2", "rightHand3", "rightHandO"
]
