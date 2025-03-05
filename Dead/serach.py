"""
    for Model Outputs only
"""
import csv
import pandas as pd
import pprint

def search_csv(file_path):
    """
    Searches for rows in a CSV file where the first column contains 'Model Outputs'.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        list: A list of row numbers where 'Model Outputs' is found in the first column.
    """
    results = []
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row_number, row in enumerate(reader, start=1):
            if row and "Model Outputs" in row[0]:  # Check first column for "Model Outputs"
                results.append(row_number)  # Store only the row number for matches
    return results

file_path = '/Users/yilinwu/Desktop/honours data/ModelOutput.csv'
results = search_csv(file_path)

# for row_number in results:
#     print(f"Found 'Model Outputs' in column 1, row {row_number}")

    
def get_capture_rates(file_path, results):
    """
    Retrieves capture rates from the rows following the rows where 'Model Outputs' is found.

    Args:
        file_path (str): The path to the CSV file.
        results (list): A list of row numbers where 'Model Outputs' is found.

    Returns:
        list: A list of capture rates from the rows following the matches.
    """
    capture_rates = []

    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row_number, row in enumerate(reader, start=1):
            if row_number in results:
                next_row = next(reader, None)
                if next_row:
                    capture_rates.append(next_row[0])  # Assuming capture rate is in the same column

    return capture_rates

capture_rates = get_capture_rates(file_path, results)
# print(f"Found capture rates: {capture_rates}")


def get_row(file_path, row_number):
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for current_row_number, row in enumerate(reader, start=1):
            if current_row_number == row_number:
                non_nan_columns = {index: content for index, content in enumerate(row) if content}
                return row, non_nan_columns
    return None, {}

row_3, non_nan_columns_row_3 = get_row(file_path, 3)
# print(f"Contents of row 3: {row_3}")
print(f"Non-NaN columns in row 3: {non_nan_columns_row_3}")

subjects = set()
variables = []
seen_variables = set()

for content in non_nan_columns_row_3.values():
    if ":" in content:
        subject, variable = content.split(":", 1)
        subjects.add(subject.strip())
        if variable.strip() not in seen_variables:
            variables.append(variable.strip())
            seen_variables.add(variable.strip())

# print(f"Subjects found: {subjects}")
# print(f"Variables found: {variables}")


# Initialize the dictionary to store the data
D = {}

# Read the CSV file into a DataFrame for easier manipulation
df = pd.read_csv(file_path)

# Iterate through the variables
for variable in variables:
    # Initialize nested dictionaries for the variable
    D[variable] = {
        'position': {'axis': {'X': [], 'Y': [], 'Z': []}, 'unit': 'mm'},
        'velocity': {'axis': {"X'": [], "Y'": [], "Z'": []}, 'unit': 'mm/s'},
        'acceleration': {'axis': {'X"': [], 'Y"': [], 'Z"': []}, 'unit': 'mm/s²'}
    }

# Print the dictionary D and its contents in a readable format
pp = pprint.PrettyPrinter(indent=4)
# Print the updated dictionary D and its contents in a readable format
# pp.pprint(D)


for variable in variables:

    # Search for 'LWJC' in non-NaN columns of row 3 and store all occurrences
    lw_indices = [index for index, content in non_nan_columns_row_3.items() if variable in content]

    if lw_indices:
        print(f"{variable} found in column indices: {lw_indices}")
    else:
        print(f"{variable} not found in row 3")

    categories = ['position', 'velocity', 'acceleration']
    axes = ['X', 'Y', 'Z']
    derivatives = ['', "'", '"']

    for i, category in enumerate(categories):
        for j, axis in enumerate(axes):
            key = axis + derivatives[i]
            D[variable][category]['axis'][key] = df.iloc[4:, lw_indices[i] + j].tolist()

# Print the updated dictionary D and its contents in a more readable format
for variable, data in D.items():
    print(f"Variable: {variable}")
    for category, details in data.items():
        print(f"  {category.capitalize()}:")
        for axis, values in details['axis'].items():
            print(f"    {axis}: {values[:5]}...")  # Print only the first 5 values for brevity
        print(f"    Unit: {details['unit']}")
    print()