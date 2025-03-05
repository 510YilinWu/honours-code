import os
import glob
import pprint
import GetCSV
import LoadExtractedData

# Define the folder path and search terms
folder_path = '/Users/yilinwu/Desktop/honours data/YW-20250221'
search_terms = ['Model Outputs']
# 'Joints', 'Model Outputs', 'Segments', 'Trajectories'

# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))


print(csv_files)



import pprint

import GetCSV  # Import the module

# Define the Raw CSV file path and search terms
# file_path = '/Users/yilinwu/Desktop/honours data/YW Trial 3 copy.csv'
# search_terms = ['Joints', 'Model Outputs', 'Segments', 'Trajectories']
search_terms = ['Model Outputs']
file_path = csv_files[0]

GetCSV.main(file_path, search_terms)# Run the main function from GetCSV

# import LoadExtractedData  # Import the module
# # Define the extracted python file path and search terms
# # extracted_data_path = "/Users/yilinwu/Desktop/honours data/Extracted data/Yilin-YW Trial 3 copy.py"
# extracted_data_path = "/Users/yilinwu/Desktop/honours data/Extracted data/Yilin-ModelOutput.py"
# extractedData=LoadExtractedData.main(extracted_data_path)# Run the main function from LoadExtractedData

# # Print the extractedData dictionary without 'variables' values
# for component, data in extractedData.items():
#     print(f"{component}:")
#     for key, value in data.items():
#         if key != 'variables':
#             print(f"  {key}: {value}")
#         else:
#             print("  variables:")
#             for var, var_data in value.items():
#                 print(f"    {var}:")
#                 for axis, axis_data in var_data.items():
#                     print(f"      {axis}:")