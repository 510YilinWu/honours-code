import pprint
import GetCSV  # Import the module

# # Define the Raw CSV file path and search terms
file_path = '/Users/yilinwu/Desktop/honours data/YW Trial 3 copy.csv'
# file_path = '/Users/yilinwu/Desktop/honours data/ModelOutput.csv'
# file_path ='/Users/yilinwu/Desktop/honours data/YW-20250221/YW Trial 6.csv'

GetCSV.main(file_path)# Run the main function from GetCSV

import LoadExtractedData  # Import the module

# # Define the extracted python file path and search terms
extracted_data_path = "/Users/yilinwu/Desktop/honours data/Extracted data/Yilin-YW Trial 3 copy.py"
# extracted_data_path = "/Users/yilinwu/Desktop/honours data/Extracted data/Yilin-YW Trial 6.py"
# extracted_data_path = "/Users/yilinwu/Desktop/honours data/Extracted data/Yilin-ModelOutput.py"

extractedData=LoadExtractedData.main(extracted_data_path)# Run the main function from LoadExtractedData

# Print the extractedData dictionary without 'variables' values
for component, data in extractedData.items():
    print(f"{component}:")
    for key, value in data.items():
        if key != 'variables':
            print(f"  {key}: {value}")
        else:
            print("  variables:")
            for var, var_data in value.items():
                print(f"    {var}:")
                for axis, axis_data in var_data.items():
                    print(f"      {axis}:")