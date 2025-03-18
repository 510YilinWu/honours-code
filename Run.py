import pprint
import os

# Import the module
import GetFilePath
import GetCSV  
import extractedDataStructure
import LoadExtractedData

folder_path = '/Users/yilinwu/Desktop/honours data/YW20250318'
file_paths = GetFilePath.get_file_paths(folder_path)
print(len(file_paths))

# Define the Raw CSV file path and search terms
file_path =file_paths[0]
output_file_path = GetCSV.main(file_path)# Run the main function from GetCSV


# Define the extracted python file path and search terms
extractedData=LoadExtractedData.main(output_file_path)# Run the main function from LoadExtractedData

# Print the extractedData dictionary without 'variables' values
extractedDataStructure.main(extractedData)