import pprint
import os

# Import the module
import GetFilePath
import GetCSV  
import extractedDataStructure
import LoadExtractedData
import Get_output_file_paths

folder_path = '/Users/yilinwu/Desktop/honours data/YW20250318'
file_paths = GetFilePath.get_file_paths(folder_path)
print(len(file_paths))

output_file_paths = Get_output_file_paths.main(file_paths)
print(len(output_file_paths))

# Define the extracted python file path and search terms
extractedData=LoadExtractedData.main(output_file_paths[0])# Run the main function from LoadExtractedData
extractedDataStructure.main(extractedData)

# # Print the extractedData dictionary without 'variables' values
# extractedDataStructure.main(extractedData)