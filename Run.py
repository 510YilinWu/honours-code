# Import the module
import extractedDataStructure
import LoadExtractedData
import Get_output_file_paths
import GetFilePath

# get a list of file paths from the extracted data folder
R_folder_path = '/Users/yilinwu/Desktop/honours data/YW20250318'
R_file_paths = GetFilePath.main(R_folder_path)
print(len(R_file_paths))

# save extracted data to .py files in folder 'Extracted data'
Get_output_file_paths.main(R_file_paths)
print("data saved")

# get a list of file paths from the extracted data folder
E_folder_path = '/Users/yilinwu/Desktop/honours data/Extracted data/YW20250318'
E_file_paths = GetFilePath.main(E_folder_path)


# Initialize the dictionary to hold the output data
D = {}
# Loop through E_file_paths and save the data into D based on the given input
for file_path in E_file_paths:
    # Skip files in __pycache__ folders
    if "__pycache__" in file_path:
        continue
    
    # Define the extracted data by calling the LoadExtractedData main function
    extractedData = LoadExtractedData.main(file_path)  # Run the main function from LoadExtractedData
    
    # Extract the last part of the file path and split it to get the subject code and trial type
    file_name = file_path.split('/')[-1].replace('.py', '')  # Remove the '.py' extension

    # Add the new entry to the dictionary, preserving the previous ones
    D[file_name] = {
        "extractedData": extractedData,
        "output_file_path": file_path
    }


extractedDataStructure.main(extractedData)

# # # Print the extractedData dictionary without 'variables' values
# # extractedDataStructure.main(extractedData)