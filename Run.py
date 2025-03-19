# Import the module
import extractedDataStructure
import LoadExtractedData
import Get_output_file_paths
import GetFilePath

# get a list of file paths from the extracted data folder
R_folder_path = '/Users/yilinwu/Desktop/honours data/YW20250318'
R_file_paths = GetFilePath.main(R_folder_path)

# save extracted data to .py files in folder 'Extracted data'
Get_output_file_paths.main(R_file_paths)
print("data saved")

# get a list of file paths from the extracted data folder
E_folder_path = '/Users/yilinwu/Desktop/honours data/Extracted data/YW20250318'
E_file_paths = GetFilePath.main(E_folder_path)

# Loop through E_file_paths and save the data into D based on the given input que
D = {}

for E_file_path in E_file_paths:
    # Define the extracted python file path and search terms
    extractedData = LoadExtractedData.main(E_file_path)  # Run the main function from LoadExtractedData
    # Extract the last part of the file path and split it to get the subject code and trial type
    file_name = E_file_path.split('/')[-1].replace('.py', '')  # Remove the '.py' extension
    
    # Create the dictionary using file_name as the key
    D[file_name] = {
        "extractedData": extractedData,
        "E_file_path": E_file_path,
    }

# extractedDataStructure.main(extractedData)

# # Print the extractedData dictionary without 'variables' values
# extractedDataStructure.main(extractedData)