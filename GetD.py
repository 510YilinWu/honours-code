
E_file_path = E_file_paths[0]
# Define the extracted python file path and search terms
extractedData = LoadExtractedData.main(E_file_path)  # Run the main function from LoadExtractedData
# Extract the last part of the file path and split it to get the subject code and trial type
file_name = E_file_path.split('/')[-1].replace('.py', '')  # Remove the '.py' extension
# Create the dictionary using file_name as the key
D = {
    file_name: {
        "extractedData": extractedData,
        "output_file_path": E_file_paths[0]
    }
}