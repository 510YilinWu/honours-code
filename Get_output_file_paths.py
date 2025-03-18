import GetCSV

def main(file_paths):
    output_file_paths = []
    for i in range(len(file_paths)):
        file_path = file_paths[i]
        output_file_path = GetCSV.main(file_path)  # Run the main function from GetCSV
        output_file_paths.append(output_file_path)
    return output_file_paths

