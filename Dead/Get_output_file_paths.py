import GetCSV

def main(R_file_paths):
    for i in range(len(R_file_paths)):
        R_file_path = R_file_paths[i]
        GetCSV.main(R_file_path)  # Run the main function from GetCSV