import os

def main(R_folder_path):
    """
    Extracts file paths from the specified folder and returns them as a list.

    :param R_folder_path: The path to the folder from which to extract file paths.
    :return: A list of file paths.
    """
    R_file_paths = []
    for root, _, files in os.walk(R_folder_path):
        for file in files:
            R_file_paths.append(os.path.join(root, file))
    return R_file_paths