import os

def main(folder_path):
    """
    Extracts file paths from the specified folder and returns them as a list.

    :param folder_path: The path to the folder from which to extract file paths.
    :return: A list of file paths.
    """
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths