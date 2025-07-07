import os
import pandas as pd

def find_bbt_files(Traj_folder, date, file_type):
    """
    Find all files containing the specified file type ('tBBT', 'sBBT', or 'iBBT') in their name within the specified date folder and sort them by name.

    Args:
        date (str): The date in the format 'month/day' (e.g., '6/13').
        file_type (str): The file type to filter by ('tBBT', 'sBBT', 'iBBT').

    Returns:
        list: A sorted list of full file paths containing the specified file type.
    """
    target_folder = os.path.join(Traj_folder, date.replace("/", os.sep))
    
    print(f"Searching in folder: {target_folder}")
    if not os.path.exists(target_folder):
        raise FileNotFoundError(f"The folder for date {date} does not exist.")
    
    bbt_files = [os.path.join(target_folder, f) for f in os.listdir(target_folder) if file_type in f]
    
    return sorted(bbt_files, key=lambda x: os.path.basename(x))

def CSV_To_traj_data(file_path):
    """
    Reads a CSV file containing trajectory data, processes it, and extracts specific columns based on predefined indices.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        dict: A dictionary containing extracted data for each prefix.
    """
    # Read the file, skipping the first 5 rows
    df = pd.read_csv(
        file_path,
        skiprows=4,
        sep=r"\s+|,",  # Split on whitespace or commas
        engine="python"
    )

    # Extract Frame from column 1 (second column)
    Frame = df.iloc[:, 0]

    # Calculate the time for each frame based on a 200 Hz frame capture rate
    time = Frame / 200  # Time in seconds

    # Define a function to extract X, Y, Z, VX, VY, VZ, AX, AY, AZ, MX, MVX, MAX for a given prefix and column indices
    def extract_columns(prefix, indices):
        return {
            f"{prefix}_X": df.iloc[:, indices[0]],
            f"{prefix}_Y": df.iloc[:, indices[1]],
            f"{prefix}_Z": df.iloc[:, indices[2]],
            f"{prefix}_VX": df.iloc[:, indices[3]],
            f"{prefix}_VY": df.iloc[:, indices[4]],
            f"{prefix}_VZ": df.iloc[:, indices[5]],
            f"{prefix}_AX": df.iloc[:, indices[6]],
            f"{prefix}_AY": df.iloc[:, indices[7]],
            f"{prefix}_AZ": df.iloc[:, indices[8]],
            f"{prefix}_radial_pos": df.iloc[:, indices[9]], # radial position; the distance from the origin 
            f"{prefix}_radial_vel": df.iloc[:, indices[10]], # radial velocity; How fast is the distance from the origin changing over time
            f"{prefix}_radial_acc": df.iloc[:, indices[11]],# radial acceleration
        }

    # Define the column indices for each prefix
    column_indices = {
        "C7": [2, 3, 4, 74, 75, 76, 146, 147, 148, 218, 242, 266],
        "T10": [5, 6, 7, 77, 78, 79, 149, 150, 151, 219, 243, 267],
        "CLAV": [8, 9, 10, 80, 81, 82, 152, 153, 154, 220, 244, 268],
        "STRN": [11, 12, 13, 83, 84, 85, 155, 156, 157, 221, 245, 269],
        "LSHO": [14, 15, 16, 86, 87, 88, 158, 159, 160, 222, 246, 270],
        "LUPA": [17, 18, 19, 89, 90, 91, 161, 162, 163, 223, 247, 271],
        "LUPB": [20, 21, 22, 92, 93, 94, 164, 165, 166, 224, 248, 272],
        "LUPC": [23, 24, 25, 95, 96, 97, 167, 168, 169, 225, 249, 273],
        "LELB": [26, 27, 28, 98, 99, 100, 170, 171, 172, 226, 250, 274],
        "LMEP": [29, 30, 31, 101, 102, 103, 173, 174, 175, 227, 251, 275],
        "LWRA": [32, 33, 34, 104, 105, 106, 176, 177, 178, 228, 252, 276],
        "LWRB": [35, 36, 37, 107, 108, 109, 179, 180, 181, 229, 253, 277],
        "LFRA": [38, 39, 40, 110, 111, 112, 182, 183, 184, 230, 254, 278],
        "LFIN": [41, 42, 43, 113, 114, 115, 185, 186, 187, 231, 255, 279],
        "RSHO": [44, 45, 46, 116, 117, 118, 188, 189, 190, 232, 256, 280],
        "RUPA": [47, 48, 49, 119, 120, 121, 191, 192, 193, 233, 257, 281],
        "RUPB": [50, 51, 52, 122, 123, 124, 194, 195, 196, 234, 258, 282],
        "RUPC": [53, 54, 55, 125, 126, 127, 197, 198, 199, 235, 259, 283],
        "RELB": [56, 57, 58, 128, 129, 130, 200, 201, 202, 236, 260, 284],
        "RMEP": [59, 60, 61, 131, 132, 133, 203, 204, 205, 237, 261, 285],
        "RWRA": [62, 63, 64, 134, 135, 136, 206, 207, 208, 238, 262, 286],
        "RWRB": [65, 66, 67, 137, 138, 139, 209, 210, 211, 239, 263, 287],
        "RFRA": [68, 69, 70, 140, 141, 142, 212, 213, 214, 240, 264, 288],
        "RFIN": [71, 72, 73, 143, 144, 145, 215, 216, 217, 241, 265, 289],
    }

    # Extract the traj_data for each prefix
    traj_data = {}
    for prefix, indices in column_indices.items():
        traj_data.update(extract_columns(prefix, indices))

    return traj_data, Frame, time

def Traj_Space_data(traj_data):

    """
    Processes trajectory data to calculate position, speed, acceleration, and jerk for each marker.

    Parameters:
        traj_data (dict): A dictionary containing trajectory data with keys for position (X, Y, Z), 
                            velocity (VX, VY, VZ), and acceleration (AX, AY, AZ) for each marker.

    Returns:
        dict: A dictionary where each key is a marker name, and the value is a tuple containing:
            - Position (pd.Series): The calculated position in space using the X, Y, Z coordinates.
            - Speed (pd.Series): The calculated speed in space using the VX, VY, VZ components.
            - Acceleration (pd.Series): The calculated acceleration in space using the AX, AY, AZ components.
            - Jerk (pd.Series): The calculated change in acceleration over time (jerk).
    """
    Traj_Space_data = {}

    # Iterate through each marker in the trajectory data
    for marker in set(k.split('_')[0] for k in traj_data.keys()):
        # Calculate the position in space using X, Y, Z
        Position = (traj_data[f"{marker}_X"]**2 + traj_data[f"{marker}_Y"]**2 + traj_data[f"{marker}_Z"]**2)**0.5

        # Calculate the speed in space using VX, VY, VZ
        Speed = (traj_data[f"{marker}_VX"]**2 + traj_data[f"{marker}_VY"]**2 + traj_data[f"{marker}_VZ"]**2)**0.5

        # Calculate the acceleration in space using AX, AY, AZ
        Acceleration = (traj_data[f"{marker}_AX"]**2 + traj_data[f"{marker}_AY"]**2 + traj_data[f"{marker}_AZ"]**2)**0.5

        # Calculate the change in acceleration over time (jerk)
        Jerk = Acceleration.diff()

        # Store the results for the current marker
        Traj_Space_data[marker] = (Position, Speed, Acceleration, Jerk)

    return Traj_Space_data

