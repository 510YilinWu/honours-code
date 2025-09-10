# rename all .png files in a directory. 
# Read them in alphabetical order and rename as 
import os
import glob
import bbtLocalisation.charucoStereoCalib
import bbtLocalisation.helper.renamePNG
import bbtLocalisation.helper
import re
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pickle
import pandas as pd




# for subject 6  onward
def rename_files(directory):
    # Get all .png files in the directory    
    files = sorted(
        glob.glob(os.path.join(directory, '*.png')),
        key=lambda x: (
            int(os.path.basename(x).split('_')[2].split('.')[0]),
            os.path.basename(x).split('_')[1]
        )
    )

    # Initialize counters for left and right images
    counters = {}

    for file in files:
        # Extract the camera and number from the filename
        parts = os.path.basename(file).split('_') 
        subject = parts[0]               # 'AAB'
        camera = parts[1]                    # 'cam0'
        number = int(parts[2].split('.')[0]) # '06.png' → 6
        index = files.index(file)

        # Determine if the image is for the right or left hand based on the index
        current_hand = 'right' if (index // 2) % 2 == 0 else 'left'

        # Initialize counter for this camera if not already done
        if camera not in counters:
            counters[camera] = {'left': 1, 'right': 1}

        # Generate the new name
        new_name = f"{current_hand}_{subject}_{camera}_{counters[camera][current_hand]:02d}.png"
        counters[camera][current_hand] += 1
        # Initialize counter for this camera if not already done
        # if camera not in counters:
        #     counters[camera] = 1

        # # Generate the new name
        # new_name = f"{current_hand}_{subject}_{camera}_{counters[camera]:02d}.png"
        # counters[camera] += 1


        # Construct full path for the new file name
        new_file_path = os.path.join(directory, new_name)

        # Rename the file
        os.rename(file, new_file_path)
        print(f'Renamed "{file}" to "{new_file_path}"')

# for first 5 subjects, the files named differ.
# def rename_files(directory):
#     # Ask for the subject name as input
#     subject = input("Enter the subject name: ")

#     # Get all .png files in the directory    
#     files = sorted(
#         glob.glob(os.path.join(directory, '*.png')),
#         key=lambda x: (
#             int(os.path.basename(x).split('_')[1].split('.')[0]),
#             os.path.basename(x).split('_')[0]
#         )
#     )

#     # Initialize counters for left and right images
#     counters = {}

#     for file in files:
#         # Extract the camera and number from the filename
#         parts = os.path.basename(file).split('_') 
#         camera = parts[0]                    # 'cam0'
#         number = int(parts[1].split('.')[0]) # '06.png' → 6
#         index = files.index(file)

#         # Determine if the image is for the right or left hand based on the index
#         current_hand = 'right' if (index // 2) % 2 == 0 else 'left'
#         # current_hand = 'right' if (number // 2) % 2 == 0 else 'left'
#         # current_hand = 'right' if (index // 4) % 2 == 0 else 'left'


#         # Initialize counter for this camera if not already done
#         if camera not in counters:
#             counters[camera] = {'left': 1, 'right': 1}

#         # Generate the new name
#         new_name = f"{current_hand}_{subject}_{camera}_{counters[camera][current_hand]:02d}.png"
#         counters[camera][current_hand] += 1

#         # Construct full path for the new file name
#         new_file_path = os.path.join(directory, new_name)

#         # Rename the file
#         os.rename(file, new_file_path)
#         print(f'Renamed "{file}" to "{new_file_path}"')

# def rename_files_by_count(directory):
#     # Get all .png files in the directory
#     files = sorted(
#         glob.glob(os.path.join(directory, '*.png')),
#         key=lambda x: (
#             os.path.basename(x).split('_')[0],  # left or right
#             os.path.basename(x).split('_')[1],  # subject
#             os.path.basename(x).split('_')[2],  # camera
#             int(os.path.basename(x).split('_')[3].split('.')[0])  # number
#         )
#     )

#     # Initialize counters for each combination of hand, subject, and camera
#     counters = {}

#     for file in files:
#         # Extract components from the filename
#         parts = os.path.basename(file).split('_')
#         hand = parts[0]  # left or right
#         subject = parts[1]  # subject
#         camera = parts[2]  # cam0 or cam1

#         # Initialize counter for this combination if not already done
#         key = (hand, subject, camera)
#         if key not in counters:
#             counters[key] = 1

#         # Generate the new name
#         new_name = f"{hand}_{subject}_{camera}_{counters[key]:02d}.png"
#         counters[key] += 1

#         # Construct full path for the new file name
#         new_file_path = os.path.join(directory, new_name)

#         # Rename the file
#         os.rename(file, new_file_path)
#         print(f'Renamed "{file}" to "{new_file_path}"')

# --- FIND BEST CALIBRATION IMAGES COMBINATION FOR EACH SUBJECT ---
def run_test_for_each_subject(subjects, tBBT_Image_folder):
    for subject in subjects:
        # Extract date and subject code from the subject string
        date, subject_code = subject.rsplit('/', 1)
        
        # Construct folder paths and test files
        calib_folder = f"{tBBT_Image_folder}{date}/Cali"  # Calibration folder for the subject
        testFolder = f"{tBBT_Image_folder}{subject}"  # Test folder for the subject
        testFiles = [f"left_{subject_code}_cam0_01.png", f"left_{subject_code}_cam1_01.png"]  # Test files for the subject

        # Sort the files to ensure consistent order
        testFiles = sorted(testFiles)

        print(f"Results for subject {subject}:")
        # Run the charucoStereoCalib main function
        p3_box2, p3_block2, bg, markerindex = bbtLocalisation.charucoStereoCalib.main(calib_folder, testFolder, testFiles)

        # Print results for the subject
        print("-" * 50)

# --- CHECK CALIBRATION FOLDERS FOR PICKLE FILES ---
def check_calibration_folders_for_pickle(all_dates, tBBT_Image_folder):
    missing_pickle_files = []
    for date in all_dates:
        cali_folder = f"{tBBT_Image_folder}{date.split('/')[0]}/{date.split('/')[1]}/Cali"
        pickle_files = [file for file in os.listdir(cali_folder) if file.endswith('.pckl')]
        if len(pickle_files) != 1:
            missing_pickle_files.append(cali_folder)
    
    if missing_pickle_files:
        print("The following calibration folders are missing a single pickle file or have multiple:")
        for folder in missing_pickle_files:
            print(folder)
    else:
        pass
        # print("All calibration folders have exactly one pickle file.")

# def process_blocks(p3_block2, bg, hand):
#     if len(p3_block2) < 16 or len(p3_block2) > 16:
#         print(f"p3_block2: {len(p3_block2)}")
#     else:
#         pass
#     if hand == 'right': # right hand
#         x = bg.grid_xxL.flatten()
#         y = bg.grid_yyL.flatten()       

#     else: # left hand
#         x = bg.grid_xxR.flatten()
#         y = bg.grid_yyR.flatten()


#     blockRange = [{'xRange': [x[i] - 20, x[i] + 20],
#                    'yRange': [y[i] - 20, y[i] + 20]} for i in range(16)]

#     blockMembership = [
#         next((idx for idx, block in enumerate(blockRange)
#               if block['xRange'][0] <= point[0] <= block['xRange'][1] and
#                  block['yRange'][0] <= point[1] <= block['yRange'][1]), None)
#         for point in p3_block2
#     ]

#     blocks_without_points = [idx for idx, block in enumerate(blockRange) if not any(
#         block['xRange'][0] <= point[0] <= block['xRange'][1] and
#         block['yRange'][0] <= point[1] <= block['yRange'][1] for point in p3_block2)]
    
#     if len(blocks_without_points) == len([i for i, membership in enumerate(blockMembership) if membership is None]):
#         for i, membership in enumerate(blockMembership):
#             if membership is None:
#                 # Allocate the point to the nearest block based on distance
#                 distances_to_blocks = [
#                     ((p3_block2[i][0] - (block['xRange'][0] + block['xRange'][1]) / 2) ** 2 +
#                      (p3_block2[i][1] - (block['yRange'][0] + block['yRange'][1]) / 2) ** 2) ** 0.5
#                     for block in blockRange
#                 ]
#                 nearest_block = distances_to_blocks.index(min(distances_to_blocks))
#                 blockMembership[i] = nearest_block
#                 print(f"Point {i} allocated to block {nearest_block}")

#                 # Update blocks_without_points if allocation occurs
#                 if nearest_block in blocks_without_points:
#                     blocks_without_points.remove(nearest_block)



#     for i, membership in enumerate(blockMembership):
#         if membership is not None:
#             pass
#         else:

#             # raise Exception(f"Error: Point {i} does not belong to any block")
#             print(f"Error: Point {i} does not belong to any block")

#     if blocks_without_points:
#         print(f"Blocks without points: {blocks_without_points}")
#     else:
#         pass

#     # # Create a list of length 16, marking blocks without points as 'nah'
#     # block_status = ['nah' if idx in blocks_without_points else idx for idx in range(16)]
#     # print(f"Block status: {block_status}")

#     # Calculate the error for block placement, including distance
#     block_errors = [
#         {
#             'point': p3_block2[i],
#             'membership': blockMembership[i],
#             # 'x': bg.grid_xxR.flatten()[blockMembership[i]] - 12.5 if blockMembership[i] is not None else None,
#             # 'y': bg.grid_yyR.flatten()[blockMembership[i]] - 12.5 if blockMembership[i] is not None else None,
#             'distance': (
#                 ((p3_block2[i][0] - (x[blockMembership[i]] - 12.5))**2 +
#                  (p3_block2[i][1] - (y[blockMembership[i]] - 12.5))**2)**0.5
#             ) if blockMembership[i] is not None else None
#         }
#         for i in range(len(p3_block2))
#     ]

#     # Add membership for blocks without points and keep point value as 'nah'
#     for block_idx in blocks_without_points:
#         blockMembership.append(None)  # Append None for blocks without points
#         block_errors.append({
#             'membership': block_idx,
#         })

    

#     return block_errors

def process_blocks(p3_block2, bg, hand, markerindex):

    if len(markerindex) < 16:
        print(f"\u26A0\uFE0F Only {len(markerindex)}, expected 16.")
    elif len(markerindex) > 16:
        print(f"\u26A0\uFE0F {len(markerindex)}, expected 16.")

    if hand == 'right':  # right hand
        x = bg.grid_xxL.flatten()
        x = x[::-1]
        y = bg.grid_yyL.flatten()
    else:  # left hand
        x = bg.grid_xxR.flatten()
        y = bg.grid_yyR.flatten()

    # Predefined block membership order
    blockMembership = [12, 1, 14, 3, 8, 13, 2, 15, 0, 5, 6, 11, 4, 9, 7, 10]

    # Initialize block errors and blocks without points
    block_errors = []
    blocks_without_points = []

    # Process detected markers
    for i in range(len(markerindex)):
        marker_idx = markerindex[i]
        point = p3_block2[i]
        membership = blockMembership[marker_idx]

        # Calculate distance for the marker
        distance = (
            ((point[0] + 12.5 - (x[membership]))**2 +
                (point[1] + 12.5 - (y[membership]))**2)**0.5
        )

        block_errors.append({
            'point': point,
            'membership': membership,
            'distance': distance
        })

    # Check for blocks without points
    for idx in range(16):
        if idx not in markerindex:
            blocks_without_points.append(blockMembership[idx])

    # print(f"Blocks without points: {sorted(blocks_without_points)}")

    return block_errors, sorted(blocks_without_points)

def find_min_max_number_in_folder(folder_path, hand):
    # Regular expression to match the image format for the specified hand
    pattern = re.compile(rf'{hand}_(\w+)_(cam[01])_(\d+)\.png')  # Allow numbers without leading zeros

    min_number = float('inf')
    max_number = -1

    # Iterate through files in the folder
    for file_name in os.listdir(folder_path):
        # Ignore files starting with "._"
        if file_name.startswith("._"):
            continue
        
        match = pattern.match(file_name)
        if match:
            number = int(match.group(3))  # Corrected to group(3) for the number

            min_number = min(min_number, number)
            max_number = max(max_number, number)

    if max_number == -1 or min_number == float('inf'):
        raise FileNotFoundError(f"No matching images found in the folder for hand: {hand}.")
    
    return min_number, max_number

# for per participant
def process_subject_images(Date, Subject, tBBT_Image_folder, hand):

    calib_folder = f"{tBBT_Image_folder}{Date}/Cali"
    testFolder = f"{tBBT_Image_folder}{Subject}"

    # Find the maximum number in the test folder
    min_number, max_number = find_min_max_number_in_folder(testFolder, hand)

    if max_number < 0:
        raise ValueError("No images found in the specified folder.")
    
    Subject_tBBTs_errors = {}
    print(f"Processing subject: {Subject}, hand: {hand}, min_number: {min_number}, max_number: {max_number}")
    for image_number in range(1, max_number + 1):
        if image_number < 10:
            image_number_str = f"0{image_number}"
        else:
            image_number_str = str(image_number)

        
        # # Update testFiles with the images corresponding to the maximum number
        # testFiles = [f"cam0_{image_number_str}.png", f"cam1_{image_number_str}.png"]

        # Find files that contain the current image_number_str and match the specified hand
        testFiles = [file_name for file_name in os.listdir(testFolder) 
            #  if re.search(rf'{hand}_cam[01]_(0*{image_number_str})\.png$', file_name) and not file_name.startswith("._")]
            #  if re.search(rf'{hand}_(\w+)_(cam[01])_(0*{image_number_str})\.png$', file_name) and not file_name.startswith("._")]
             if re.search(rf'{hand}_(\w+)_(cam[01])_{image_number_str}\.png$', file_name) and not file_name.startswith("._")]
        # print(f"Processing image number: {image_number_str}, testFiles: {testFiles}")

        if len(testFiles) != 2:
            raise FileNotFoundError(f"Expected 2 files for image number {image_number_str}, but found {len(testFiles)}: {testFiles}")
        
        testFiles = sorted(testFiles)  # Sort the files to ensure consistent order
        p3_box2, p3_block2, bg, markerindex = bbtLocalisation.charucoStereoCalib.main(calib_folder, testFolder, testFiles) # YW - MAC

        # # Determine the hand for the current image
        # index_since_min = image_number - min_number
        # if (index_since_min // 2) % 2 == 0:
        #     current_hand = 'right'
        # else:
        #     current_hand = 'left'

        # print(f"Test files: {testFiles}: {current_hand} hand")

        # # Process blocks for the current hand
        # block_errors = process_blocks(p3_block2, bg, current_hand)

    #     print(f"Test files: {testFiles}: {hand} hand")

        # Process blocks for the current hand
        block_errors, blocks_without_points = process_blocks(p3_block2, bg, hand, markerindex)

        # Store the errors in the dictionary
        Subject_tBBTs_errors[image_number] = {
            'p3_box2': p3_box2,
            'p3_block2': p3_block2,
            'bg': bg,
            'block_errors': block_errors,
            'blocks_without_points': blocks_without_points,
        }
    return Subject_tBBTs_errors

# This function extracts ordered distances from the errors for each subject and hand.
def extract_ordered_distances(All_Subject_tBBTs_errors, DataProcess_folder):
    all_ordered_distances = {}
    for (subject, hand), Subject_tBBTs_errors in All_Subject_tBBTs_errors.items():
        if subject not in all_ordered_distances:
            all_ordered_distances[subject] = {}
        if hand not in all_ordered_distances[subject]:
            all_ordered_distances[subject][hand] = {}

        for image_number, data in sorted(Subject_tBBTs_errors.items()):
            block_errors = data.get('block_errors', [])
            distances = [None] * 16  # Initialize with None for missing values
            for error in block_errors:
                membership = error.get('membership')
                distance = error.get('distance')
                if membership is not None and 0 <= membership < 16:
                    distances[membership] = distance
            all_ordered_distances[subject][hand][image_number] = distances
        
        # Save the combined errors for the subject as a pickle file
        file_name = f"{subject.replace('/', '_')}_errors.pkl"
        file_path = os.path.join(DataProcess_folder, file_name)
        with open(file_path, 'wb') as pkl_file:
            pickle.dump(all_ordered_distances[subject], pkl_file)
        print(f"Saved combined errors for subject {subject} to {file_path}")

    # return all_ordered_distances

# Iterate through all dates in All_dates
def process_all_subjects_images(All_dates, tBBT_Image_folder, DataProcess_folder):
    All_Subject_tBBTs_errors = {}
    for hand in ['left', 'right']:  # Process left hand first, then right hand
        for subject in All_dates:
            date = "/".join(subject.split("/")[:2])  # Extract the date from the subject
            try:
                print(f"Processing date: {date}, subject: {subject}, hand: {hand}")
                Subject_tBBTs_errors = process_subject_images(date, subject, tBBT_Image_folder, hand)
                All_Subject_tBBTs_errors[(subject, hand)] = Subject_tBBTs_errors
                print(f"Processing completed successfully for date: {date}, subject: {subject}, hand: {hand}")
            except Exception as e:
                print(f"An error occurred while processing date: {date}, subject: {subject}, hand: {hand}: {e}")

    # # Extract ordered distances and save them
    # extract_ordered_distances(All_Subject_tBBTs_errors, DataProcess_folder)
    
    # output_file_path = os.path.join(DataProcess_folder, "All_Subject_tBBTs_errors.pkl")
    # with open(output_file_path, 'wb') as f:
    #     pickle.dump(All_Subject_tBBTs_errors, f)
    return All_Subject_tBBTs_errors

# This function loads the processed errors for selected subjects and aggregates them into a single dictionary.
def load_selected_subject_errors(All_dates, DataProcess_folder):
    """
    Loads the processed errors for selected subjects and aggregates them into a single dictionary.

    Args:
        All_dates (list): List of dates to load errors for.
        DataProcess_folder (str): Path to the data processing folder.

    Returns:
        dict: A dictionary containing the aggregated errors for all selected subjects.
    """
    errors = {}
    for date in All_dates:
        subject_filename = f"{date.replace('/', '_')}_errors.pkl"
        try:
            file_path = os.path.join(DataProcess_folder, subject_filename)
            with open(file_path, 'rb') as file:
                errors[date] = pickle.load(file)
        except FileNotFoundError:
            print(f"Warning: Errors file for {date} not found. Skipping.")
    return errors

# Calculate RMS reprojection error statistics
def compute_rms_reprojection_error_stats(csv_filepath="/Users/yilinwu/Desktop/Yilin-Honours/RMS re-projection error.csv"):
    RMSReprojectionError = pd.read_csv(csv_filepath)
    # Compute overall statistics for RMSReprojectionError DataFrame
    rms_min = RMSReprojectionError.min().min()    # Overall minimum
    rms_max = RMSReprojectionError.max().max()    # Overall maximum
    rms_mean = RMSReprojectionError.stack().mean()  # Mean of all values
    rms_std = RMSReprojectionError.stack().std()    # Standard deviation of all values

    print("RMS Reprojection Error Statistics:")
    print(f"Minimum: {rms_min}")
    print(f"Maximum: {rms_max}")
    print(f"Mean: {rms_mean}")
    print(f"Standard Deviation: {rms_std}")

    return rms_min, rms_max, rms_mean, rms_std

# -------------------------------------------------------------------------------------------------------------------
## --------------------------------------- PLOTTING FUNCTIONS --------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
### Separate data by hand and plot 3D scatter plots for each hand
# Create 2D projection density heatmaps for all data, one figure for left hand and one for right hand
# You can change the colormap by setting cmap_choice to any valid matplotlib colormap 
# (e.g., 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', etc.)
def plot_xy_density_for_each_hand(All_Subject_tBBTs_errors, cmap_choice):
    for h in ['left', 'right']:
        xs, ys = [], []
        # Loop over all (subject, hand) keys in the errors dictionary
        for key in All_Subject_tBBTs_errors:
            subject_key, hand_key = key
            if hand_key != h:
                continue
            # For each trial in the subject-hand entry, extract the p3_block2 data points
            for trial in All_Subject_tBBTs_errors[key]:
                trial_data = All_Subject_tBBTs_errors[key][trial]
                # Get the 3D points from p3_block2 (if available)
                p3_data = trial_data.get('p3_block2', None)
                if p3_data is None:
                    continue
                p3_data = np.array(p3_data)
                # Ensure data is in (n_points, 3) format
                if p3_data.ndim == 2 and p3_data.shape[0] == 3:
                    p3_data = p3_data.T
                elif p3_data.ndim == 1:
                    p3_data = p3_data.reshape(-1, 3)
                if p3_data.size == 0:
                    continue
                # For left hand, expect x > -50; for right hand, expect x < 0.
                if hand_key == 'left':
                    if np.any(p3_data[:, 0] <= -50):
                        invalid_idx = np.where(p3_data[:, 0] <= -50)[0]
                        print(f"Error: Subject '{subject_key}', hand '{hand_key}', trial '{trial}' has x values <= -50 at indices {invalid_idx} with values {p3_data[invalid_idx, 0]}")
                elif hand_key == 'right':
                    if np.any(p3_data[:, 0] >= 0):
                        invalid_idx = np.where(p3_data[:, 0] >= 0)[0]
                        print(f"Error: Subject '{subject_key}', hand '{hand_key}', trial '{trial}' has non-negative x values at indices {invalid_idx} with values {p3_data[invalid_idx, 0]}")
                xs.extend(p3_data[:, 0])
                ys.extend(p3_data[:, 1])
                
        xs = np.array(xs)
        ys = np.array(ys)
        if xs.size != 0 and ys.size != 0:
            print("X max:", xs.max(), "X min:", xs.min(), "Y max:", ys.max(), "Y min:", ys.min())
        else:
            print(f"No data available for {h} hand.")
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        hb = ax.hist2d(xs, ys, bins=50, cmap=cmap_choice)
        fig.colorbar(hb[3], ax=ax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        # ax.set_title(f'XY Projection Density for {h.capitalize()} Hand')
        plt.tight_layout()
        plt.show()

### Combine data from both hands and plot combined density heatmap with grid markers
def plot_combined_xy_density(All_Subject_tBBTs_errors, cmap_choice):
    """
    Combines x and y data from both hands in the errors dictionary, plots a 2D density heatmap,
    and overlays grid markers as black crosses.
    
    Parameters:
        All_Subject_tBBTs_errors (dict): Dictionary containing error data.
        cmap_choice: A valid matplotlib colormap.
    """
    import matplotlib.pyplot as plt

    combined_xs, combined_ys = [], []

    for h in ['left', 'right']:
        xs, ys = [], []
        # Loop over all (subject, hand) keys in the errors dictionary
        for key in All_Subject_tBBTs_errors:
            subject_key, hand_key = key
            if hand_key != h:
                continue
            # For each trial in the subject-hand entry, extract the p3_block2 data points
            for trial in All_Subject_tBBTs_errors[key]:
                trial_data = All_Subject_tBBTs_errors[key][trial]
                # Get the 3D points from p3_block2 (if available)
                p3_data = trial_data.get('p3_block2', None)
                if p3_data is None:
                    continue
                p3_data = np.array(p3_data)
                # Ensure data is in (n_points, 3) format
                if p3_data.ndim == 2 and p3_data.shape[0] == 3:
                    p3_data = p3_data.T
                elif p3_data.ndim == 1:
                    p3_data = p3_data.reshape(-1, 3)
                if p3_data.size == 0:
                    continue

                # Check for invalid x-values based on hand
                if hand_key == 'left':
                    if np.any(p3_data[:, 0] <= -50):
                        invalid_idx = np.where(p3_data[:, 0] <= -50)[0]
                        print(f"Error: Subject '{subject_key}', hand '{hand_key}', trial '{trial}' has x values <= -50 at indices {invalid_idx} with values {p3_data[invalid_idx, 0]}")
                elif hand_key == 'right':
                    if np.any(p3_data[:, 0] >= 0):
                        invalid_idx = np.where(p3_data[:, 0] >= 0)[0]
                        print(f"Error: Subject '{subject_key}', hand '{hand_key}', trial '{trial}' has non-negative x values at indices {invalid_idx} with values {p3_data[invalid_idx, 0]}")
                xs.extend(p3_data[:, 0])
                ys.extend(p3_data[:, 1])
        
        if len(xs) > 0 and len(ys) > 0:
            combined_xs.extend(xs)
            combined_ys.extend(ys)
        else:
            print(f"No data available for {h} hand.")

    combined_xs = np.array(combined_xs)
    combined_ys = np.array(combined_ys)
    print(len(combined_xs), len(combined_ys))
    print("Combined Data: X max:", np.max(combined_xs), "X min:", np.min(combined_xs),
            "Y max:", np.max(combined_ys), "Y min:", np.min(combined_ys))

    if combined_xs.size == 0 or combined_ys.size == 0:
        print("No combined data available.")
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        hb = ax.hist2d(combined_xs, combined_ys, bins=50, cmap=cmap_choice)
        fig.colorbar(hb[3], ax=ax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("XY Projection Density for Combined Hands")
        
        # Define grid markers to overlay as black crosses
        grid_xxR = np.array([[ 12.5       ,  66.92857143, 121.35714286, 175.78571429],
                                [ 12.5       ,  66.92857143, 121.35714286, 175.78571429],
                                [ 12.5       ,  66.92857143, 121.35714286, 175.78571429],
                                [ 12.5       ,  66.92857143, 121.35714286, 175.78571429]])
    
        grid_yyR = np.array([[ 12.5       ,  12.5       ,  12.5       ,  12.5       ],
                                [ 66.92857143,  66.92857143,  66.92857143,  66.92857143],
                                [121.35714286, 121.35714286, 121.35714286, 121.35714286],
                                [175.78571429, 175.78571429, 175.78571429, 175.78571429]])
    
        grid_xxL = np.array([[-246.5       , -192.07142857, -137.64285714,  -83.21428571],
                                [-246.5       , -192.07142857, -137.64285714,  -83.21428571],
                                [-246.5       , -192.07142857, -137.64285714,  -83.21428571],
                                [-246.5       , -192.07142857, -137.64285714,  -83.21428571]])
    
        grid_yyL = np.array([[ 12.5       ,  12.5       ,  12.5       ,  12.5       ],
                                [ 66.92857143,  66.92857143,  66.92857143,  66.92857143],
                                [121.35714286, 121.35714286, 121.35714286, 121.35714286],
                                [175.78571429, 175.78571429, 175.78571429, 175.78571429]])
    
        # Overlay grid markers as black crosses with all values subtracted by 12.5
        ax.scatter((grid_xxR - 12.5).flatten(), (grid_yyR - 12.5).flatten(),
                    marker='x', color='black', s=50, linewidths=2)
        ax.scatter((grid_xxL - 12.5).flatten(), (grid_yyL - 12.5).flatten(),
                    marker='x', color='black', s=50, linewidths=2)


    
        plt.tight_layout()
        plt.show()

### Combine 16 blocks data into one for each subject and hand
def Combine_16_blocks(All_Subject_tBBTs_errors):
    """
    For each subject and hand in All_Subject_tBBTs_errors, extract the 'p3_block2' data
    and 'blocks_without_points' from trial index 1, and compute new coordinates based on
    blockMembership and grid adjustments.
    
    For left-hand entries, grid values from grid_xxR and grid_yyR are used.
    For right-hand entries, grid values from grid_xxL and grid_yyL are used.
    
    Returns:
        dict: Mapping from (subject, hand) key to a list of tuples (new_x, new_y, block)
    """
    # Block membership mapping (order of blocks)
    blockMembership = [12, 1, 14, 3, 8, 13, 2, 15, 0, 5, 6, 11, 4, 9, 7, 10]

    # Define grid markers for right hand (for left-hand data adjustment)
    grid_xxR = np.array([[ 12.5      ,  66.92857143, 121.35714286, 175.78571429],
                           [ 12.5      ,  66.92857143, 121.35714286, 175.78571429],
                           [ 12.5      ,  66.92857143, 121.35714286, 175.78571429],
                           [ 12.5      ,  66.92857143, 121.35714286, 175.78571429]])
    grid_yyR = np.array([[ 12.5      ,  12.5      ,  12.5      ,  12.5      ],
                           [ 66.92857143,  66.92857143,  66.92857143,  66.92857143],
                           [121.35714286, 121.35714286, 121.35714286, 121.35714286],
                           [175.78571429, 175.78571429, 175.78571429, 175.78571429]])
    
    # Define grid markers for left hand (for right-hand data adjustment)
    grid_xxL = np.array([[-246.5      , -192.07142857, -137.64285714,  -83.21428571],
                           [-246.5      , -192.07142857, -137.64285714,  -83.21428571],
                           [-246.5      , -192.07142857, -137.64285714,  -83.21428571],
                           [-246.5      , -192.07142857, -137.64285714,  -83.21428571]])
    
    grid_yyL = np.array([[ 12.5      ,  12.5      ,  12.5      ,  12.5      ],
                           [ 66.92857143,  66.92857143,  66.92857143,  66.92857143],
                           [121.35714286, 121.35714286, 121.35714286, 121.35714286],
                           [175.78571429, 175.78571429, 175.78571429, 175.78571429]])
    
    results = {}
    
    # Iterate over every (subject, hand) key in the dictionary
    for key in All_Subject_tBBTs_errors:
        subject, hand = key
        # Use trial index 1 as in the snippet
        try:
            trial_entry = All_Subject_tBBTs_errors[key][1]
        except (KeyError, IndexError):
            continue  # Skip keys without trial 1
        
        # Get p3_block2 data and blocks_without_points
        p3_data = trial_entry.get('p3_block2', None)
        blocks_without_points = trial_entry.get('blocks_without_points', None)
        if p3_data is None or blocks_without_points is None:
            continue

        # Ensure p3_data is a numpy array and in shape (n_points, 3)
        p3_data = np.array(p3_data)
        if p3_data.ndim == 2 and p3_data.shape[0] == 3:
            p3_data = p3_data.T
        elif p3_data.ndim == 1:
            p3_data = p3_data.reshape(-1, 3)
        if p3_data.size == 0:
            continue
        
        # Choose grid arrays based on hand: 'left' uses grid_xxR/yyR, 'right' uses grid_xxL/yyL
        if hand.lower() == 'left':
            grid_x = (grid_xxR - 12.5).flatten()
            grid_y = (grid_yyR - 12.5).flatten()
        elif hand.lower() == 'right':
            grid_x = (grid_xxL - 12.5).flatten()
            grid_x = grid_x[::-1]
            grid_y = (grid_yyL - 12.5).flatten()
        else:
            continue

        data_index = 0
        new_coords = []  # List to store the new coordinates for this entry
        
        # Loop over each block position (total 16 blocks)
        for i in range(16):
            current_block = blockMembership[i]
            # If this block was not marked as missing
            if current_block not in blocks_without_points:
                # Get block_points from the p3_data
                block_points = p3_data[data_index]
                data_index += 1
                # Adjust the coordinates: add 12.5 offset and subtract grid offset for the current block
                new_x = block_points[0] - grid_x.flatten()[current_block]
                new_y = block_points[1] - grid_y.flatten()[current_block]
                new_coords.append((new_x, new_y, blockMembership[i]))
        results[key] = new_coords

    return results

### Plot left and right hand 16 blocks as one density histograms with 0.0 at the center of the view
def plot_left_right_hand_new_coordinates_density(Combine_blocks, cmap=None, bins=20, xlim=(-15, 15), ylim=(-15, 15)):
    """
    Plots 2D density histograms (hist2d) for new coordinates of left-hand and right-hand data
    as subplots, keeping 0.0 at the center of the view.

    Parameters:
        Combine_blocks (dict): Dictionary mapping (subject, hand) to a list of tuples (new_x, new_y, block).
        cmap: A matplotlib colormap. If None, defaults to a white-to-red colormap.
        bins (int): Number of bins for the histogram.
        xlim (tuple): x-axis limits (centered at 0).
        ylim (tuple): y-axis limits (centered at 0).
    """
    import matplotlib.pyplot as plt

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("WhiteRed", ["white", "red"], N=256)

    xs_left, ys_left = [], []
    xs_right, ys_right = [], []

    for key, coords in Combine_blocks.items():
        subject, hand = key
        if hand.lower() == 'left':
            for new_x, new_y, block in coords:
                xs_left.append(new_x)
                ys_left.append(new_y)
        elif hand.lower() == 'right':
            for new_x, new_y, block in coords:
                xs_right.append(new_x)
                ys_right.append(new_y)

    xs_left = np.array(xs_left)
    ys_left = np.array(ys_left)
    xs_right = np.array(xs_right)
    ys_right = np.array(ys_right)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    hb1 = ax1.hist2d(xs_right, ys_right, bins=bins, cmap=cmap)
    ax1.plot(0, 0, 'ko', markersize=8)
    # ax1.set_xlim(xlim)
    # ax1.set_ylim(ylim)
    ax1.grid(False)
    fig.colorbar(hb1[3], ax=ax1)
    ax1.set_xlabel("New X")
    ax1.set_ylabel("New Y")
    ax1.set_title("2D Density of All right-Hand New Coordinates")

    hb2 = ax2.hist2d(xs_left, ys_left, bins=bins, cmap=cmap)
    ax2.plot(0, 0, 'ko', markersize=8)
    # ax2.set_xlim(xlim)
    # ax2.set_ylim(ylim)
    ax2.grid(False)
    fig.colorbar(hb2[3], ax=ax2)
    ax2.set_xlabel("New X")
    ax2.set_ylabel("New Y")
    ax2.set_title("2D Density of All Left-Hand New Coordinates")


    plt.tight_layout()
    plt.show()

### Plot left and right hand 16 blocks as one polar histograms (rose diagrams)
def plot_left_right_hand_polar_histogram(Combine_blocks, cmap_choice):
    """
    Plots polar histograms (rose diagrams) for left-hand and right-hand new coordinates as subplots.
    
    Parameters:
        Combine_blocks (dict): Dictionary mapping keys (subject, hand) to lists of tuples (new_x, new_y, block).
        cmap_choice: A matplotlib colormap to map histogram density.
    """
    import matplotlib.pyplot as plt

    # Collect coordinates for left and right hands
    xs_left, ys_left = [], []
    xs_right, ys_right = [], []
    for key, coords in Combine_blocks.items():
        subject, hand = key
        hand_lower = hand.lower()
        for new_x, new_y, block in coords:
            if hand_lower == 'left':
                xs_left.append(new_x)
                ys_left.append(new_y)
            elif hand_lower == 'right':
                xs_right.append(new_x)
                ys_right.append(new_y)

    xs_left = np.array(xs_left)
    ys_left = np.array(ys_left)
    xs_right = np.array(xs_right)
    ys_right = np.array(ys_right)

    # Convert Cartesian to polar coordinates
    r_left = np.sqrt(xs_left**2 + ys_left**2)
    theta_left = np.arctan2(ys_left, xs_left)

    r_right = np.sqrt(xs_right**2 + ys_right**2)
    theta_right = np.arctan2(ys_right, xs_right)

    num_bins = 20

    # Create subplots for left and right hands
    fig, axes = plt.subplots(1, 2, subplot_kw=dict(projection='polar'), figsize=(12, 6))

    # Left hand polar histogram
    counts_left, bin_edges_left = np.histogram(theta_left, bins=num_bins, weights=r_left)
    width_left = bin_edges_left[1] - bin_edges_left[0]
    max_left = counts_left.max() if counts_left.max() != 0 else 1
    normalized_counts_left = counts_left / max_left

    axes[1].bar(bin_edges_left[:-1], counts_left, width=width_left, bottom=0.0,
                color=[cmap_choice(val) for val in normalized_counts_left],
                edgecolor='k', alpha=0.75)
    axes[1].set_title("Left Hand Polar Histogram")

    # Right hand polar histogram
    counts_right, bin_edges_right = np.histogram(theta_right, bins=num_bins, weights=r_right)
    width_right = bin_edges_right[1] - bin_edges_right[0]
    max_right = counts_right.max() if counts_right.max() != 0 else 1
    normalized_counts_right = counts_right / max_right

    axes[0].bar(bin_edges_right[:-1], counts_right, width=width_right, bottom=0.0,
                color=[cmap_choice(val) for val in normalized_counts_right],
                edgecolor='k', alpha=0.75)
    axes[0].set_title("Right Hand Polar Histogram")

    plt.tight_layout()
    plt.show()

### Plot p3_box2 and p3_block2 coordinates in a 3D scatter plot for a specific subject, hand, and trial
def plot_p3_coordinates(All_Subject_tBBTs_errors, subject='07/22/HW', hand='left', trial_index=26):
    """
    Extracts p3_box2 and p3_block2 coordinates from All_Subject_tBBTs_errors and plots them
    in a 3D scatter plot.

    Parameters:
        All_Subject_tBBTs_errors (dict): Dictionary containing error data.
        subject (str): Subject identifier.
        hand (str): Hand identifier ('left' or 'right').
        trial_index (int): Index of the trial to extract data from.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

    # Extract the p3_box2 values; they represent x, y, z coordinates
    p3_box2 = All_Subject_tBBTs_errors[(subject, hand)][trial_index]['p3_box2']
    p3_box2_array = np.array(p3_box2)
    if p3_box2_array.ndim == 2 and p3_box2_array.shape[0] == 3:
        coords_box2 = p3_box2_array.T
    else:
        coords_box2 = p3_box2_array

    # Extract the p3_block2 values; process similarly
    p3_block2 = All_Subject_tBBTs_errors[(subject, hand)][trial_index]['p3_block2']
    p3_block2_array = np.array(p3_block2)
    if p3_block2_array.ndim == 2 and p3_block2_array.shape[0] == 3:
        coords_block2 = p3_block2_array.T
    else:
        coords_block2 = p3_block2_array

    # Create a 3D scatter plot and overlay both coordinate sets
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords_box2[:, 0], coords_box2[:, 1], coords_box2[:, 2],
               c='r', marker='o', label='p3_box2')
    ax.scatter(coords_block2[:, 0], coords_block2[:, 1], coords_block2[:, 2],
               c='b', marker='^', label='p3_block2')

    ax.set_title("3D Plot of p3_box2 and p3_block2 Coordinates")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.show()

### Plot hand trajectory with velocity-coded coloring and highlighted segments
def plot_trajectory(results, subject='07/22/HW', hand='right', trial=1,
                    file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT53.csv',
                    overlay_trial=0, velocity_segment_only=False, plot_mode='all'):
    """
    Plots individual coordinate plots and two 3D trajectory plots for the specified trial.
    Colors each trajectory point based on the instantaneous velocity for a selected segment 
    if velocity_segment_only is True; otherwise all points are colored according to velocity.
    Points outside a defined segment are colored lightgrey.
    
    Additionally, the 'plot_mode' option allows plotting:
      - 'all': the whole trial.
      - 'segment': only from the first highlight index to the last highlight index.
    
    Parameters:
        results (dict): The results dictionary containing trajectory data.
        subject (str): Subject key in the results dictionary.
        hand (str): Hand key ('right' or 'left') in the results dictionary.
        trial (int): The trial index to use for the main trajectory data.
        file_path (str): The file key for selecting trajectory data.
        overlay_trial (int): The trial index used to extract overlay indices for highlighting.
        velocity_segment_only (bool): If True, apply velocity-coded color only within highlighted segments.
        plot_mode (str): 'all' to plot the entire trial or 'segment' to plot only from the first to the last highlight.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as mcolors

    # Extract trajectory data for the given trial
    traj_data = results[subject][hand][trial][file_path]['traj_data']
    coord_prefix = "RFIN_" if hand == "right" else "LFIN_"
    coord_x = np.array(traj_data[coord_prefix + "X"])
    coord_y = np.array(traj_data[coord_prefix + "Y"])
    coord_z = np.array(traj_data[coord_prefix + "Z"])
    
    # Extract overlay index (or indices) from the overlay_trial
    overlay_index = results[subject][hand][overlay_trial][file_path]
    highlight_indices = overlay_index if isinstance(overlay_index, (list, np.ndarray)) else [overlay_index]
    highlight_indices = sorted(highlight_indices)
    
    n_points = len(coord_x)
    
    # Compute instantaneous velocity from the trajectory space (assume constant sampling rate 200Hz)
    vel = results[subject][hand][trial][file_path]['traj_space']['RFIN'][1]
    
    # Normalize velocities between 0 and 1
    v_min = np.min(vel)
    v_max = np.max(vel)
    if v_max - v_min > 0:
        v_norm = (vel - v_min) / (v_max - v_min)
    else:
        v_norm = np.ones_like(vel)
    
    # Map velocity to colors via Viridis with exponential scaling for contrast
    point_colors = [plt.cm.viridis(1 - (v_norm[i]**2)) for i in range(n_points)]
    
    # If velocity_segment_only is True, only the points within each paired segment retain their velocity color.
    if velocity_segment_only and highlight_indices:
        segments = []
        for idx in range(0, len(highlight_indices) - 1, 2):
            segments.append((highlight_indices[idx], highlight_indices[idx+1]))
        for i in range(n_points):
            in_segment = any(min(seg) <= i <= max(seg) for seg in segments)
            if not in_segment:
                point_colors[i] = mcolors.to_rgba('lightgrey')
    
    # Determine the indices to plot based on plot_mode option
    if plot_mode == 'segment' and highlight_indices:
        start_idx = min(highlight_indices[0], highlight_indices[-1])
        end_idx = max(highlight_indices[0], highlight_indices[-1])
    else:
        start_idx = 0
        end_idx = n_points - 1

    # Slice the data to plot
    plot_indices = np.arange(start_idx, end_idx + 1)
    coord_x_plot = coord_x[plot_indices]
    coord_y_plot = coord_y[plot_indices]
    coord_z_plot = coord_z[plot_indices]
    vel_plot    = np.array(vel)[plot_indices]
    colors_plot = [point_colors[i] for i in plot_indices]
    time_points = plot_indices / 200

    # Create the plot layout: 4 rows on the left (velocity, X, Y, Z) and one 3D plot on the right
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(nrows=4, ncols=2, width_ratios=[1, 1.2])
    
    ax_vel = fig.add_subplot(gs[0, 0])
    ax_vel.scatter(time_points, vel_plot, c=colors_plot, marker='o', s=5)
    ax_vel.set_ylabel('Velocity')
    ax_vel.set_title('Instantaneous Velocity')
    
    ax_x = fig.add_subplot(gs[1, 0])
    ax_x.scatter(time_points, coord_x_plot, c=colors_plot, marker='o', s=5)
    ax_x.set_ylabel('X')
    
    ax_y = fig.add_subplot(gs[2, 0])
    ax_y.scatter(time_points, coord_y_plot, c=colors_plot, marker='o', s=5)
    ax_y.set_ylabel('Y')
    
    ax_z = fig.add_subplot(gs[3, 0])
    ax_z.scatter(time_points, coord_z_plot, c=colors_plot, marker='o', s=5)
    ax_z.set_xlabel('Time (s)')
    ax_z.set_ylabel('Z')
    
    # Overlay markers at the designated highlight indices (if they fall within our plot range)
    for order, idx in enumerate(highlight_indices, start=1):
        if start_idx <= idx <= end_idx:
            t_val = idx / 200
            color = 'green' if order % 2 == 1 else 'blue'
            marker = 'o' if order % 2 == 1 else 'X'
            ax_vel.scatter(t_val, vel[idx], color=color, marker=marker, s=50)
            ax_x.scatter(t_val, coord_x[idx], color=color, marker=marker, s=50)
            ax_y.scatter(t_val, coord_y[idx], color=color, marker=marker, s=50)
            ax_z.scatter(t_val, coord_z[idx], color=color, marker=marker, s=50)
    
    # Right side: 3D Plot
    ax3d = fig.add_subplot(gs[:, 1], projection='3d')
    ax3d.scatter(coord_x_plot, coord_y_plot, coord_z_plot, c=colors_plot, marker='o', s=5)
    ax3d.set_xlabel(coord_prefix + "X (mm)", fontsize=14, labelpad=0)
    ax3d.set_ylabel(coord_prefix + "Y (mm)", fontsize=14, labelpad=0)
    ax3d.set_zlabel(coord_prefix + "Z (mm)", fontsize=14, labelpad=0)
    ax3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax3d.set_xlim([min(coord_x_plot), max(coord_x_plot)])
    ax3d.set_ylim([min(coord_y_plot), max(coord_y_plot)])
    ax3d.set_zlim([min(coord_z_plot), max(coord_z_plot)])


    
    # Overlay markers on the 3D plot for indices within plot range
    for order, idx in enumerate(highlight_indices, start=1):
        if start_idx <= idx <= end_idx:
            color = 'green' if order % 2 == 1 else 'blue'
            marker = 'o' if order % 2 == 1 else 'X'
            ax3d.scatter(coord_x[idx], coord_y[idx], coord_z[idx], color=color, marker=marker, s=50)

    # ax3d.text(coord_x[highlight_indices[0]], coord_y[highlight_indices[0]], coord_z[highlight_indices[0]],
    #         "start", color='green', fontsize=20)
    # ax3d.text(coord_x[highlight_indices[1]], coord_y[highlight_indices[1]], coord_z[highlight_indices[1]],
    #         "end", color='blue', fontsize=20)    
    plt.tight_layout()
    plt.show()
    
    # Additional 3D Trajectory Plot for the first segment from first to last highlight, if possible
    if len(highlight_indices) >= 2:
        seg_start = highlight_indices[0]
        seg_end = highlight_indices[1]
        seg_indices = np.arange(seg_start, seg_end + 1)
        seg_coord_x = coord_x[seg_indices]
        seg_coord_y = coord_y[seg_indices]
        seg_coord_z = coord_z[seg_indices]
        seg_colors = [point_colors[i] for i in seg_indices]
        
        fig2 = plt.figure(figsize=(10, 8))
        ax3d_seg = fig2.add_subplot(111, projection='3d')
        ax3d_seg.scatter(seg_coord_x, seg_coord_y, seg_coord_z, c=seg_colors, marker='o', s=5)
        ax3d_seg.scatter(coord_x[seg_start], coord_y[seg_start], coord_z[seg_start],
                         color='green', marker='o', s=50, label='start')
        ax3d_seg.scatter(coord_x[seg_end], coord_y[seg_end], coord_z[seg_end],
                         color='blue', marker='X', s=50, label='end')
        
        ax3d_seg.set_xlabel(coord_prefix + "X (mm)")
        ax3d_seg.set_ylabel(coord_prefix + "Y (mm)")
        ax3d_seg.set_zlabel(coord_prefix + "Z (mm)")
        ax3d_seg.set_title("Selected 3D Trajectory Segment")
        ax3d_seg.legend()
        plt.tight_layout()
        plt.show()

### Combine hand trajectory and error coordinates in a single 3D plot
def combined_plot_trajectory_and_errors(results, All_Subject_tBBTs_errors,
                                        subject='07/22/HW', hand='right',
                                        trial=1, trial_index=26,
                                        file_path='/Users/yilinwu/Desktop/Yilin-Honours/Subject/Traj/2025/07/22/HW/HW_tBBT53.csv',
                                        overlay_trial=0, velocity_segment_only=False, plot_mode='all'):
    """
    Combines the 3D hand trajectory from the results dictionary with the p3_box2 and
    p3_block2 coordinates from the errors dictionary on a single 3D plot.
    
    Optionally, the hand trajectory can be plotted for the whole trial ('all') or
    only from the first segment start to the last segment end ('segment').
    
    Parameters:
        results (dict): Dictionary containing trajectory data.
        All_Subject_tBBTs_errors (dict): Dictionary containing error coordinate data.
        subject (str): Subject key.
        hand (str): 'right' or 'left'. Determines coordinate prefix.
        trial (int): Trial index for extracting hand trajectory data.
        trial_index (int): Index for extracting error coordinate data.
        file_path (str): File key in the results dictionary.
        overlay_trial (int): Trial index for overlay indices (if needed).
        velocity_segment_only (bool): If True, color the trajectory points using velocity
                                      only within designated highlighted segments.
        plot_mode (str): 'all' to plot the full trial or 'segment' to plot only from the first
                         segment start to the last segment end.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # --------------------
    # Extract hand trajectory data from results
    traj_data = results[subject][hand][trial][file_path]['traj_data']
    coord_prefix = "RFIN_" if hand == "right" else "LFIN_"
    # Note: Negate X for visualization consistency
    coord_x_full = -np.array(traj_data[coord_prefix + "X"])
    coord_y_full = np.array(traj_data[coord_prefix + "Y"])
    coord_z_full = np.array(traj_data[coord_prefix + "Z"])

    # Optionally, extract overlay indices from a different trial for highlighting (if needed)
    overlay_index = results[subject][hand][overlay_trial][file_path]
    highlight_indices = overlay_index if isinstance(overlay_index, (list, np.ndarray)) else [overlay_index]
    highlight_indices = sorted(highlight_indices)

    # --------------------
    # Compute instantaneous velocity from trajectory space (assume sampling rate of 200Hz)
    vel = results[subject][hand][trial][file_path]['traj_space']['RFIN'][1]
    n_points = len(coord_x_full)
    v_min = np.min(vel)
    v_max = np.max(vel)
    if v_max - v_min > 0:
        v_norm = (vel - v_min) / (v_max - v_min)
    else:
        v_norm = np.ones_like(vel)
    # Map velocity to colors via Viridis with exponential scaling for contrast
    point_colors_full = [plt.cm.viridis(1 - (v_norm[i]**2)) for i in range(n_points)]
    
    # If velocity_segment_only is True, only the points within each paired segment retain their velocity color.
    if velocity_segment_only and highlight_indices:
        segments = []
        for idx in range(0, len(highlight_indices) - 1, 2):
            segments.append((highlight_indices[idx], highlight_indices[idx+1]))
        for i in range(n_points):
            in_segment = any(min(seg) <= i <= max(seg) for seg in segments)
            if not in_segment:
                point_colors_full[i] = mcolors.to_rgba('lightgrey')
    
    # Determine plot range based on plot_mode option
    if plot_mode == 'segment' and highlight_indices:
        start_idx = min(highlight_indices)
        end_idx = max(highlight_indices)
    else:
        start_idx = 0
        end_idx = n_points - 1
    
    # Slice hand trajectory data and colors for plotting
    coord_x = coord_x_full[start_idx:end_idx + 1]
    coord_y = coord_y_full[start_idx:end_idx + 1]
    coord_z = coord_z_full[start_idx:end_idx + 1]
    point_colors = point_colors_full[start_idx:end_idx + 1]

    # --------------------
    # Extract error coordinates (p3_box2 and p3_block2) from All_Subject_tBBTs_errors
    # p3_box2 extraction
    p3_box2 = All_Subject_tBBTs_errors[(subject, hand)][trial_index]['p3_box2']
    p3_box2_array = np.array(p3_box2)
    if p3_box2_array.ndim == 2 and p3_box2_array.shape[0] == 3:
        coords_box2 = p3_box2_array.T
    else:
        coords_box2 = p3_box2_array

    # p3_block2 extraction
    p3_block2 = All_Subject_tBBTs_errors[(subject, hand)][trial_index]['p3_block2']
    p3_block2_array = np.array(p3_block2)
    if p3_block2_array.ndim == 2 and p3_block2_array.shape[0] == 3:
        coords_block2 = p3_block2_array.T
    else:
        coords_block2 = p3_block2_array

    # --------------------
    # Create a single 3D plot that overlays hand trajectory with error coordinates
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the hand trajectory with velocity-based coloring (or full lightgrey/velocity colors)
    ax.scatter(coord_x, coord_y, coord_z, c=point_colors, marker='o', s=5, label='Hand Trajectory')
    
    # Optionally, emphasize overlay indices (if within plot range) by plotting markers at these points
    for order, idx in enumerate(highlight_indices, start=1):
        if start_idx <= idx <= end_idx:
            if order % 2 == 1:
                ax.scatter(coord_x_full[idx], coord_y_full[idx], coord_z_full[idx],
                           color='green', marker='o', s=50,
                           label='Overlay Start' if order == 1 else "")
            else:
                ax.scatter(coord_x_full[idx], coord_y_full[idx], coord_z_full[idx],
                           color='blue', marker='X', s=50,
                           label='Overlay End' if order == 2 else "")
    
    # Overlay error coordinates: p3_box2 (red circles) and p3_block2 (purple triangles)
    ax.scatter(coords_box2[:, 0], coords_box2[:, 1], coords_box2[:, 2],
               c='red', marker='o', s=80, label='p3_box2')
    ax.scatter(coords_block2[:, 0], coords_block2[:, 1], coords_block2[:, 2],
               c='purple', marker='^', s=80, label='p3_block2')
    
    # Set axis labels using the coordinate prefix for proper units
    ax.set_xlabel(coord_prefix + "X (mm)")
    ax.set_ylabel(coord_prefix + "Y (mm)")
    ax.set_zlabel(coord_prefix + "Z (mm)")
    
    ax.legend()
    plt.tight_layout()
    plt.show()
