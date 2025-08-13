# rename all .png files in a directory. 
# Read them in alphabetical order and rename as 
import os
import glob
import bbtLocalisation.charucoStereoCalib
import bbtLocalisation.helper.renamePNG
import bbtLocalisation.helper
import re
import pickle


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
        p3_box2, p3_block2, bg = bbtLocalisation.charucoStereoCalib.main(calib_folder, testFolder, testFiles)

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

def process_blocks(p3_block2, bg, hand):
    if len(p3_block2) < 16 or len(p3_block2) > 16:
        print(f"p3_block2: {len(p3_block2)}")
    else:
        pass
    if hand == 'right': # right hand
        x = bg.grid_xxL.flatten()
        y = bg.grid_yyL.flatten()       

    else: # left hand
        x = bg.grid_xxR.flatten()
        y = bg.grid_yyR.flatten()


    blockRange = [{'xRange': [x[i] - 20, x[i] + 20],
                   'yRange': [y[i] - 20, y[i] + 20]} for i in range(16)]

    blockMembership = [
        next((idx for idx, block in enumerate(blockRange)
              if block['xRange'][0] <= point[0] <= block['xRange'][1] and
                 block['yRange'][0] <= point[1] <= block['yRange'][1]), None)
        for point in p3_block2
    ]

    blocks_without_points = [idx for idx, block in enumerate(blockRange) if not any(
        block['xRange'][0] <= point[0] <= block['xRange'][1] and
        block['yRange'][0] <= point[1] <= block['yRange'][1] for point in p3_block2)]
    
    if len(blocks_without_points) == len([i for i, membership in enumerate(blockMembership) if membership is None]):
        for i, membership in enumerate(blockMembership):
            if membership is None:
                # Allocate the point to the nearest block based on distance
                distances_to_blocks = [
                    ((p3_block2[i][0] - (block['xRange'][0] + block['xRange'][1]) / 2) ** 2 +
                     (p3_block2[i][1] - (block['yRange'][0] + block['yRange'][1]) / 2) ** 2) ** 0.5
                    for block in blockRange
                ]
                nearest_block = distances_to_blocks.index(min(distances_to_blocks))
                blockMembership[i] = nearest_block
                print(f"Point {i} allocated to block {nearest_block}")


    for i, membership in enumerate(blockMembership):
        if membership is not None:
            pass
        else:

            # raise Exception(f"Error: Point {i} does not belong to any block")
            print(f"Error: Point {i} does not belong to any block")

    if blocks_without_points:
        print(f"Blocks without points: {blocks_without_points}")
    else:
        pass

    # # Create a list of length 16, marking blocks without points as 'nah'
    # block_status = ['nah' if idx in blocks_without_points else idx for idx in range(16)]
    # print(f"Block status: {block_status}")

    # Calculate the error for block placement, including distance
    block_errors = [
        {
            'point': p3_block2[i],
            'membership': blockMembership[i],
            # 'x': bg.grid_xxR.flatten()[blockMembership[i]] - 12.5 if blockMembership[i] is not None else None,
            # 'y': bg.grid_yyR.flatten()[blockMembership[i]] - 12.5 if blockMembership[i] is not None else None,
            'distance': (
                ((p3_block2[i][0] - (x[blockMembership[i]] - 12.5))**2 +
                 (p3_block2[i][1] - (y[blockMembership[i]] - 12.5))**2)**0.5
            ) if blockMembership[i] is not None else None
        }
        for i in range(len(p3_block2))
    ]

    # Add membership for blocks without points and keep point value as 'nah'
    for block_idx in blocks_without_points:
        blockMembership.append(None)  # Append None for blocks without points
        block_errors.append({
            'membership': block_idx,
        })

    

    return block_errors

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
        p3_box2, p3_block2, bg = bbtLocalisation.charucoStereoCalib.main(calib_folder, testFolder, testFiles) # YW - MAC

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
        block_errors = process_blocks(p3_block2, bg, hand)

        # Store the errors in the dictionary
        Subject_tBBTs_errors[image_number] = {
            'p3_box2': p3_box2,
            'p3_block2': p3_block2,
            'bg': bg,
            'block_errors': block_errors
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
    
    # Extract ordered distances and save them
    extract_ordered_distances(All_Subject_tBBTs_errors, DataProcess_folder)
    
    # return All_Subject_tBBTs_errors

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

