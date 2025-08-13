import pickle
import os
import numpy as np

# --- UPDATE BLOCK DISTANCE KEYS TO MATCH FILENAMES IN REACH METRICS ---
def update_block_distance_keys(Block_Distance, reach_metrics, reach_sparc_test_windows_1, reach_TW_metrics):
    """
    Updates the keys of Block_Distance to match the filenames in reach_metrics for each subject and hand.

    Args:
        Block_Distance (dict): Dictionary containing block distance data.
        reach_metrics (dict): Dictionary containing reach metrics data.
        reach_sparc_test_windows_1 (dict): Dictionary containing SPARC test window data.
        reach_TW_metrics (dict): Dictionary containing time window metrics data.

    Returns:
        None: Updates Block_Distance in place.
    """
    for subject in Block_Distance:
        for hand in Block_Distance[subject]:
            if subject in reach_metrics['reach_durations'] and hand in reach_metrics['reach_durations'][subject] and \
               subject in reach_sparc_test_windows_1 and hand in reach_sparc_test_windows_1[subject] and \
               subject in reach_TW_metrics['reach_LDLJ'] and hand in reach_TW_metrics['reach_LDLJ'][subject] and \
               len(Block_Distance[subject][hand]) == len(reach_metrics['reach_durations'][subject][hand]) == \
               len(reach_sparc_test_windows_1[subject][hand]) == len(reach_TW_metrics['reach_LDLJ'][subject][hand]):

                filenames = list(reach_metrics['reach_durations'][subject][hand].keys())  # Get filenames in a list

                if len(filenames) != len(Block_Distance[subject][hand]):
                    print(f"Error: Mismatch in lengths for subject {subject}, hand {hand}.")
                    print(f"Filenames length: {len(filenames)}, Block_Distance length: {len(Block_Distance[subject][hand])}")
                else:
                    # Update Block_Distance keys to match filenames
                    updated_distance = {filenames[i]: v for i, v in enumerate(Block_Distance[subject][hand].values())}
                    Block_Distance[subject][hand] = updated_distance

                    # # Example usage
                    # print(f"Subject: {subject}, Hand: {hand}")
                    # print(Block_Distance[subject][hand])

# --- COMBINE DURATIONS, SPARC, LDLJ, AND DISTANCE, CALCULATED SPEED AND ACCURACY FOR ALL DATES ---
def combine_metrics_for_all_dates(reach_metrics, reach_sparc_test_windows_1, reach_TW_metrics, Block_Distance, all_dates):
    """
    Combines reach durations, SPARC, LDLJ, and distance metrics into a single dictionary for all dates and hands.

    Args:
        reach_metrics (dict): Dictionary containing reach durations.
        reach_sparc_test_windows_1 (dict): Dictionary containing SPARC metrics.
        reach_TW_metrics (dict): Dictionary containing LDLJ metrics.
        Block_Distance (dict): Dictionary containing distance metrics.
        all_dates (list): List of all dates to process.

    Returns:
        dict: Combined metrics for all dates and hands.
    """
    combined_metrics = {}

    for date in all_dates:
        combined_metrics[date] = {}
        for hand in ['left', 'right']:
            if (
                date in reach_metrics['reach_durations'] and hand in reach_metrics['reach_durations'][date] and
                date in reach_sparc_test_windows_1 and hand in reach_sparc_test_windows_1[date] and
                date in reach_TW_metrics['reach_LDLJ'] and hand in reach_TW_metrics['reach_LDLJ'][date] and
                date in Block_Distance and hand in Block_Distance[date]
            ):
                combined_metrics[date][hand] = {
                    "durations": {k: np.float64(v) for k, v in reach_metrics['reach_durations'][date][hand].items()},
                    "sparc": {k: np.float64(v) for k, v in reach_sparc_test_windows_1[date][hand].items()},
                    "ldlj": {k: np.float64(v) for k, v in reach_TW_metrics['reach_LDLJ'][date][hand].items()},
                    "distance": {k: np.float64(v) for k, v in Block_Distance[date][hand].items()},
                    "speed": {k: 1 / np.float64(v) if v != 0 else np.nan for k, v in reach_metrics['reach_durations'][date][hand].items()},
                    "accuracy": {k: 1 / np.float64(v) if v != 0 else np.nan for k, v in Block_Distance[date][hand].items()}
                }


    return combined_metrics

# --- CALCULATE MOTOR ACUITY FOR ALL REACHES, EACH HAND ---
def calculate_motor_acuity_for_all(all_combined_metrics):
    for subject in all_combined_metrics:
        hands = ['left', 'right']

        for hand in hands:
            trials = all_combined_metrics[subject][hand]['speed'].keys()

            for trial_path in trials:
                speeds = list(all_combined_metrics[subject][hand]['speed'][trial_path])
                accuracies = list(all_combined_metrics[subject][hand]['accuracy'][trial_path])

                # Calculate motor acuity for all reaches
                motor_acuity_list = []
                for reach_index in range(len(speeds)):
                    if np.isnan(accuracies[reach_index]):
                        motor_acuity = np.nan
                    else:
                        motor_acuity = np.sqrt(speeds[reach_index]**2 + accuracies[reach_index]**2)
                    motor_acuity_list.append(motor_acuity)

                if 'motor_acuity' not in all_combined_metrics[subject][hand]:
                    all_combined_metrics[subject][hand]['motor_acuity'] = {}
                all_combined_metrics[subject][hand]['motor_acuity'][trial_path] = motor_acuity_list

    return all_combined_metrics

# --- LOCATE NaN INDICES (UNDETECTED BLOCK) FOR ALL SUBJECTS ---
def find_nan_indices_all_subjects(all_combined_metrics):
    nan_reach_indices = {}

    for subject in all_combined_metrics:
        nan_reach_indices[subject] = {}
        for hand in ['left', 'right']:
            if hand in all_combined_metrics[subject]:
                distance = all_combined_metrics[subject][hand]['distance']
                accuracy = all_combined_metrics[subject][hand]['accuracy']
                motor_acuity = all_combined_metrics[subject][hand]['motor_acuity']

                nan_indices = [
                    (trial_idx, value_idx)
                    for trial_idx, trial in enumerate(distance.values())
                    for value_idx, value in enumerate(trial)
                    if np.isnan(value)
                ]

                if not (
                    nan_indices == [
                        (trial_idx, value_idx)
                        for trial_idx, trial in enumerate(accuracy.values())
                        for value_idx, value in enumerate(trial)
                        if np.isnan(value)
                    ] == [
                        (trial_idx, value_idx)
                        for trial_idx, trial in enumerate(motor_acuity.values())
                        for value_idx, value in enumerate(trial)
                        if np.isnan(value)
                    ]
                ):
                    print(
                        f"Subject: {subject}, Hand: {hand} - NaN indices do not match."
                    )
                nan_reach_indices[subject][hand] = nan_indices

    return nan_reach_indices

# --- SAVE ALL COMBINED METRICS PER SUBJECT AS PICKLE FILE ---
def save_combined_metrics_per_subject(all_combined_metrics, output_folder):
    """
    Saves the combined metrics dictionary as separate pickle files for each subject.

    Args:
        all_combined_metrics (dict): The combined metrics to save.
        output_folder (str): The folder where the pickle files will be saved.

    Returns:
        None
    """

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for subject, metrics in all_combined_metrics.items():
        # Replace slashes in subject names to create valid file paths
        sanitized_subject = subject.replace("/", "_")
        output_file = f"{output_folder}/{sanitized_subject}_combined_metrics.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(metrics, f)
        print(f"Combined metrics for subject {subject} saved to {output_file}")

# --- LOAD ALL COMBINED METRICS PER SUBJECT FROM PICKLE FILE ---
def load_combined_metrics_per_subject(input_folder):
    """
    Loads the combined metrics dictionary from separate pickle files for each subject.

    Args:
        input_folder (str): The folder where the pickle files are located.

    Returns:
        dict: A dictionary containing combined metrics for each subject.
    """
    all_combined_metrics = {}

    # Iterate through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith("_combined_metrics.pkl"):
            subject = filename.replace("_combined_metrics.pkl", "").replace("_", "/")
            file_path = os.path.join(input_folder, filename)

            with open(file_path, 'rb') as f:
                all_combined_metrics[subject] = pickle.load(f)
            print(f"Combined metrics for subject {subject} loaded from {file_path}")

    return all_combined_metrics 

# --- PROCESS AND SAVE COMBINED METRICS FOR ALL SUBJECTS ---
def process_and_save_combined_metrics(Block_Distance, reach_metrics, reach_sparc_test_windows_1, reach_TW_metrics, All_dates, DataProcess_folder):
    """
    Combines multiple processing steps into one function:
    1. Updates Block_Distance keys to match filenames in reach_metrics.
    2. Combines durations, SPARC, LDLJ, and distance, and calculates speed and accuracy for all dates.
    3. Calculates motor acuity for all reaches for each hand.
    4. Saves all combined metrics per subject as pickle files.

    Args:
        Block_Distance (dict): Dictionary containing block distance data.
        reach_metrics (dict): Dictionary containing reach metrics data.
        reach_sparc_test_windows_1 (dict): Dictionary containing SPARC test window data.
        reach_TW_metrics (dict): Dictionary containing time window metrics data.
        All_dates (list): List of all dates to process.
        DataProcess_folder (str): The folder where the pickle files will be saved.

    Returns:
        None
    """
    # Step 1: Update Block_Distance keys
    update_block_distance_keys(Block_Distance, reach_metrics, reach_sparc_test_windows_1, reach_TW_metrics)

    # Step 2: Combine metrics for all dates
    all_combined_metrics = combine_metrics_for_all_dates(
        reach_metrics, reach_sparc_test_windows_1, reach_TW_metrics, Block_Distance, All_dates
    )

    # Step 3: Calculate motor acuity for all reaches
    all_combined_metrics = calculate_motor_acuity_for_all(all_combined_metrics)

    # Step 4: Save combined metrics per subject
    save_combined_metrics_per_subject(all_combined_metrics, DataProcess_folder)

