"""
Properties
    Filename: Current Trial 
    File Extension: .CSV 
    First Frame: First Frame 
    Last Frame: End Frame 
    Delimiter: ,
    Export Gait Cycle Parameters: None
    Export Events: None
    Digital Device Sampling: MX Frames 
    Local Numeric Format: None

Devices 
    Devices for Export: None 
    Combined Forceplates: None 

    Device Velocitres 
    Devices for Export: None 
    Combined Forceplates: None 

    Device Accelerations 
    Devices for Export: None 
    Combined Forceplates: None 

Joints 
    Kinematics: All 

    Joint Velocities
    Kinematics: None 

    Joint Accelerations 
    Kinematics: None 

Model Outputs 
    Model Outputs: All 
    Modeled Markers: All 

    Model Output Velocities 
    Model Outputs: empty 
    Modeled Markers: All 

    Model Output Accelerations 
    Model Outputs: empty 
    Modeled Markers: All 

Segments 
    Global Angle: All

    Segment Velocities 
    Global Angle: All

    Segment Accelerations 
    Global Angle: All

Trajectories 
    Components: All
    Distance From Origin: All
    Trajectory Count: Yes

    Trajectory Velocities 
    Components: All
    Distance From Origin: All
    Trajectory Count: All

    Trajectory Accelerations 
    Components: All
    Distance From Origin: All
    Trajectory Count: All
"""

import csv
import pandas as pd
import pprint
import sys
import os


def GetComponentsIndex(file_path):
    """
    Searches for specific terms in the first column of a CSV file and records the row numbers and 
    the first column of the subsequent row for each term found.
    Args:
        file_path (str): The path to the CSV file to be searched.
    Returns:
        dict: A dictionary where the keys are the search terms and the values are lists containing 
              the row numbers where the terms were found and the capture rates (first column of the subsequent row).
    Raises:
        ValueError: If any of the search terms are not found in the CSV file.
    Example:
        >>> search_csv('/path/to/file.csv')
        {'Joints': [1, '100'], 'Model Outputs': [4470, '100'], 'Segments': [8939, '100'], 'Trajectories': [13408, '100']}
    """
    search_terms = ['Joints', 'Model Outputs', 'Segments', 'Trajectories']
    ComponentsIndex = {term: [] for term in search_terms}
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row_number, row in enumerate(reader, start=1):
            if row:
                for term in search_terms:
                    if term in row[0]:  # Check first column for search terms
                        ComponentsIndex[term].append(row_number)  # Store the row number for matches
                        next_row = next(reader, None)
                        if next_row:
                            ComponentsIndex[term].append(next_row[0])  # Store the next row's first column as capture rate
    missing_terms = [term for term in search_terms if not ComponentsIndex[term]]
    if missing_terms:
        print(f"Missing search terms in CSV: {', '.join(missing_terms)}")
        search_terms = [term for term in search_terms if term not in missing_terms]
        ComponentsIndex = {term: ComponentsIndex[term] for term in search_terms}
    return ComponentsIndex, search_terms

def extractHeader(file_path, search_terms):
    """
    Extracts headers from a CSV file based on specified search terms.
    Args:
        file_path (str): The path to the CSV file.
        search_terms (list of str): A list of terms to search for in the first column of the CSV file.
    Returns:
        dict: A dictionary where each key is a search term and the value is a list of header rows found after the search term.
    """
    extracted_header = {term: [] for term in search_terms}
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                for term in search_terms:
                    if term in row[0]:  # Check first column for search terms
                        header_row = [next(reader, None) for _ in range(2)][-1]  # Get the header row (1 rows after the search term)
                        if header_row:
                            extracted_header[term].append(header_row)  # Store the header_row
    return extracted_header

def extract_subjects_and_variables(extracted_header):
    """
    Extracts subjects and variables from the extracted headers.
    Args:
        extracted_header (dict): A dictionary where each key is a search term and the value is a list of header rows found after the search term.
    Returns:
        dict: A dictionary where each key is a search term and the value is a tuple containing the subject and a list of unique variables.
    """
    subjects_and_variables = {term: (None, []) for term in extracted_header}
    for term, headers in extracted_header.items():
        for header in headers:
            for item in header:
                if ':' in item:
                    subject, variable = item.split(':', 1)
                    variable = variable.strip('| ')  # Remove leading/trailing '| ' for Trajectories
                    if subjects_and_variables[term][0] is None:
                        subjects_and_variables[term] = (subject, [variable])
                    else:
                        if variable not in subjects_and_variables[term][1]:
                            subjects_and_variables[term][1].append(variable)
    return subjects_and_variables

def GetFrame(file_path):
    frame_indices, maxFrameIndex, FrameValue = [], [], []
    maxFrameValue, data = float('-inf'), []

    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader, start=1):
            if not row: continue
            data.append(row)

            if 'Frame' in row[0]: frame_indices.append(i)
            try:
                value = float(row[0])
                if value > maxFrameValue:
                    maxFrameValue, maxFrameIndex = value, [i]
                elif value == maxFrameValue:
                    maxFrameIndex.append(i)
            except ValueError:
                pass  

    if not (frame_indices and maxFrameIndex):
        return {"error": "Frame or max value index not found."}

    frame_range = range(frame_indices[0] + 2, maxFrameIndex[0] + 1)
    FrameValue = [data[i - 1][0] for i in frame_range if i - 1 < len(data)]

    return {
        "frame_indices": frame_indices,
        "maxFrameIndex": maxFrameIndex,
        "FrameValue": FrameValue,
    }

def update_properties_dict(file_path):
    global properties_dict
    properties_dict = {}
    ComponentsIndex, search_terms = GetComponentsIndex(file_path)
    properties_dict.update({
        "ComponentsIndex": ComponentsIndex,
        "extracted_header": extractHeader(file_path, search_terms),
        "subjects_and_variables": extract_subjects_and_variables(extractHeader(file_path, search_terms)),
        "frame_property": GetFrame(file_path)
    })

"""
This module provides functions to create structured dictionaries for capturing motion data, including joints, model outputs, segments, and trajectories.

Functions:
    create_rotation_structure() -> dict:
        Creates a dictionary structure for rotation data with position, velocities, and accelerations.
    
    create_translation_structure() -> dict:
        Creates a dictionary structure for translation data with position, velocities, and accelerations.
    
    create_joints_structure() -> dict:
        Creates a dictionary structure for joints, including rotation and translation data for specific joints.
    
    create_model_outputs_structure(unit: str) -> dict:
        Creates a dictionary structure for model outputs with position, velocities, and accelerations, using the specified unit.
    
    create_model_outputs() -> dict:
        Creates a dictionary structure for model outputs for predefined variables.
    
    create_segment_structure() -> dict:
        Creates a dictionary structure for segments, including rotation and translation data.
    
    create_segments() -> dict:
        Creates a dictionary structure for segments for predefined variables.
    
    create_trajectory_structure() -> dict:
        Creates a dictionary structure for trajectories with position, velocities, accelerations, and magnitude data.
    
    create_trajectories() -> dict:
        Creates a dictionary structure for trajectories for predefined variables.

Variables:
    Joints (dict): A dictionary containing the structure for joints data.
    ModelOutputs (dict): A dictionary containing the structure for model outputs data.
    Segments (dict): A dictionary containing the structure for segments data.
    Trajectories (dict): A dictionary containing the structure for trajectories data.
"""

def create_rotation_structure():
    return {
        "position": {"RX": None, "RY": None, "RZ": None, "unit": "deg"},
        "Velocities": {"RX'": None, "RY'": None, "RZ'": None, "unit": "deg/s"},
        "Accelerations": {"RX''": None, "RY''": None, "RZ''": None, "unit": "deg/s^2"}
    }

def create_translation_structure():
    return {
        "position": {"TX": None, "TY": None, "TZ": None, "unit": "mm"},
        "Velocities": {"TX'": None, "TY'": None, "TZ'": None, "unit": "mm/s"},
        "Accelerations": {"TX''": None, "TY''": None, "TZ''": None, "unit": "mm/s^2"}
    }

def create_joints_structure():
    joints_variable = properties_dict["subjects_and_variables"]["Joints"][1]
    joints_structure = {}
    for joint in joints_variable:
        if joint == "World_Thorax":
            joints_structure[joint] = {
                "Rotation": create_rotation_structure(),
                "Translation": create_translation_structure()
            }
        else:
            joints_structure[joint] = {"Rotation": create_rotation_structure()}
    return joints_structure

def GetJoints():
    return {
        "subject": properties_dict["subjects_and_variables"]["Joints"][0],
        "capture rate": properties_dict["ComponentsIndex"]["Joints"][1],
        "Frame": {
            "frame_indices": properties_dict["frame_property"]["frame_indices"][list(properties_dict["ComponentsIndex"].keys()).index('Joints')],
            "firstFrame": properties_dict["frame_property"]["frame_indices"][list(properties_dict["ComponentsIndex"].keys()).index('Joints')] + 2,
            "maxFrameIndex": properties_dict["frame_property"]["maxFrameIndex"][list(properties_dict["ComponentsIndex"].keys()).index('Joints')]
        },
        "variables": create_joints_structure()
    }

def create_model_outputs_structure(unit):
    return {
        "position": {"X": None, "Y": None, "Z": None, "unit": unit},
        "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": f"{unit}/s"},
        "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": f"{unit}/s^2"}
    }

def create_model_outputs():
    model_outputs_variable = properties_dict["subjects_and_variables"]["Model Outputs"][1]
    return {output: create_model_outputs_structure("mm") for output in model_outputs_variable}

def GetModelOutputs():
    return {
        "subject": properties_dict["subjects_and_variables"]["Model Outputs"][0],
        "capture rate": properties_dict["ComponentsIndex"]["Model Outputs"][1],
        "Frame": {
            "frame_indices": properties_dict["frame_property"]["frame_indices"][list(properties_dict["ComponentsIndex"].keys()).index('Model Outputs')],
            "firstFrame": properties_dict["frame_property"]["frame_indices"][list(properties_dict["ComponentsIndex"].keys()).index('Model Outputs')] + 2,
            "maxFrameIndex": properties_dict["frame_property"]["maxFrameIndex"][list(properties_dict["ComponentsIndex"].keys()).index('Model Outputs')]
        },
        "variables": create_model_outputs()
    }

def create_segment_structure():
    return {
        "Rotation": create_rotation_structure(),
        "Translation": create_translation_structure()
    }

def create_segments():
    segments_variable = properties_dict["subjects_and_variables"]["Segments"][1]
    return {segment: create_segment_structure() for segment in segments_variable}

def GetSegments():
    return {
        "subject": properties_dict["subjects_and_variables"]["Segments"][0],
        "capture rate": properties_dict["ComponentsIndex"]["Segments"][1],
        "Frame": {
            "frame_indices": properties_dict["frame_property"]["frame_indices"][list(properties_dict["ComponentsIndex"].keys()).index('Segments')],
            "firstFrame": properties_dict["frame_property"]["frame_indices"][list(properties_dict["ComponentsIndex"].keys()).index('Segments')] + 2,
            "maxFrameIndex": properties_dict["frame_property"]["maxFrameIndex"][list(properties_dict["ComponentsIndex"].keys()).index('Segments')]
        },
        "variables": create_segments()
    }

def create_trajectory_structure():
    return {
        "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
        "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
        "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
        "Magnitude": {
            "position": {"unit": "mm"},
            "Velocities": {"unit": "mm/s"},
            "Accelerations": {"unit": "mm/s^2"}
        }
    }

def create_trajectories():
    trajectory_variables = properties_dict["subjects_and_variables"]["Trajectories"][1]
    return {var: create_trajectory_structure() for var in trajectory_variables}

def GetTrajectories():
    return {
        "subject": properties_dict["subjects_and_variables"]["Trajectories"][0],
        "capture rate": properties_dict["ComponentsIndex"]["Trajectories"][1],
        "Frame": {
            "frame_indices": properties_dict["frame_property"]["frame_indices"][list(properties_dict["ComponentsIndex"].keys()).index('Trajectories')],
            "firstFrame": properties_dict["frame_property"]["frame_indices"][list(properties_dict["ComponentsIndex"].keys()).index('Trajectories')] + 2,
            "maxFrameIndex": properties_dict["frame_property"]["maxFrameIndex"][list(properties_dict["ComponentsIndex"].keys()).index('Trajectories')]
        },
        "variables": create_trajectories(),
        "Trajectory Count": {
            "position": {"unit": "count"},
            "Velocities": {"unit": "Count', Hz"},
            "Accelerations": {"unit": "Count'', Hz/s"}
        }
    }

def initialize_extracted_data(properties_dict):
    """
    Initializes the extracted data dictionary by running the corresponding function for each component.
    Args:
        properties_dict (dict): The dictionary containing properties and extracted headers.
    Returns:
        dict: The initialized extracted data dictionary.
    """
    extractedData = {}
    component_functions = {
        "Joints": GetJoints,
        "Model Outputs": GetModelOutputs,
        "Segments": GetSegments,
        "Trajectories": GetTrajectories
    }
    for component, func in component_functions.items():
        if component in properties_dict["ComponentsIndex"]:
            extractedData[component] = func()
    return extractedData

def populate_model_outputs(extractedData, properties_dict, file_path):
    """
    Populates the extractedData dictionary with data points for Model Outputs variables.
    Args:
        extractedData (dict): The dictionary to be populated with data points.
        properties_dict (dict): The dictionary containing extracted headers and other properties.
    """
    extracted_header = properties_dict["extracted_header"]["Model Outputs"]
    first_frame = extractedData['Model Outputs']['Frame']['firstFrame']
    max_frame_index = extractedData['Model Outputs']['Frame']['maxFrameIndex']
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)  # Read all data into a list
        for variable in extractedData['Model Outputs']['variables']:
            for header in extracted_header:
                if f"Yilin:{variable}" in header:
                    col_indices = [i for i, item in enumerate(header) if item == f"Yilin:{variable}"]
                    for col_index in col_indices:
                        if data[extractedData['Model Outputs']['Frame']['frame_indices']-1][col_index] == 'X':
                            extractedData['Model Outputs']['variables'][variable]['position']['X'] = [row[col_index] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Model Outputs']['variables'][variable]['position']['Y'] = [row[col_index+1] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Model Outputs']['variables'][variable]['position']['Z'] = [row[col_index+2] for row in data[first_frame-1:max_frame_index]]
                        if data[extractedData['Model Outputs']['Frame']['frame_indices']-1][col_index] == "X'":
                            extractedData['Model Outputs']['variables'][variable]['Velocities']["X'"] = [row[col_index] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Model Outputs']['variables'][variable]['Velocities']["Y'"] = [row[col_index+1] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Model Outputs']['variables'][variable]['Velocities']["Z'"] = [row[col_index+2] for row in data[first_frame-1:max_frame_index]]
                        if data[extractedData['Model Outputs']['Frame']['frame_indices']-1][col_index] == "X''":
                            extractedData['Model Outputs']['variables'][variable]['Accelerations']["X''"] = [row[col_index] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Model Outputs']['variables'][variable]['Accelerations']["Y''"] = [row[col_index+1] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Model Outputs']['variables'][variable]['Accelerations']["Z''"] = [row[col_index+2] for row in data[first_frame-1:max_frame_index]]

def populate_joints(extractedData, properties_dict, file_path):
    extracted_header = properties_dict["extracted_header"]["Joints"]
    first_frame = extractedData['Joints']['Frame']['firstFrame']
    max_frame_index = extractedData['Joints']['Frame']['maxFrameIndex']
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)
        for variable in extractedData['Joints']['variables']:
            for header in extracted_header:
                if f"Yilin:{variable}" in header:
                    col_indices = [i for i, item in enumerate(header) if item == f"Yilin:{variable}"]
                    for col_index in col_indices:
                        if data[extractedData['Joints']['Frame']['frame_indices']-1][col_index] == 'RX':
                            extractedData['Joints']['variables'][variable]['Rotation']['position']['RX'] = [row[col_index] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Joints']['variables'][variable]['Rotation']['position']['RY'] = [row[col_index+1] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Joints']['variables'][variable]['Rotation']['position']['RZ'] = [row[col_index+2] for row in data[first_frame-1:max_frame_index]]
                        if data[extractedData['Joints']['Frame']['frame_indices']-1][col_index] == "RX'":
                            extractedData['Joints']['variables'][variable]['Rotation']['Velocities']["RX'"] = [row[col_index] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Joints']['variables'][variable]['Rotation']['Velocities']["RY'"] = [row[col_index+1] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Joints']['variables'][variable]['Rotation']['Velocities']["RZ'"] = [row[col_index+2] for row in data[first_frame-1:max_frame_index]]
                        if data[extractedData['Joints']['Frame']['frame_indices']-1][col_index] == "RX''":
                            extractedData['Joints']['variables'][variable]['Rotation']['Accelerations']["RX''"] = [row[col_index] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Joints']['variables'][variable]['Rotation']['Accelerations']["RY''"] = [row[col_index+1] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Joints']['variables'][variable]['Rotation']['Accelerations']["RZ''"] = [row[col_index+2] for row in data[first_frame-1:max_frame_index]]

def populate_segments(extractedData, properties_dict, file_path):
    extracted_header = properties_dict["extracted_header"]["Segments"]
    first_frame = extractedData['Segments']['Frame']['firstFrame']
    max_frame_index = extractedData['Segments']['Frame']['maxFrameIndex']
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)
        for variable in extractedData['Segments']['variables']:
            for header in extracted_header:
                if f"Yilin:{variable}" in header:
                    col_indices = [i for i, item in enumerate(header) if item == f"Yilin:{variable}"]
                    for col_index in col_indices:
                        if data[extractedData['Segments']['Frame']['frame_indices']-1][col_index] == 'RX':
                            extractedData['Segments']['variables'][variable]['Rotation']['position']['RX'] = [row[col_index] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Segments']['variables'][variable]['Rotation']['position']['RY'] = [row[col_index+1] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Segments']['variables'][variable]['Rotation']['position']['RZ'] = [row[col_index+2] for row in data[first_frame-1:max_frame_index]]
                        if data[extractedData['Segments']['Frame']['frame_indices']-1][col_index] == "RX'":
                            extractedData['Segments']['variables'][variable]['Rotation']['Velocities']["RX'"] = [row[col_index] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Segments']['variables'][variable]['Rotation']['Velocities']["RY'"] = [row[col_index+1] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Segments']['variables'][variable]['Rotation']['Velocities']["RZ'"] = [row[col_index+2] for row in data[first_frame-1:max_frame_index]]
                        if data[extractedData['Segments']['Frame']['frame_indices']-1][col_index] == "RX''":
                            extractedData['Segments']['variables'][variable]['Rotation']['Accelerations']["RX''"] = [row[col_index] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Segments']['variables'][variable]['Rotation']['Accelerations']["RY''"] = [row[col_index+1] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Segments']['variables'][variable]['Rotation']['Accelerations']["RZ''"] = [row[col_index+2] for row in data[first_frame-1:max_frame_index]]

def populate_trajectories(extractedData, properties_dict, file_path):
    extracted_header = properties_dict["extracted_header"]["Trajectories"]
    first_frame = extractedData['Trajectories']['Frame']['firstFrame']
    max_frame_index = extractedData['Trajectories']['Frame']['maxFrameIndex']
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)
        for variable in extractedData['Trajectories']['variables']:
            for header in extracted_header:
                if f"Yilin:{variable}" in header:
                    col_indices = [i for i, item in enumerate(header) if item == f"Yilin:{variable}"]
                    for col_index in col_indices:
                        if data[extractedData['Trajectories']['Frame']['frame_indices']-1][col_index] == 'X':
                            extractedData['Trajectories']['variables'][variable]['position']['X'] = [row[col_index] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Trajectories']['variables'][variable]['position']['Y'] = [row[col_index+1] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Trajectories']['variables'][variable]['position']['Z'] = [row[col_index+2] for row in data[first_frame-1:max_frame_index]]
                        if data[extractedData['Trajectories']['Frame']['frame_indices']-1][col_index] == "X'":
                            extractedData['Trajectories']['variables'][variable]['Velocities']["X'"] = [row[col_index] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Trajectories']['variables'][variable]['Velocities']["Y'"] = [row[col_index+1] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Trajectories']['variables'][variable]['Velocities']["Z'"] = [row[col_index+2] for row in data[first_frame-1:max_frame_index]]
                        if data[extractedData['Trajectories']['Frame']['frame_indices']-1][col_index] == "X''":
                            extractedData['Trajectories']['variables'][variable]['Accelerations']["X''"] = [row[col_index] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Trajectories']['variables'][variable]['Accelerations']["Y''"] = [row[col_index+1] for row in data[first_frame-1:max_frame_index]]
                            extractedData['Trajectories']['variables'][variable]['Accelerations']["Z''"] = [row[col_index+2] for row in data[first_frame-1:max_frame_index]]

def populate_all_data(extractedData, properties_dict, file_path):
    component_functions = {
        "Model Outputs": populate_model_outputs,
        "Joints": populate_joints,
        "Segments": populate_segments,
        "Trajectories": populate_trajectories
    }
    existing_components = []
    for component, func in component_functions.items():
        if component in properties_dict["ComponentsIndex"]:
            func(extractedData, properties_dict, file_path)
            existing_components.append(component)
    # print("Existing components:", existing_components)

def save_extracted_data(extractedData, file_path):
    """
    Saves the extractedData dictionary to a Python file and returns the output file path.
    Args:
        extractedData (dict): The dictionary containing the extracted data.
        file_path (str): The path to the CSV file from which data was extracted.
    Returns:
        str: The path to the saved Python file.
    """
    subject_name = extractedData['Model Outputs']['subject']
    file_name = file_path.split('/')[-1].split('.')[0]

    output_folder_path = '/Users/yilinwu/Desktop/honours data/Extracted data/YW20250318'
    os.makedirs(output_folder_path, exist_ok=True)  # Ensure the folder exists
    output_file_path = f'{output_folder_path}/{subject_name}-{file_name}.py'
    with open(output_file_path, 'w') as file:
        file.write("extractedData = ")
        pprint.pprint(extractedData, stream=file)

    # print(f"Extracted data has been saved to {output_file_path}")
    return output_file_path

def main(file_path):
    update_properties_dict(file_path)  # Initialize the properties dictionary
    extractedData = initialize_extracted_data(properties_dict)  # Initialize the extracted data format
    populate_all_data(extractedData, properties_dict, file_path)  # Extract data
    save_extracted_data(extractedData, file_path)  # Call the function to save the extracted data

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python GetCSV.py <file_path> ")
        sys.exit(1)
    file_path = sys.argv[1]
    main(file_path)