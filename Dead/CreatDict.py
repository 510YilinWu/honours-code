import pprint
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
    joints_variable = [
        "LeftForeArm_LeftHand", "LeftUpperArm_LeftForeArm", "RightForeArm_RightHand", 
        "RightUpperArm_RightForeArm", "Thorax_LeftUpperArm", "Thorax_RightUpperArm", 
        "World_Thorax" 
    ]
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

Joints = {
    "Joints": {
        "subject": None,
        "capture rate": None,
        "Frame": None,
        "variables": create_joints_structure()
    }
}

def create_model_outputs_structure(unit):
    return {
        "position": {"X": None, "Y": None, "Z": None, "unit": unit},
        "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": f"{unit}/s"},
        "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": f"{unit}/s^2"}
    }

def create_model_outputs():
    ModelOutputsvariable = ["LWJC", "RWJC", "Thorax1", "Thorax2", "Thorax3", "ThoraxO", 
              "leftHand1", "leftHand2", "leftHand3", "leftHandO", 
              "rightHand1", "rightHand2", "rightHand3", "rightHandO"]
    return {joint: create_model_outputs_structure("mm") for joint in ModelOutputsvariable}

ModelOutputs = {
    "Model Outputs": {
        "subject": None,
        "capture rate": None,
        "Frame": None,
        "variables": create_model_outputs()
    }
}

def create_segment_structure():
    return {
        "Rotation": create_rotation_structure(),
        "Translation": create_translation_structure()
    }

def create_segments():
    segmentsvariable = ["LeftForeArm", "LeftHand", "LeftUpperArm", "RightForeArm", "RightHand", "RightUpperArm", "Thorax"]
    return {segment: create_segment_structure() for segment in segmentsvariable}

Segments = {
    "Segments": {
        "subject": None,
        "capture rate": None,
        "Frame": None,
        "variables": create_segments()
    }
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
    trajectory_variables = [
        "C7", "T10", "CLAV", "STRN", "LSHO", "LUPA", "LUPB", "LUPC", "LELB", "LMEP", 
        "LWRA", "LWRB", "LFIN", "RSHO", "RUPA", "RUPB", "RUPC", "RELB", "RMEP", 
        "RWRA", "RWRB", "RFRA", "RFIN"
    ]
    return {var: create_trajectory_structure() for var in trajectory_variables}

Trajectories = {
    "Trajectories": {
        "subject": None,
        "capture rate": None,
        "Frame": None,
        "variables": create_trajectories(),
        "Trajectory Count": {
            "position": {"unit": "count"},
            "Velocities": {"unit": "Count', Hz"},
            "Accelerations": {"unit": "Count'', Hz/s"}
        }
    }
}

pprint.pprint(Joints)
pprint.pprint(ModelOutputs)
pprint.pprint(Segments)
pprint.pprint(Trajectories)


