def classify_joint_movement(joint_name: str, euler_sequence: str, axis: str) -> str:
    """
    Classifies a movement type based on the joint, Euler sequence, and axis (X, Y, or Z).
    
    Parameters:
        joint_name (str): Name of the joint (e.g., "Shoulder", "Elbow", "Wrist").
        euler_sequence (str): The Euler angle sequence (e.g., "XZ'Y''").
        axis (str): The axis of interest ('X', 'Y', or 'Z').
        
    Returns:
        str: A string describing the type of movement.
    """
    
    movement_map = {
        "Shoulder": {
            "XZ'Y''": {
                'X': 'Flexion-Extension',
                'Z': 'Abduction-Adduction',
                'Y': 'Internal-External Rotation'
            },
            "YZ'Y''": {
                'Y1': 'Plane of Elevation',
                'Z': 'Elevation',
                'Y2': 'Internal-External Rotation'
            }
        },
        "Elbow": {
            "X'ZY''": {
                'X': 'Flexion-Extension',
                'Z': 'pronation-Supination',
            }
        },
        "Wrist": {
            "X'ZY''": {
                'X': 'Flexion-Extension',
                'Y': 'Radial-Ulnar Deviation',
                'Z': 'Abduction-Adduction'
            }
        }
    }

    # Special case for Shoulder YZ'Y''
    if joint_name == "Shoulder" and euler_sequence == "YZ'Y''":
        if axis == 'Y':
            return "Could be either 'Plane of Elevation' or 'Internal-External Rotation'"
        elif axis == 'Z':
            return movement_map[joint_name][euler_sequence]['Z']
        else:
            return "Invalid axis for YZ'Y'' sequence"
    
    # General lookup
    try:
        return movement_map[joint_name][euler_sequence][axis]
    except KeyError:
        return "Invalid input: Check joint, sequence, or axis."


# Example usage
print(classify_joint_movement("Shoulder", "XZ'Y''", 'X'))  # ➜ Flexion-Extension
print(classify_joint_movement("Shoulder", "XZ'Y''", 'Y'))  # ➜ Internal-External Rotation
print(classify_joint_movement("Shoulder", "XZ'Y''", 'Z'))  # ➜ Abduction-Adduction

# print(classify_joint_movement("Shoulder", "YZ'Y''", 'Z'))  # ➜ Elevation
# print(classify_joint_movement("Shoulder", "YZ'Y''", 'Y'))  # ➜ Could be either 'Plane of Elevation' or 'Internal-External Rotation'

print(classify_joint_movement("Elbow", "X'ZY''", 'X'))     # ➜ Flexion-Extension
print(classify_joint_movement("Elbow", "X'ZY''", 'Z'))     # ➜ pronation-Supination

print(classify_joint_movement("Wrist", "X'ZY''", 'X'))     # ➜ Flexion-Extension
print(classify_joint_movement("Wrist", "X'ZY''", 'Y'))     # ➜ Radial-Ulnar Deviation

