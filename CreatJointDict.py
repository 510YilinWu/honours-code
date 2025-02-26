import pprint

data_structure = {
    "Joints": {
        "subject": None,
        "capture rate": None,
        "Frame": None,
        "variables": {
            "LeftForeArm_LeftHand": {
                "Rotation": {
                    "position": {
                        "RX": None,
                        "RY": None,
                        "RZ": None,
                        "unit": "deg"
                    },
                    "Velocities": {
                        "RX'": None,
                        "RY'": None,
                        "RZ'": None,
                        "unit": "deg/s"
                    },
                    "Accelerations": {
                        "RX''": None,
                        "RY''": None,
                        "RZ''": None,
                        "unit": "deg/s^2"
                    }
                }
            },
            "LeftUpperArm_LeftForeArm": {
                "Rotation": {
                    "position": {
                        "RX": None,
                        "RY": None,
                        "RZ": None,
                        "unit": "deg"
                    },
                    "Velocities": {
                        "RX'": None,
                        "RY'": None,
                        "RZ'": None,
                        "unit": "deg/s"
                    },
                    "Accelerations": {
                        "RX''": None,
                        "RY''": None,
                        "RZ''": None,
                        "unit": "deg/s^2"
                    }
                }
            },
            "RightForeArm_RightHand": {
                "Rotation": {
                    "position": {
                        "RX": None,
                        "RY": None,
                        "RZ": None,
                        "unit": "deg"
                    },
                    "Velocities": {
                        "RX'": None,
                        "RY'": None,
                        "RZ'": None,
                        "unit": "deg/s"
                    },
                    "Accelerations": {
                        "RX''": None,
                        "RY''": None,
                        "RZ''": None,
                        "unit": "deg/s^2"
                    }
                }
            },
            "RightUpperArm_RightForeArm": {
                "Rotation": {
                    "position": {
                        "RX": None,
                        "RY": None,
                        "RZ": None,
                        "unit": "deg"
                    },
                    "Velocities": {
                        "RX'": None,
                        "RY'": None,
                        "RZ'": None,
                        "unit": "deg/s"
                    },
                    "Accelerations": {
                        "RX''": None,
                        "RY''": None,
                        "RZ''": None,
                        "unit": "deg/s^2"
                    }
                }
            },
            "Thorax_LeftUpperArm": {
                "Rotation": {
                    "position": {
                        "RX": None,
                        "RY": None,
                        "RZ": None,
                        "unit": "deg"
                    },
                    "Velocities": {
                        "RX'": None,
                        "RY'": None,
                        "RZ'": None,
                        "unit": "deg/s"
                    },
                    "Accelerations": {
                        "RX''": None,
                        "RY''": None,
                        "RZ''": None,
                        "unit": "deg/s^2"
                    }
                }
            },
            "Thorax_RightUpperArm": {
                "Rotation": {
                    "position": {
                        "RX": None,
                        "RY": None,
                        "RZ": None,
                        "unit": "deg"
                    },
                    "Velocities": {
                        "RX'": None,
                        "RY'": None,
                        "RZ'": None,
                        "unit": "deg/s"
                    },
                    "Accelerations": {
                        "RX''": None,
                        "RY''": None,
                        "RZ''": None,
                        "unit": "deg/s^2"
                    }
                }
            },
            "World_Thorax": {
                "Rotation": {
                    "position": {
                        "RX": None,
                        "RY": None,
                        "RZ": None,
                        "unit": "deg"
                    },
                    "Velocities": {
                        "RX'": None,
                        "RY'": None,
                        "RZ'": None,
                        "unit": "deg/s"
                    },
                    "Accelerations": {
                        "RX''": None,
                        "RY''": None,
                        "RZ''": None,
                        "unit": "deg/s^2"
                    }
                },
                "Translation": {
                    "position": {
                        "TX": None,
                        "TY": None,
                        "TZ": None,
                        "unit": "mm"
                    },
                    "Velocities": {
                        "TX'": None,
                        "TY'": None,
                        "TZ'": None,
                        "unit": "mm/s"
                    },
                    "Accelerations": {
                        "TX''": None,
                        "TY''": None,
                        "TZ''": None,
                        "unit": "mm/s^2"
                    }
                }
            }
        }
    }
}

pprint.pprint(data_structure)