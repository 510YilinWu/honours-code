import pprint

data_structure = {
    "Segments": {
        "subject": None,
        "capture rate": None,
        "Frame": None,
        "variables": {
            "LeftForeArm": {
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
            },
            "LeftHand": {
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
            },
            "LeftUpperArm": {
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
            },
            "RightForeArm": {
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
            },
            "RightHand": {
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
            },
            "RightUpperArm": {
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
            },
            "Thorax": {
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