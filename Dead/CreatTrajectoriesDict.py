import pprint

data = {
    "Trajectories": {
        "subject": None,
        "capture rate": None,
        "Frame": None,
        "variables": {
            "C7": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "T10": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "CLAV": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "STRN": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "LSHO": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "LUPA": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "LUPB": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "LUPC": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "LELB": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "LMEP": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "LWRA": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "LWRB": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "LFIN": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "RSHO": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "RUPA": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "RUPB": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "RUPC": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "RELB": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "RMEP": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "RWRA": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "RWRB": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "RFRA": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            },
            "RFIN": {
                "position": {"X": None, "Y": None, "Z": None, "unit": "mm"},
                "Velocities": {"X'": None, "Y'": None, "Z'": None, "unit": "mm/s"},
                "Accelerations": {"X''": None, "Y''": None, "Z''": None, "unit": "mm/s^2"},
                "Magnitude": {
                    "position": {"unit": "mm"},
                    "Velocities": {"unit": "mm/s"},
                    "Accelerations": {"unit": "mm/s^2"}
                }
            }
        },
        "Trajectory Count": {
            "position": {"unit": "count"},
            "Velocities": {"unit": "Count', Hz"},
            "Accelerations": {"unit": "Count'', Hz/s"}
        }
    }
}

pprint.pprint(data)