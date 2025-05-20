import pandas as pd


# # Read the file, skipping the first 5 rows
# df = pd.read_csv(
#     "/Users/yilinwu/Desktop/honours data/05/13/YW_tBBT01_Trajectories.csv",
#     skiprows=4,
#     sep=r"\s+|,",            # Split on whitespace or commas
#     engine="python"
# )

# # 1. Get the size of the DataFrame
# rows, columns = df.shape
# print(f"Number of rows: {rows}, Number of columns: {columns}")


# Read the file, skipping the first 5 rows
df = pd.read_csv(
    "/Users/yilinwu/Desktop/honours data/05/13/YW_tBBT01_Trajectories.csv",
    skiprows=0,             # Skip all the junk‐header lines
    nrows=1,            # Stop after reading 1 rows
    sep=r"\s+|,",            # Split on whitespace or commas
    engine="python"
)

rows, columns = df.shape
print(f"Number of rows: {rows}, Number of columns: {columns}")



# # Extract column 1 (second column)
# column1 = df.iloc[:, 0]

# print(column1)




# #    Adjust skiprows if your real file has more or fewer header-lines.
# df = pd.read_csv(
#     "/Users/yilinwu/Desktop/honours data/05/13/YW_tBBT01_Trajectories.csv",
#     skiprows=5,             # skip all the junk‐header lines
#     sep=r"\s+|,",           # split on whitespace or commas
#     engine="python",        # needed for the regex sep
# )






# # # 2. Take a look at what you’ve got
# # print(df.columns.tolist())

columns = {
    "C7": [2, 3, 4, 74, 75, 76, 146, 147, 148, 218, 242, 266],
    "T10": [5, 6, 7, 77, 78, 79, 149, 150, 151, 219, 243, 267],
    "CLAV": [8, 9, 10, 152, 153, 154, 220, 244, 268],
    "STRN": [11, 12, 13, 155, 156, 157, 221, 245, 269],
    "LSHO": [14, 15, 16, 158, 159, 160, 222, 246, 270],
    "LUPA": [17, 18, 19, 161, 162, 163, 223, 247, 271],
    "LUPB": [20, 21, 22, 164, 165, 166, 224, 248, 272],
    "LUPC": [23, 24, 25, 167, 168, 169, 225, 249, 273],
    "LELB": [26, 27, 28, 170, 171, 172, 226, 250, 274],
    "LMEP": [29, 30, 31, 173, 174, 175, 227, 251, 275],
    "LWRA": [32, 33, 34, 176, 177, 178, 228, 252, 276],
    "LWRB": [35, 36, 37, 179, 180, 181, 229, 253, 277],
    "LFRA": [38, 39, 40, 182, 183, 184, 230, 254, 278],
    "LFIN": [41, 42, 43, 185, 186, 187, 231, 255, 279],
    "RSHO": [44, 45, 46, 188, 189, 190, 232, 256, 280],
    "RUPA": [47, 48, 49, 191, 192, 193, 233, 257, 281],
    "RUPB": [50, 51, 52, 194, 195, 196, 234, 258, 282],
    "RUPC": [53, 54, 55, 197, 198, 199, 235, 259, 283],
    "RELB": [56, 57, 58, 200, 201, 202, 236, 260, 284],
    "RMEP": [59, 60, 61, 203, 204, 205, 237, 261, 285],
    "RWRA": [62, 63, 64, 206, 207, 208, 238, 262, 286],
    "RWRB": [65, 66, 67, 209, 210, 211, 239, 263, 287],
    "RFRA": [68, 69, 70, 212, 213, 214, 240, 264, 288],
    "RFIN": [71, 72, 73, 215, 216, 217, 241, 265, 289],
}

data = {key: df.iloc[:, indices] for key, indices in columns.items()}



# Define indices for position, velocity, acceleration, and other metrics
indices = {
    "pos_idx": {
        "C7": (2, 3, 4), "T10": (5, 6, 7), "CLAV": (8, 9, 10), "STRN": (11, 12, 13),
        "LSHO": (14, 15, 16), "LUPA": (17, 18, 19), "LUPB": (20, 21, 22), "LUPC": (23, 24, 25),
        "LELB": (26, 27, 28), "LMEP": (29, 30, 31), "LWRA": (32, 33, 34), "LWRB": (35, 36, 37),
        "LFRA": (38, 39, 40), "LFIN": (41, 42, 43), "RSHO": (44, 45, 46), "RUPA": (47, 48, 49),
        "RUPB": (50, 51, 52), "RUPC": (53, 54, 55), "RELB": (56, 57, 58), "RMEP": (59, 60, 61),
        "RWRA": (62, 63, 64), "RWRB": (65, 66, 67), "RFRA": (68, 69, 70), "RFIN": (71, 72, 73)
    },
    "vel_idx": {
        "C7": (74, 75, 76), "T10": (77, 78, 79)
    },
    "other_idx": {
        "C7": (80, 81, 82), "T10": (83, 84, 85)  # Example indices between 79 and 146
    },
    "acc_idx": {
        "C7": (146, 147, 148), "T10": (149, 150, 151), "CLAV": (152, 153, 154), "STRN": (155, 156, 157),
        "LSHO": (158, 159, 160), "LUPA": (161, 162, 163), "LUPB": (164, 165, 166), "LUPC": (167, 168, 169),
        "LELB": (170, 171, 172), "LMEP": (173, 174, 175), "LWRA": (176, 177, 178), "LWRB": (179, 180, 181),
        "LFRA": (182, 183, 184), "LFIN": (185, 186, 187), "RSHO": (188, 189, 190), "RUPA": (191, 192, 193),
        "RUPB": (194, 195, 196), "RUPC": (197, 198, 199), "RELB": (200, 201, 202), "RMEP": (203, 204, 205),
        "RWRA": (206, 207, 208), "RWRB": (209, 210, 211), "RFRA": (212, 213, 214), "RFIN": (215, 216, 217)
    },
    "mx_idx": {
        "C7": (218, 242, 266), "T10": (219, 243, 267), "CLAV": (220, 244, 268), "STRN": (221, 245, 269),
        "LSHO": (222, 246, 270), "LUPA": (223, 247, 271), "LUPB": (224, 248, 272), "LUPC": (225, 249, 273),
        "LELB": (226, 250, 274), "LMEP": (227, 251, 275), "LWRA": (228, 252, 276), "LWRB": (229, 253, 277),
        "LFRA": (230, 254, 278), "LFIN": (231, 255, 279), "RSHO": (232, 256, 280), "RUPA": (233, 257, 281),
        "RUPB": (234, 258, 282), "RUPC": (235, 259, 283), "RELB": (236, 260, 284), "RMEP": (237, 261, 285),
        "RWRA": (238, 262, 286), "RWRB": (239, 263, 287), "RFRA": (240, 264, 288), "RFIN": (241, 265, 289)
    }
}



T10_AX = df.iloc[:, 149]
T10_AY = df.iloc[:, 150]
T10_AZ = df.iloc[:, 151]

T10_MX = df.iloc[:, 219]
T10_MVX = df.iloc[:, 243]
T10_MAX = df.iloc[:, 267]

CLAV_AX = df.iloc[:, 152]
CLAV_AY = df.iloc[:, 153]
CLAV_AZ = df.iloc[:, 154]

CLAV_MX = df.iloc[:, 220]
CLAV_MVX = df.iloc[:, 244]
CLAV_MAX = df.iloc[:, 268]

STRN_AX = df.iloc[:, 155]
STRN_AY = df.iloc[:, 156]
STRN_AZ = df.iloc[:, 157]

STRN_MX = df.iloc[:, 221]
STRN_MVX = df.iloc[:, 245]
STRN_MAX = df.iloc[:, 269]

LSHO_AX = df.iloc[:, 158]
LSHO_AY = df.iloc[:, 159]
LSHO_AZ = df.iloc[:, 160]

LSHO_MX = df.iloc[:, 222]
LSHO_MVX = df.iloc[:, 246]
LSHO_MAX = df.iloc[:, 270]

LUPA_AX = df.iloc[:, 161]
LUPA_AY = df.iloc[:, 162]
LUPA_AZ = df.iloc[:, 163]

LUPA_MX = df.iloc[:, 223]
LUPA_MVX = df.iloc[:, 247]
LUPA_MAX = df.iloc[:, 271]

LUPB_AX = df.iloc[:, 164]
LUPB_AY = df.iloc[:, 165]
LUPB_AZ = df.iloc[:, 166]

LUPB_MX = df.iloc[:, 224]
LUPB_MVX = df.iloc[:, 248]
LUPB_MAX = df.iloc[:, 272]

LUPC_AX = df.iloc[:, 167]
LUPC_AY = df.iloc[:, 168]
LUPC_AZ = df.iloc[:, 169]

LUPC_MX = df.iloc[:, 225]
LUPC_MVX = df.iloc[:, 249]
LUPC_MAX = df.iloc[:, 273]

LELB_AX = df.iloc[:, 170]
LELB_AY = df.iloc[:, 171]
LELB_AZ = df.iloc[:, 172]

LELB_MX = df.iloc[:, 226]
LELB_MVX = df.iloc[:, 250]
LELB_MAX = df.iloc[:, 274]

LMEP_AX = df.iloc[:, 173]
LMEP_AY = df.iloc[:, 174]
LMEP_AZ = df.iloc[:, 175]

LMEP_MX = df.iloc[:, 227]
LMEP_MVX = df.iloc[:, 251]
LMEP_MAX = df.iloc[:, 275]

LWRA_AX = df.iloc[:, 176]
LWRA_AY = df.iloc[:, 177]
LWRA_AZ = df.iloc[:, 178]

LWRA_MX = df.iloc[:, 228]
LWRA_MVX = df.iloc[:, 252]
LWRA_MAX = df.iloc[:, 276]

LWRB_AX = df.iloc[:, 179]
LWRB_AY = df.iloc[:, 180]
LWRB_AZ = df.iloc[:, 181]

LWRB_MX = df.iloc[:, 229]
LWRB_MVX = df.iloc[:, 253]
LWRB_MAX = df.iloc[:, 277]

LFRA_AX = df.iloc[:, 182]
LFRA_AY = df.iloc[:, 183]
LFRA_AZ = df.iloc[:, 184]

LFRA_MX = df.iloc[:, 230]
LFRA_MVX = df.iloc[:, 254]
LFRA_MAX = df.iloc[:, 278]

LFIN_AX = df.iloc[:, 185]
LFIN_AY = df.iloc[:, 186]
LFIN_AZ = df.iloc[:, 187]

LFIN_MX = df.iloc[:, 231]
LFIN_MVX = df.iloc[:, 255]
LFIN_MAX = df.iloc[:, 279]

RSHO_AX = df.iloc[:, 188]
RSHO_AY = df.iloc[:, 189]
RSHO_AZ = df.iloc[:, 190]

RSHO_MX = df.iloc[:, 232]
RSHO_MVX = df.iloc[:, 256]
RSHO_MAX = df.iloc[:, 280]

RUPA_AX = df.iloc[:, 191]
RUPA_AY = df.iloc[:, 192]
RUPA_AZ = df.iloc[:, 193]

RUPA_MX = df.iloc[:, 233]
RUPA_MVX = df.iloc[:, 257]
RUPA_MAX = df.iloc[:, 281]

RUPB_AX = df.iloc[:, 194]
RUPB_AY = df.iloc[:, 195]
RUPB_AZ = df.iloc[:, 196]

RUPB_MX = df.iloc[:, 234]
RUPB_MVX = df.iloc[:, 258]
RUPB_MAX = df.iloc[:, 282]

RUPC_AX = df.iloc[:, 197]
RUPC_AY = df.iloc[:, 198]
RUPC_AZ = df.iloc[:, 199]

RUPC_MX = df.iloc[:, 235]
RUPC_MVX = df.iloc[:, 259]
RUPC_MAX = df.iloc[:, 283]

RELB_AX = df.iloc[:, 200]
RELB_AY = df.iloc[:, 201]
RELB_AZ = df.iloc[:, 202]

RELB_MX = df.iloc[:, 236]
RELB_MVX = df.iloc[:, 260]
RELB_MAX = df.iloc[:, 284]

RMEP_AX = df.iloc[:, 203]
RMEP_AY = df.iloc[:, 204]
RMEP_AZ = df.iloc[:, 205]

RMEP_MX = df.iloc[:, 237]
RMEP_MVX = df.iloc[:, 261]
RMEP_MAX = df.iloc[:, 285]

RWRA_AX = df.iloc[:, 206]
RWRA_AY = df.iloc[:, 207]
RWRA_AZ = df.iloc[:, 208]

RWRA_MX = df.iloc[:, 238]
RWRA_MVX = df.iloc[:, 262]
RWRA_MAX = df.iloc[:, 286]

RWRB_AX = df.iloc[:, 209]
RWRB_AY = df.iloc[:, 210]
RWRB_AZ = df.iloc[:, 211]

RWRB_MX = df.iloc[:, 239]
RWRB_MVX = df.iloc[:, 263]
RWRB_MAX = df.iloc[:, 287]

RFRA_AX = df.iloc[:, 212]
RFRA_AY = df.iloc[:, 213]
RFRA_AZ = df.iloc[:, 214]

RFRA_MX = df.iloc[:, 240]
RFRA_MVX = df.iloc[:, 264]
RFRA_MAX = df.iloc[:, 288]

RFIN_AX = df.iloc[:, 215]
RFIN_AY = df.iloc[:, 216]
RFIN_AZ = df.iloc[:, 217]

RFIN_MX = df.iloc[:, 241]
RFIN_MVX = df.iloc[:, 265]
RFIN_MAX = df.iloc[:, 289]

T10_VX = df.iloc[:, 77]
T10_VY = df.iloc[:, 78]
T10_VZ = df.iloc[:, 79]


T10_X = df.iloc[:, 5]
T10_Y = df.iloc[:, 6]
T10_Z = df.iloc[:, 7]

CLAV_X = df.iloc[:, 8]
CLAV_Y = df.iloc[:, 9]
CLAV_Z = df.iloc[:, 10]

STRN_X = df.iloc[:, 11]
STRN_Y = df.iloc[:, 12]
STRN_Z = df.iloc[:, 13]

LSHO_X = df.iloc[:, 14]
LSHO_Y = df.iloc[:, 15]
LSHO_Z = df.iloc[:, 16]

LUPA_X = df.iloc[:, 17]
LUPA_Y = df.iloc[:, 18]
LUPA_Z = df.iloc[:, 19]

LUPB_X = df.iloc[:, 20]
LUPB_Y = df.iloc[:, 21]
LUPB_Z = df.iloc[:, 22]

LUPC_X = df.iloc[:, 23]
LUPC_Y = df.iloc[:, 24]
LUPC_Z = df.iloc[:, 25]

LELB_X = df.iloc[:, 26]
LELB_Y = df.iloc[:, 27]
LELB_Z = df.iloc[:, 28]

LMEP_X = df.iloc[:, 29]
LMEP_Y = df.iloc[:, 30]
LMEP_Z = df.iloc[:, 31]

LWRA_X = df.iloc[:, 32]
LWRA_Y = df.iloc[:, 33]
LWRA_Z = df.iloc[:, 34]

LWRB_X = df.iloc[:, 35]
LWRB_Y = df.iloc[:, 36]
LWRB_Z = df.iloc[:, 37]

LFRA_X = df.iloc[:, 38]
LFRA_Y = df.iloc[:, 39]
LFRA_Z = df.iloc[:, 40]

LFIN_X = df.iloc[:, 41]
LFIN_Y = df.iloc[:, 42]
LFIN_Z = df.iloc[:, 43]

RSHO_X = df.iloc[:, 44]
RSHO_Y = df.iloc[:, 45]
RSHO_Z = df.iloc[:, 46]

RUPA_X = df.iloc[:, 47]
RUPA_Y = df.iloc[:, 48]
RUPA_Z = df.iloc[:, 49]

RUPB_X = df.iloc[:, 50]
RUPB_Y = df.iloc[:, 51]
RUPB_Z = df.iloc[:, 52]

RUPC_X = df.iloc[:, 53]
RUPC_Y = df.iloc[:, 54]
RUPC_Z = df.iloc[:, 55]

RELB_X = df.iloc[:, 56]
RELB_Y = df.iloc[:, 57]
RELB_Z = df.iloc[:, 58]

RMEP_X = df.iloc[:, 59]
RMEP_Y = df.iloc[:, 60]
RMEP_Z = df.iloc[:, 61]

RWRA_X = df.iloc[:, 62]
RWRA_Y = df.iloc[:, 63]
RWRA_Z = df.iloc[:, 64]

RWRB_X = df.iloc[:, 65]
RWRB_Y = df.iloc[:, 66]
RWRB_Z = df.iloc[:, 67]

RFRA_X = df.iloc[:, 68]
RFRA_Y = df.iloc[:, 69]
RFRA_Z = df.iloc[:, 70]

RFIN_X = df.iloc[:, 71]
RFIN_Y = df.iloc[:, 72]
RFIN_Z = df.iloc[:, 73]
