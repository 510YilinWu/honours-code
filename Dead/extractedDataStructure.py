# show the extractedData dictionary structure

def main(extractedData):
    for component, data in extractedData.items():
        print(f"{component}:")
        for key, value in data.items():
            if key != 'variables':
                print(f"  {key}: {value}")
            else:
                print("  variables:")
                for var, var_data in value.items():
                    print(f"    {var}:")
                    for axis, axis_data in var_data.items():
                        print(f"      {axis}:")


# # Print the extractedData dictionary without 'variables' values
# for component, data in extractedData.items():
#     print(f"{component}:")
#     for key, value in data.items():
#         if key != 'variables':
#             print(f"  {key}: {value}")
#         else:
#             print("  variables:")
#             for var, var_data in value.items():
#                 print(f"    {var}:")
#                 for axis, axis_data in var_data.items():
#                     print(f"      {axis}:")