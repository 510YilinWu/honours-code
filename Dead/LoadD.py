import importlib.util
import sys

def LoadD(D_file_path):
    """
    Load the extracted data from the specified Python file and print it.
    """
    spec = importlib.util.spec_from_file_location("D", D_file_path)
    D_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(D_module)
    return D_module.D

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python LoadD.py <D_file_path>")
        sys.exit(1)
    
    D_file_path = sys.argv[1]
    LoadD(D_file_path)