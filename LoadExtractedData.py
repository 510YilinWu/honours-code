import importlib.util
import sys

# Load the extracted data from the saved Python file
def main(extracted_data_path):
    spec = importlib.util.spec_from_file_location("extractedData", extracted_data_path)
    extracted_data_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extracted_data_module)
    return extracted_data_module.extractedData

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python LoadExtractedData.py <extracted_data_path>")
        sys.exit(1)
    
    extracted_data_path = sys.argv[1]
    main(extracted_data_path)

