import gdown
import os
import zipfile
import pandas as pd
# from dotenv import load_dotenv # No longer needed if ID is a command-line argument
import argparse # For command-line arguments
import shutil # Will be used if we need to manage temp folders, but not for this simpler change.

# Hardcoded default ID for the main bird dataset if no ID is provided via command line
DEFAULT_MAIN_DATASET_ID = "1G9KBY2ULPwnNbMIjUQ1cHMlLpQABpofv"
DEFAULT_OUTPUT_PATH_FOR_MAIN_DATASET = "bird_sound_dataset"

def download_from_google_drive(folder_id=None, file_id=None, output_path='.', unzip=False):
    """
    Download files or folders from Google Drive.
    
    Args:
        folder_id (str): ID of the Google Drive folder (use this if downloading a folder)
        file_id (str): ID of a specific file (use this if downloading a single file)
        output_path (str): Where to save the downloaded files
        unzip (bool): Whether to unzip the downloaded file (if it's a zip)
    
    Returns:
        str: Path to the downloaded files
    """
    os.makedirs(output_path, exist_ok=True)
    
    if folder_id:
        # Download an entire folder
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        
        # Use gdown to download the folder as a zip
        zip_path = os.path.join(output_path, "drive_folder.zip")
        gdown.download_folder(url, output=output_path, quiet=False, remaining_ok=True)
        print(f"Folder downloaded to {output_path}")
        return output_path
        
    elif file_id:
        # Download a single file
        url = f"https://drive.google.com/uc?id={file_id}"
        output_file = os.path.join(output_path, "downloaded_file")
        gdown.download(url, output_file, quiet=False, remaining_ok=True)
        
        # Unzip if requested
        if unzip and output_file.endswith('.zip'):
            with zipfile.ZipFile(output_file, 'r') as zip_ref:
                zip_ref.extractall(output_path)
            print(f"File downloaded and extracted to {output_path}")
            return output_path
        else:
            print(f"File downloaded to {output_file}")
            return output_file
    else:
        print("Please provide either folder_id or file_id")
        return None

def prepare_bird_dataset(drive_folder_id, output_path="bird_sound_dataset"):
    """
    Download bird sounds from Google Drive and organize them into the required structure
    
    Args:
        drive_folder_id (str): Google Drive folder ID containing bird sounds
        output_path (str): Where to save the organized dataset
    """
    # First, install gdown if not already installed
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        import gdown
    
    # Download the folder
    print("Downloading bird sounds from Google Drive...")
    download_from_google_drive(folder_id=drive_folder_id, output_path=output_path)
    
    print(f"Bird sounds downloaded to {output_path}")
    print("Make sure your folder structure follows the required format:")
    print("bird_sound_dataset/")
    print("├── species_1/")
    print("│   ├── recording1.wav")
    print("│   ├── recording2.wav")
    print("├── species_2/")
    print("│   ├── recording1.wav")
    print("└── ...")

# Usage example:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Downloads a folder from Google Drive. \n"
            "- If no arguments are provided, downloads a default bird dataset. \n"
            "- If Folder ID and Local Folder Name are provided, downloads that specific folder."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Group for optional specific download parameters
    specific_download_group = parser.add_argument_group(
        title="Specific Folder Download (provide both or none)",
        description="To download a specific folder, you must provide both its Google Drive ID and a local name for it."
    )
    specific_download_group.add_argument(
        "folder_id",
        nargs='?',  # Optional positional argument
        default=None, 
        help=(
            "Google Drive Folder ID of the folder to download. \n"
            "Required if 'local_folder_name' is also provided."
        )
    )
    specific_download_group.add_argument(
        "local_folder_name",
        nargs='?',  # Optional positional argument
        default=None,
        help=(
            "The name for the local directory where the specified folder's contents will be saved. \n"
            "This directory will be created inside './" + DEFAULT_OUTPUT_PATH_FOR_MAIN_DATASET + "/'. \n"
            "Required if 'folder_id' is also provided."
        )
    )

    args = parser.parse_args()

    if args.folder_id is None and args.local_folder_name is None:
        # Scenario 1: No specific arguments provided, download default main dataset
        print(f"No specific folder arguments provided. Downloading the default main bird dataset (ID: {DEFAULT_MAIN_DATASET_ID}).")
        print(f"Output will be in: ./{DEFAULT_OUTPUT_PATH_FOR_MAIN_DATASET}/")
        
        prepare_bird_dataset(drive_folder_id=DEFAULT_MAIN_DATASET_ID, output_path=DEFAULT_OUTPUT_PATH_FOR_MAIN_DATASET)
        print(f"Default dataset download attempt finished. Check the '{DEFAULT_OUTPUT_PATH_FOR_MAIN_DATASET}' folder.")
    elif args.folder_id is not None and args.local_folder_name is not None:
        # Scenario 2: Specific Folder ID and Local Folder Name are provided
        target_folder_id = args.folder_id
        local_name = args.local_folder_name
        
        # Ensure the main dataset directory exists first
        os.makedirs(DEFAULT_OUTPUT_PATH_FOR_MAIN_DATASET, exist_ok=True)
        
        # The local output directory will be named as specified, inside the main dataset path.
        output_directory_path = os.path.join(DEFAULT_OUTPUT_PATH_FOR_MAIN_DATASET, local_name) 

        print(f"Google Drive Folder ID provided: {target_folder_id}")
        print(f"Local folder name specified: {local_name}")
        print(f"Attempting to download folder into: ./{output_directory_path}/")

        download_from_google_drive(folder_id=target_folder_id, output_path=output_directory_path)
        print(f"Download attempt for folder ID {target_folder_id} finished. Check the ./{output_directory_path}/ folder.")
    else:
        # Invalid combination of arguments (e.g., only one of folder_id or local_folder_name provided)
        parser.error(
            "Invalid arguments. You must provide EITHER no arguments (for default dataset download) "
            "OR BOTH Folder ID and Local Folder Name (for specific folder download)."
        )