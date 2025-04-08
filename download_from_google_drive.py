import gdown
import os
import zipfile
import pandas as pd
from dotenv import load_dotenv
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
    # Replace with your actual Google Drive folder ID
    # The folder ID is the part after /folders/ in your Google Drive URL
    # Example: https://drive.google.com/drive/folders/1AbCdEfG2HiJkLmNoPqRsTuVwXyZ - the ID is 1AbCdEfG2HiJkLmNoPqRsTuVwXyZ
    #load the environment variables
    load_dotenv()
    DRIVE_FOLDER_ID = os.getenv("DRIVE_DATASET_FOLDER_ID")  
    print("Drive folder ID Size: "  + str(len(DRIVE_FOLDER_ID)))
    print("Folder ID: " + DRIVE_FOLDER_ID[:5] + "..." + DRIVE_FOLDER_ID[-5:])
    prepare_bird_dataset(DRIVE_FOLDER_ID)