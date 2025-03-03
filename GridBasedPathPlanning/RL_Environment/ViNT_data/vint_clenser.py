import os
import glob

def clear_files(base_path):
    """
    Recursively removes all .png, .npy, and .txt files from specified subdirectories.
    
    Parameters:
    - base_path (str): The root directory containing the target folders.
    """
    # List of target folders
    folders = [
        "dynamic_obs",
        "global_guidances",
        "local_guidances",
        "static_obs",
        "tjps_paths",
        "whole_maps_rgb",
        "local_maps_rgb",
    ]
    
    # File extensions to delete
    file_extensions = ['*.png', '*.npy', '*.txt']
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            print(f"Processing folder: {folder_path}")
            for ext in file_extensions:
                # Find all files with the given extension
                files = glob.glob(os.path.join(folder_path, '**', ext), recursive=True)
                for file_path in files:
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")
        else:
            print(f"Folder does not exist: {folder_path}")

# Example usage
base_directory = "/Users/czimbermark/Documents/SZTAKI/G2RL+/grid-based-path-planning/GridBasedPathPlanning/RL_Environment/ViNT_data"  # Set the full path to ViNT_data
clear_files(base_directory)