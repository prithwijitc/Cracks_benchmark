import os
import numpy as np
from PIL import Image

# Define the mapping function
def modify_labels(label_array):
    # Map labels 0 and 1 to 0 (background)
    # Map labels 2 and 3 to 1 (object)
    label_array[label_array == 1] = 0
    label_array[label_array != 0] = 1  # Convert everything not 0 to 1s
    return label_array

# Directory containing the original subfolders with .png files
source_root_dir = "/home/prithwijit/FAULTSEG/faultSeg/data/f3/Fault segmentations"  # Update with your source folder path

# Directory where the modified files will be saved
destination_root_dir = "/home/prithwijit/FAULTSEG/data/segment_label_numpy"  # Update with your destination folder path

# Traverse through all subfolders
for subdir, _, files in os.walk(source_root_dir):
    for file in files:
        if file.endswith(".png"):
            # Full path to the original file
            source_file_path = os.path.join(subdir, file)
            
            # Determine the corresponding destination path
            relative_path = os.path.relpath(subdir, source_root_dir)
            destination_subdir = os.path.join(destination_root_dir, relative_path)
            os.makedirs(destination_subdir, exist_ok=True)  # Create destination subfolder if it doesn't exist
            
            # Modify the destination filename to save as .npy
            destination_file_path = os.path.join(destination_subdir, os.path.splitext(file)[0] + ".npy")
            
            # Open the image as grayscale
            image = Image.open(source_file_path)

            
            # Convert to numpy array for processing
            image_array = np.array(image, dtype=np.uint8)[1:255, :]
            
            # Modify the labels
            modified_array = modify_labels(image_array)
            
            # Save the modified array directly as a .npy file
            np.save(destination_file_path, modified_array)

print("All labels have been updated and saved as NumPy files in the new folder structure.")
