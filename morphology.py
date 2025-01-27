import os
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation, generate_binary_structure

# Directory containing the processed subfolders with .npy files
processed_root_dir = "/home/prithwijit/FAULTSEG/data/segment_label_numpy"  # Update with the folder containing modified labels

# Directory to save the skeletonized and dilated images as .npy files
final_output_dir = "/home/prithwijit/FAULTSEG/data/segment_label_npy_morphed"  # Update if saving to a different folder

# Ensure the output directory exists
os.makedirs(final_output_dir, exist_ok=True)

# Traverse through all subfolders
for subdir, _, files in os.walk(processed_root_dir):
    for file in files:
        if file.endswith(".npy"):
            # Full path to the processed .npy file
            processed_file_path = os.path.join(subdir, file)
            
            # Determine the corresponding final destination path
            relative_path = os.path.relpath(subdir, processed_root_dir)
            final_subdir = os.path.join(final_output_dir, relative_path)
            os.makedirs(final_subdir, exist_ok=True)  # Create the destination subfolder if it doesn't exist
            
            # Determine the .npy file path for saving the result
            npy_file_path = os.path.join(final_subdir, file)
            
            # Load the binary numpy array
            binary_array = np.load(processed_file_path) > 0  # Convert to binary (boolean array)
            
            # Apply skeletonization
            skeleton = skeletonize(binary_array)
            
            # Apply dilation
            s = generate_binary_structure(2, 1)
            dilated = binary_dilation(skeleton, structure=s, iterations=1)
            
            # Save the result as a .npy file
            np.save(npy_file_path, dilated)

print("Skeletonization and dilation applied to all .npy files and saved as .npy files.")
