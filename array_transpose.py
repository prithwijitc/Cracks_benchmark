import os
import numpy as np

# Define the folder containing the .npy files
folder_path = "/home/prithwijit/FAULTSEG/data/volumes"

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".npy"):  # Process only .npy files
        file_path = os.path.join(folder_path, filename)
        
        # Load the .npy file
        array = np.load(file_path)
        
        # Check if the array has the expected shape (400, 255, 701)
        if array.shape == (400, 254, 701):
            # Transpose the array to shape (400, 701, 255)
            transposed_array = np.transpose(array, (0, 2, 1))
            
            # Save the transposed array, overwriting the original file
            np.save(file_path, transposed_array)
            print(f"Processed and saved: {filename}")
        else:
            print(f"Skipping {filename} (unexpected shape: {array.shape})")
