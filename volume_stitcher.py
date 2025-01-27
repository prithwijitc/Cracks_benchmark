import os
import numpy as np

def stitch_sections(input_folder, output_folder, section_count=400, start_index=1):
    """
    Stitch .npy files in subfolders into 3D numpy arrays and save them in the output folder.
    Missing sections are replaced with zeros. Dynamically determines section shape.

    Parameters:
        input_folder (str): Path to the folder containing subfolders with .npy files.
        output_folder (str): Path to the folder where stitched 3D arrays will be saved.
        section_count (int): Total number of sections (default is 400).
        start_index (int): Starting index of sections (default is 1).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):
            section_shape = None
            
            # Find the first available section to determine the shape
            for i in range(start_index, start_index + section_count):
                section_filename = f"section_{i:03d}.npy"
                section_path = os.path.join(subfolder_path, section_filename)
                if os.path.exists(section_path):
                    first_section = np.load(section_path)
                    section_shape = first_section.shape
                    break
            
            if section_shape is None:
                print(f"No sections found in {subfolder}, skipping.")
                continue
            
            # Create a 3D array to hold the stitched sections
            stitched_array = np.zeros((section_count, *section_shape), dtype=np.float32)

            for i in range(start_index, start_index + section_count):
                section_filename = f"section_{i:03d}.npy"
                section_path = os.path.join(subfolder_path, section_filename)

                if os.path.exists(section_path):
                    section = np.load(section_path)
                    
                    # Check if the shape matches the determined shape
                    if section.shape != section_shape:
                        raise ValueError(f"Shape mismatch in {section_path}: "
                                         f"expected {section_shape}, got {section.shape}")
                    
                    stitched_array[i - start_index] = section
                else:
                    print(f"Section {section_filename} missing in {subfolder}, replaced with zeros.")
            
            # Save the stitched 3D array with the subfolder name
            output_path = os.path.join(output_folder, f"{subfolder}.npy")
            np.save(output_path, stitched_array)
            print(f"Saved stitched array for {subfolder} to {output_path}")

# Example usage
input_folder = "/home/prithwijit/FAULTSEG/data/segment_label_npy_morphed"
output_folder = "/home/prithwijit/FAULTSEG/data/volumes"
stitch_sections(input_folder, output_folder)
