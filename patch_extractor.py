import numpy as np
import os

def create_sliding_patches_with_multiple_labels(image_volume, label_volumes_dir, patch_size=(128, 128, 128), stride=128, save_dir="patches"):
    """
    Extracts sliding patches from a single image volume and multiple label volumes,
    saving them with consistent numbering across image and label patches.

    Args:
        image_volume (np.ndarray): 3D numpy array for the image volume.
        label_volumes_dir (str): Directory containing multiple 3D label volumes.
        patch_size (tuple): Size of the patches (default is (128, 128, 128)).
        stride (int): Step size for sliding window (default is 128).
        save_dir (str): Directory to save the extracted patches.

    Returns:
        None
    """
    # Create output directories
    image_save_dir = os.path.join(save_dir, "images")
    label_save_dir = os.path.join(save_dir, "labels")
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)

    # Get all label volumes from the directory
    label_files = [f for f in os.listdir(label_volumes_dir) if f.endswith(".npy")]

    patch_count = 0

    for label_file in label_files:
        # Load the label volume
        label_volume_path = os.path.join(label_volumes_dir, label_file)
        label_volume = np.load(label_volume_path)

        if image_volume.shape != label_volume.shape:
            raise ValueError(f"Image volume and label volume {label_file} must have the same dimensions.")

        x, y, z = image_volume.shape
        px, py, pz = patch_size

        # Pad volumes to ensure patches fit
        pad_x = (0, (stride - x % stride) % stride) if x % stride != 0 else (0, 0)
        pad_y = (0, (stride - y % stride) % stride) if y % stride != 0 else (0, 0)
        pad_z = (0, (stride - z % stride) % stride) if z % stride != 0 else (0, 0)

        padded_image_volume = np.pad(image_volume, (pad_x, pad_y, pad_z), mode="constant", constant_values=0)
        padded_label_volume = np.pad(label_volume, (pad_x, pad_y, pad_z), mode="constant", constant_values=0)

        # Extract patches
        for i in range(0, padded_image_volume.shape[0] - px + 1, stride):
            for j in range(0, padded_image_volume.shape[1] - py + 1, stride):
                for k in range(0, padded_image_volume.shape[2] - pz + 1, stride):
                    image_patch = padded_image_volume[i:i + px, j:j + py, k:k + pz]
                    label_patch = padded_label_volume[i:i + px, j:j + py, k:k + pz]

                    # Save patches with consistent numbering
                    patch_name = f"{patch_count}.npy"
                    image_patch_path = os.path.join(image_save_dir, patch_name)
                    label_patch_path = os.path.join(label_save_dir, patch_name)

                    np.save(image_patch_path, image_patch)
                    np.save(label_patch_path, label_patch)

                    patch_count += 1

    print(f"Saved {patch_count} patches in {save_dir}.")

# Example usage
if __name__ == "__main__":
    # Generate dummy 3D image volume for demonstration
    image_volume = np.load("/home/prithwijit/FAULTSEG/data/data/image_volume.npy")

    # Directory containing dummy label volumes
    label_volumes_dir = "/home/prithwijit/FAULTSEG/data/volumes/practitioners"

    create_sliding_patches_with_multiple_labels(image_volume, label_volumes_dir, patch_size=(128, 128, 128), stride=128, save_dir="/home/prithwijit/FAULTSEG/data/patches/practitioner")
