import os
import shutil
import numpy as np
import rasterio
import torch
import matplotlib.pyplot as plt


def delete_folders_without_land_cover(parent_folder):

    if not os.path.isdir(parent_folder):
        print(f"Error: {parent_folder} does not exist.")
        return
     
    # Traverse all subdirectories of the parent folder
    for root, dirs, files in os.walk(parent_folder, topdown=False):
        for dir_name in dirs:
            # Construct full path to the directory
            dir_path = os.path.join(root, dir_name)
            
            # Check if the directory contains the 'land_cover.png' file
            if 'land_cover.png' not in os.listdir(dir_path):
                print(f"Removing: {dir_path}")
                # Delete the directory if it doesn't contain 'land_cover.png'
                shutil.rmtree(dir_path)

def load_segment_target(fname):
    # Ensure the file exists and is a .png or .tif
    if not os.path.exists(fname):
        raise FileNotFoundError(f"File {fname} not found.")

    # Read the segmentation mask using rasterio
    with rasterio.open(fname) as dataset:
        # Read the image data (assuming it's a single channel mask)
        im = dataset.read(1)  # read the first band, segmentation masks are usually single-band
        
        # Check if the image is 2D (height, width) for a segmentation mask
        if len(im.shape) != 2:
            raise ValueError("Segmentation mask must be a 2D array (height, width).")

    # Print out unique values in the segmentation mask
    unique_values = np.unique(im)
    print(f"Unique values in the segmentation mask: {unique_values}")
    categories = ['background', 'water', 'developed', 'tree', 'shrub', 'grass', 'crop', 'bare', 'snow', 'wetland', 'mangroves', 'moss']

    # Convert mask to a tensor
    im_tensor = torch.as_tensor(im, dtype=torch.int32)
    
    # Optionally, create a valid mask (valid = 1 for foreground, 0 for background)
    valid_im = torch.ones(im.shape, dtype=torch.bool)  # Initially, mark all as valid

    # Visualize the segmentation mask using matplotlib
    plt.imshow(im, cmap='gray')
    plt.title('Segmentation Mask')
    plt.colorbar()
    plt.show()

    return {
        'valid': torch.tensor(1, dtype=torch.int32),
        'valid_im': valid_im,
        'im': im_tensor,
    }

#if __name__ == "__main__":
#    load_segment_target("test_images/land_cover.png")
#    parent_folder = "./satlaspretrain_dataset_labels_static_0000/static/"  # Replace with your folder path
#    delete_folders_without_land_cover(parent_folder)