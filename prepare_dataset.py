import os
import csv
import argparse
import rasterio
import numpy as np

# Example usage:
# python prepare_dataset.py --root_dataset_folder "data/satlaspretrain_dataset_sentinel2_a_0001/sentinel2" --root_label_folder "/data" --output_csv "dataset_links.csv"

def load_segment_target(fname):
    """
    Loads a segmentation mask using rasterio from a .png or .tif file,
    prints unique values, and returns the unique values along with the image tensor.
    """
    if not os.path.exists(fname):
        raise FileNotFoundError(f"File {fname} not found.")

    with rasterio.open(fname) as dataset:
        # Read the first band (segmentation masks are assumed to be single-channel)
        im = dataset.read(1)
        if len(im.shape) != 2:
            raise ValueError("Segmentation mask must be a 2D array (height, width).")

    unique_vals = np.unique(im)
    print(f"Unique values in {fname}: {unique_vals}")
    del im
    return unique_vals

def find_label_folders(root_label_folder):
    """
    Automatically finds all label folders inside a specified directory
    that start with 'satlaspretrain_dataset_labels_static'.
    """
    if not os.path.exists(root_label_folder):
        raise FileNotFoundError(f"Label folder root '{root_label_folder}' does not exist!")

    label_folders = [
        os.path.join(root_label_folder, d) for d in os.listdir(root_label_folder)
        if d.startswith("satlaspretrain_dataset_labels_static") and os.path.isdir(os.path.join(root_label_folder, d))
    ]
    return label_folders

def main():
    parser = argparse.ArgumentParser(
        description="Generate CSV linking dataset images with label mask paths and category flags."
    )
    parser.add_argument("--root_dataset_folder", type=str, required=True,
                        help="Root folder for dataset images (e.g., satlaspretrain_dataset_sentinel2_a_0001/sentinel2)")
    parser.add_argument("--root_label_folder", type=str, required=True,
                        help="Root folder containing label folders (e.g., a directory containing 'satlaspretrain_dataset_labels_static_XXXX')")
    parser.add_argument("--output_csv", type=str, default="output.csv", help="Output CSV file path")
    args = parser.parse_args()

    # Find label folders inside the specified root label folder
    label_folders = find_label_folders(args.root_label_folder)
    if not label_folders:
        print("No label folders found! Make sure they exist in the specified root directory.")
        return

    print(f"Found label folders: {label_folders}")

    # Define the fixed categories (index corresponds to pixel value)
    categories = ['background', 'water', 'developed', 'tree', 'shrub', 'grass', 'crop', 'bare', 'snow', 'wetland', 'mangroves', 'moss']

    csv_rows = []
    header = ["image_path", "label_path"] + categories

    # Iterate through each S2A folder in the dataset images root
    for s2a_folder in os.listdir(args.root_dataset_folder):
        s2a_path = os.path.join(args.root_dataset_folder, s2a_folder)
        if not os.path.isdir(s2a_path):
            continue

        # Look for the 'tci' folder inside this S2A folder
        tci_folder = os.path.join(s2a_path, "tci")
        if not os.path.isdir(tci_folder):
            continue

        # Collect all PNG filenames from the tci folder
        for file in os.listdir(tci_folder):
            if not file.lower().endswith(".png"):
                continue
            # Get image ID (filename without extension)
            image_id = os.path.splitext(file)[0]

            # Build the image path for CSV:
            image_path = os.path.join(args.root_dataset_folder, s2a_folder, "band", f"{image_id}.png")

            # Search for a corresponding label folder in the specified label root
            label_path = None
            for lab_root in label_folders:
                candidate_folder = os.path.join(lab_root, "static", image_id)
                candidate_label = os.path.join(candidate_folder, "land_cover.png")
                if os.path.isdir(candidate_folder) and os.path.exists(candidate_label):
                    label_path = candidate_label
                    break  # use the first found match

            # If no corresponding label found, skip this image
            if label_path is None:
                continue

            # Load the segmentation mask to get unique values
            try:
                unique_vals = load_segment_target(label_path)
            except Exception as e:
                print(f"Error loading label {label_path}: {e}")
                continue

            # Create binary flags for each category based on whether the category index appears in unique_vals
            flags = ["True" if idx in unique_vals else "False" for idx in range(len(categories))]

            # Create CSV row: image_path, label_path, then category flags
            row = [image_path, label_path] + flags
            csv_rows.append(row)

    # Write CSV file
    with open(args.output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(csv_rows)

    print(f"CSV file saved to {args.output_csv}")

if __name__ == "__main__":
    main()