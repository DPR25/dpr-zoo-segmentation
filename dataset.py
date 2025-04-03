import os
import csv
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch.utils.data import Dataset
import cv2

class SatlasSentinel2Dataset(Dataset):
    CATEGORY_MAP = {
        0: "background",
        1: "water",
        2: "developed",
        3: "tree",
        4: "shrub",
        5: "grass",
        6: "crop",
        7: "bare",
        8: "snow",
        9: "wetland",
        10: "mangroves",
        11: "moss"
    }

    def __init__(self, csv_file, bands=["tci"], normalize=True, gammacorr=False, brighten=False, 
                 alpha=0.13, beta=0, gamma=2, filter_categories=None):
        """
        Args:
            csv_file (str): Path to the CSV file containing image-mask paths.
            bands (list): List of bands to load (e.g., ["tci", "b5", "b6"]).
            normalize (bool): Apply min-max normalization.
            gammacorr (bool): Apply gamma correction.
            brighten (bool): Apply brightness adjustment.
            alpha (float): Brightening multiplier.
            beta (float): Brightening offset.
            gamma (float): Gamma correction factor.
            filter_categories (list): List of category names to keep (e.g., ["tree", "water"]).
        """
        self.bands = bands
        self.normalize_flag = normalize
        self.gammacorr_flag = gammacorr
        self.brighten_flag = brighten
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.filter_categories = filter_categories
        self.valid_classes = self.get_valid_classes()
        self.data = self.load_and_filter_csv(csv_file)

    def get_valid_classes(self):
        """Get numeric class values for filtering, based on category names."""
        if self.filter_categories is None:
            return set(self.CATEGORY_MAP.keys())  # Allow all categories
        return {k for k, v in self.CATEGORY_MAP.items() if v in self.filter_categories}

    def load_and_filter_csv(self, csv_file):
        """Loads image-mask pairs from CSV and removes entries with no relevant categories."""
        valid_data = []

        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # Read the header
            for row in reader:
                image_path, label_path, *flags = row

                # Convert category flags to booleans
                flags = list(map(lambda x: x == "True", flags))

                # Filter by categories
                if self.valid_classes and not any(idx for idx in self.valid_classes):
                    continue  # Skip if none of the required categories exist

                valid_data.append((image_path, label_path))

        print(f"Loaded {len(valid_data)} valid image-label pairs.")
        return valid_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path = self.data[idx]

        # Load image bands
        image = self.load_bands(img_path)

        # Load segmentation mask with filtering
        mask = self.load_mask(mask_path)

        return image, mask

    def load_bands(self, image_path):
        """Loads selected bands and stacks them into a numpy array using OpenCV."""
        
        # Check if "tci" is in the list of bands
        has_tci = "tci" in self.bands

        # Path for sample band to get image size (assumes the first band exists)
        sample_band_path = image_path.replace("band", self.bands[0])
        
        # Read the first band to get the dimensions of the image (height, width)
        sample_img = cv2.imread(sample_band_path, cv2.IMREAD_UNCHANGED)
        height, width = sample_img.shape[:2]

        # If "tci" is in the bands, we need 3 additional channels
        num_channels = len(self.bands)-1 + 3 if has_tci else len(self.bands)

        # Initialize an empty array to store the stacked bands (C, H, W)
        stacked_bands = np.zeros((num_channels, height, width), dtype=np.float32)

        channel_idx = 0 

        if has_tci:
            tci_band_path = image_path.replace("band", "tci")
            
            # Load the TCI band as a 3-channel RGB image
            tci_data = cv2.imread(tci_band_path, cv2.IMREAD_COLOR)
            tci_data = cv2.cvtColor(tci_data, cv2.COLOR_BGR2RGB)
            if tci_data is None:
                raise FileNotFoundError(f"TCI Band file {tci_band_path} not found.")

            # Apply optional preprocessing (if flags are set)
            if self.brighten_flag:
                tci_data = self.brighten(tci_data)
            if self.gammacorr_flag:
                tci_data = self.gammacorr(tci_data)
            if self.normalize_flag:
                tci_data = self.normalize(tci_data)

            stacked_bands[channel_idx] = tci_data[:, :, 0]  # Red channel
            stacked_bands[channel_idx + 1] = tci_data[:, :, 1]  # Green channel
            stacked_bands[channel_idx + 2] = tci_data[:, :, 2]  # Blue channel

            channel_idx += 3

        # Now load the rest of the bands
        for band in self.bands:
            if band == "tci":
                continue  # Skip if we already loaded "tci"

            band_path = image_path.replace("band", band)
            
            if not os.path.exists(band_path):
                raise FileNotFoundError(f"Band file {band_path} not found.")

            # Load other bands as single-channel images
            band_data = cv2.imread(band_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

            # Apply optional preprocessing (if flags are set)
            if self.brighten_flag:
                band_data = self.brighten(band_data)
            if self.gammacorr_flag:
                band_data = self.gammacorr(band_data)
            if self.normalize_flag:
                band_data = self.normalize(band_data)

            # Assign the single band data to the correct channel in the stacked_bands
            stacked_bands[channel_idx] = band_data
            channel_idx += 1  # Move to the next channel in the stacked_bands array

        return stacked_bands  # Shape: (C, H, W)

    def load_mask(self, mask_path):
        """Loads the segmentation mask using OpenCV."""
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file {mask_path} not found.")

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        return mask.astype(np.uint8)

    def brighten(self, band):
        """Adjusts brightness using alpha and beta values."""
        return np.clip(self.alpha * (band/255) + self.beta, 0, 1)

    def gammacorr(self, band):
        """Applies gamma correction."""
        return np.power(band, 1 / self.gamma)

    def normalize(self, band):
        """Performs min-max normalization."""
        _min, _max = band.min(), band.max()
        return (band - _min) / (_max - _min) if _min != _max else np.zeros(band.shape)

    def visualize_sample(self, idx):
        """Visualizes an image and its corresponding label mask."""
        image, mask = self.__getitem__(idx)

        # Convert image to RGB (Assuming first 3 bands are RGB-like)
        if image.shape[0] >= 3:
            rgb_image = np.stack([image[0], image[1], image[2]], axis=-1)
            rgb_image = np.clip(rgb_image, 0, 1)
        else:
            rgb_image = image[0]  # Grayscale fallback

        # Define a color map for the mask
        category_colors = [
            (0, 0, 0), # unknown
            (0, 0, 1), # (blue) water
            (1, 0, 0), # (red) developed
            (0, 192/255, 0), # (dark green) tree
            (200/255, 170/255, 120/255), # (brown) shrub
            (0, 1, 0), # (green) grass
            (1, 1, 0), # (yellow) crop
            (128/255, 128/255, 128/255), # (grey) bare
            (1, 1, 1), # (white) snow
            (0, 1, 1), # (cyan) wetland
            (1, 0, 1), # (pink) mangroves
            (128/255, 0, 128/255), # (purple) moss
        ]

        # Create a custom ListedColormap
        mask_cmap = ListedColormap(category_colors)

        # Define the boundaries between classes
        bounds = np.arange(len(category_colors) + 1)

        # Create a norm to map values to colors
        norm = BoundaryNorm(bounds, mask_cmap.N)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(rgb_image)
        axes[0].set_title("Satellite Image")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap=mask_cmap, norm=norm)
        axes[1].set_title("Segmentation Mask")
        axes[1].axis("off")

        legend_patches = [Patch(color=category_colors[i], label=category) for i, category in self.CATEGORY_MAP.items()]

        plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), title="Categories")

        plt.show()

#if __name__ == "__main__":
#    ds = SatlasSentinel2Dataset("dataset_links.csv")
#    ds.visualize_sample(12)