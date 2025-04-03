import gc
import os
import torch
import zipfile
import requests
import torch.nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import satlaspretrain_models

class AmazonRainforestDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the folder containing .tif images
            mask_dir (str): Path to the folder containing .tif segmentation masks
            transform (callable, optional): Optional transformations for data augmentation
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Load image (GeoTIFF)
        with rasterio.open(img_path) as img_file:
            image = img_file.read()
            image = image[:3, :, :]


        # Load mask (GeoTIFF)
        with rasterio.open(mask_path) as mask_file:
            mask = mask_file.read(1, out_shape=(mask_file.height, mask_file.width))
        
        # Normalize image (0-1)
        image = (image.astype(np.float32) / 255.0) * 0.1
        image = np.clip(image, 0, 1)

        # Convert mask to binary (0 for background, 1 for forest)
        mask = (mask > 0).astype(np.float32)

        # Apply transformations if provided
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask

def plot_sample(image, mask, figsize=(12, 6)):
    plt.figure(figsize=figsize)
    
    # Ensure image is in (H, W, C) format for plotting
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    if image.shape[0] == 3 and len(image.shape) == 3:  # If image is in (C, H, W) format
        image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C)
    
    # For mask visualization
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy()
    
    # Handle different mask formats
    if len(mask.shape) == 3:
        if mask.shape[0] == 1:  # Single channel mask (1, H, W)
            mask = mask.squeeze(0)
        elif mask.shape[0] == 2:  # Two-channel output - common for binary segmentation
            # Take the second channel (class 1 - foreground)
            mask = mask[1]  # Or use argmax: mask = np.argmax(mask, axis=0)
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)  # Now image is in (H, W, C) format
    plt.title("RGB Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Forest Mask/Prediction")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()


def train_segmentation_model():

    device = 'cuda'

    weights_manager = satlaspretrain_models.Weights()
    model = weights_manager.get_pretrained_model("Sentinel2_SwinB_SI_RGB", fpn=True, head=satlaspretrain_models.Head.SEGMENT, num_categories=2, device='cuda')
    
    train_dataset = AmazonRainforestDataset("data/AMAZON/Training/image", "data/AMAZON/Training/label")
    val_dataset = AmazonRainforestDataset("data/AMAZON/Validation/images", "data/AMAZON/Validation/masks")

    model = model.to(device)

    num_epochs = 1
    criterion = torch.nn.CrossEntropyLoss()
    val_step = 1  # evaluate every val_step epochs
    save_steps = 1

    save_path = 'weights/'  # where to save model weights
    os.makedirs(save_path, exist_ok=True)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    batch_size = 2
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Training loop.
    for epoch in range(num_epochs):
        print("Starting Epoch...", epoch)

        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            output, loss = model(data, target)
            print("Train Loss = ", loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation.
        if epoch % val_step == 0:
            model.eval()

            for val_data, val_target in val_loader:
                val_data = val_data.to(device)
                val_target = val_target.to(device)

                val_output, val_loss = model(val_data, val_target)

                val_accuracy = (val_output.argmax(dim=1) == val_target).float().mean().item()
                print("Validation accuracy = ", val_accuracy)
    
    train_loader = None
    val_loader = None
    train_dataset = None
    val_dataset = None

    del train_dataset
    del train_loader
    del val_dataset
    del val_loader
    gc.collect()
    torch.cuda.empty_cache()

    torch.save(model.state_dict(), save_path + str(epoch) + '_model_weights.pth')



def predict_segmentation():

    device = 'cuda'

    weights_manager = satlaspretrain_models.Weights()
    recent_weights = torch.load('weights/0_model_weights.pth')
    model = weights_manager.get_pretrained_model("Sentinel2_SwinB_SI_RGB", fpn=True, head=satlaspretrain_models.Head.SEGMENT, num_categories=2, device='cuda')
    model.load_state_dict(recent_weights)
    model = model.to(device)

    test_dataset = AmazonRainforestDataset("data/AMAZON/Test/image", "data/AMAZON/Test/mask")
    image, mask = test_dataset[1]

    model.eval()
    output, loss = model(torch.from_numpy(image).unsqueeze(0).to(device))

    plot_sample(image, output[0])
    plot_sample(image, mask)

def predict_segmentation_single_image(image_path):

    device = 'cuda'
    model_path = torch.load('weights/0_model_weights.pth')


    # Load and preprocess the image
    image = Image.open(image_path)
    image_np = np.array(image) / 255.0  # Normalize to 0-1

    # Convert to (C, H, W) format if needed
    if image_np.shape[2] == 3:  # If image is in (H, W, C) format
        image_np = np.transpose(image_np, (2, 0, 1))  # Convert to (C, H, W)

    weights_manager = satlaspretrain_models.Weights()
    model = weights_manager.get_pretrained_model(
        "Sentinel2_SwinB_SI_RGB", 
        fpn=True, 
        head=satlaspretrain_models.Head.SEGMENT, 
        num_categories=2, 
        device=device
    )
    model.load_state_dict(model_path)
    model = model.to(device)
    model.eval()

    image_tensor = torch.from_numpy(image_np).float().unsqueeze(0).to(device)
    
    # Run prediction
    with torch.no_grad():
        output, _ = model(image_tensor)

    # Process the output for visualization
    pred_mask = output[0].cpu().detach().numpy()

    # For binary segmentation, take the second channel (forest probability) or use argmax
    if pred_mask.shape[0] == 2:
        forest_prob = pred_mask[1]  # Forest probability map
        pred_binary = (forest_prob > 0.5).astype(np.float32)  # Binary prediction using threshold
    else:
        pred_binary = (pred_mask > 0.5).astype(np.float32)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(np.transpose(image_np, (1, 2, 0)))  # Convert back to (H, W, C) for display
    plt.title("Original Image")
    plt.axis("off")
    
    # Probability map
    plt.subplot(1, 3, 2)
    if pred_mask.shape[0] == 2:
        plt.imshow(forest_prob, cmap='viridis')
        plt.colorbar(label='Forest Probability')
    else:
        plt.imshow(pred_mask.squeeze(), cmap='viridis')
        plt.colorbar(label='Probability')
    plt.title("Forest Probability")
    plt.axis("off")
    
    # Binary prediction
    plt.subplot(1, 3, 3)
    plt.imshow(pred_binary, cmap='gray')
    plt.title("Binary Forest Mask (threshold=0.5)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

predict_segmentation_single_image("data/test2.png")