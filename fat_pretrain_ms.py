import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset2 import SatlasSentinel2DatasetV2
import satlaspretrain_models

def modify_input_bands(model, selected_bands):
    """
    Modify the first conv layer of the model to accept a new set of bands (channels).

    Args:
        model (nn.Module): The model with a .backbone.features[0][0] Conv2d layer.
        selected_bands (list of str): List of bands to retain (e.g. ["b4", "b3", "b2", "b08"]).
    """

    BAND_ORDER = ["b04", "b03", "b02", "b05", "b06", "b07", "b08", "b11", "b12"]

    # Validate bands
    for band in selected_bands:
        if band not in BAND_ORDER:
            raise ValueError(f"Band '{band}' is not in known band order: {BAND_ORDER}")

    # Get indices of selected bands
    band_indices = [BAND_ORDER.index(band) for band in selected_bands]

    # Locate the first conv layer
    first_conv = model.backbone.backbone.features[0][0]
    old_weights = first_conv.weight.data  # shape: (out_channels, in_channels, kH, kW)

    # Create new Conv2d layer
    new_conv = torch.nn.Conv2d(
        in_channels=len(selected_bands),
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None
    )

    # Copy selected channel weights
    new_conv.weight.data = old_weights[:, band_indices, :, :].clone()

    # Copy bias if exists
    if first_conv.bias is not None:
        new_conv.bias.data = first_conv.bias.data.clone()

    # Replace conv layer in model
    model.backbone.backbone.features[0][0] = new_conv

    print("Modified input multi-scale bands.")

def fat_pretrain(model, loader, epochs, model_name, device='cuda', grad_accum_steps=1, use_scheduler=True):
    """
    Trains a segmentation model with optional gradient accumulation and scheduler.
    
    Args:
        model (torch.nn.Module): The model.
        loader (DataLoader): Training data loader.
        epochs (int): Number of epochs.
        model_name (str): For logging/checkpointing.
        device (str): Device to use.
        grad_accum_steps (int): Accumulate gradients over this many steps.
        scheduler (optional): Learning rate scheduler.
    """
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('loss_plots', exist_ok=True)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6) if use_scheduler else None

    # Training loop
    print(f"Initiating {model_name}'s training session ...")
    for epoch in range(epochs):
        model.train()
        progress_bar = tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}')
        running_loss = 0
        batch_losses = []

        optimizer.zero_grad()

        for step, (inputs, masks) in enumerate(progress_bar):
            inputs = inputs.float().to(device)
            masks = masks.squeeze(1).long().to(device)

            outputs, loss = model(inputs, masks)
            loss = loss / grad_accum_steps  # Normalize for accumulation
            loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1 == len(loader)):
                print(loss)
                optimizer.step()
                optimizer.zero_grad()

            loss_value = loss.item() * grad_accum_steps  # true unnormalized loss
            batch_losses.append(loss_value)
            running_loss += loss_value

            del inputs, masks, outputs, loss

        avg_loss = running_loss / len(loader)
        print(f'Epoch {epoch+1} Avg Loss: {avg_loss:.4f}')

        if scheduler:
            scheduler.step()

        # Save loss curve
        plt.figure()
        plt.plot(batch_losses)
        plt.title(f'Epoch {epoch+1} Loss Curve')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.savefig(f'loss_plots/{model_name}_epoch{epoch+1}.png')
        plt.close()

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoints/{model_name}_epoch{epoch+1}.pth')

    print(f"{model_name}'s training session has concluded.")


def main():

    EPOCHS = 100
    GRAD_ACCUM_STEPS = 3
    BATCH_SIZE = 16

    # Change bands as needed
    selected_bands = ["tci", "b08"]
    modified_input_bands = ["b04", "b03", "b02", "b08"]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')

    ds = SatlasSentinel2DatasetV2("output.csv", bands=selected_bands)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    print('Loading model ...')
    weights_manager = satlaspretrain_models.Weights()
    model = weights_manager.get_pretrained_model("Sentinel2_SwinB_SI_MS", fpn=True, head=satlaspretrain_models.Head.SEGMENT, num_categories=12, device='cuda')

    modify_input_bands(model, selected_bands=modified_input_bands)

    model.to(device)

    fat_pretrain(model, loader, EPOCHS, f'Satlas_MS_{"-".join(selected_bands)}', 'cuda', grad_accum_steps=GRAD_ACCUM_STEPS, use_scheduler=True)

if __name__ == '__main__':
    main()