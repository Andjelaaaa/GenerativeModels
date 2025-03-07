# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -

# # 3D Latent Diffusion Model
# In this tutorial, we will walk through the process of using the MONAI Generative Models package to generate synthetic data using Latent Diffusion Models (LDM)  [1, 2]. Specifically, we will focus on training an LDM to create synthetic brain images from the Brats dataset.
#
# [1] - Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
#
# [2] - Pinaya et al. "Brain imaging generation with latent diffusion models" https://arxiv.org/abs/2209.07162

# ### Set up imports

# +
import os
import torch
import wandb
from datetime import datetime
import matplotlib.pyplot as plt
from torch.nn import L1Loss
from tqdm import tqdm
import nibabel as nib

# === MONAI/generative imports
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRangePercentilesd,
    CropForegroundd,
    Resized
)
from monai.data import DataLoader
from monai.utils import set_determinism
import numpy as np

# LDM code
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import LatentDiffusionInferer

# === Your custom dataset code (BCP/CP/Combined) ===
# Make sure these classes/functions are defined or imported
# from your own script. For brevity, we place them inline here.
import pandas as pd
import glob
from torch.utils.data import Dataset

def save_and_log_image(epoch, images, reconstruction, subject_id, session_id, save_dir):
    """Save slices as a PNG and log to Weights & Biases (W&B) with subject/session info."""
    idx = 0  # Use the first sample in batch
    input_np = images[idx, 0].detach().cpu().numpy()
    rec_np = reconstruction[idx, 0].detach().cpu().numpy()
    
    # Pick three z-slices to visualize
    zdim = input_np.shape[-1]
    slices_to_show = [zdim // 4, zdim // 2, 3 * zdim // 4]
    
    row_images = []
    for slice_idx in slices_to_show:
        input_slice = input_np[..., slice_idx]
        rec_slice = rec_np[..., slice_idx]
        side_by_side = np.hstack([input_slice, rec_slice])
        row_images.append(side_by_side)
    
    # Stack slices vertically
    final_image = np.vstack(row_images)
    
    # Create save directory if not exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the image as PNG
    save_path = os.path.join(save_dir, f"epoch-{epoch}_sub-{subject_id}_ses-{session_id}.png")
    plt.imsave(save_path, final_image, cmap='gray')
    
    # Log to W&B
    wandb.log({"comparison_input_recon": wandb.Image(save_path, caption=f"Epoch {epoch} - {subject_id} - {session_id}")})
    
    print(f"Saved and logged image: {save_path}")

# === END: dataset definitions ===
# for reproducibility purposes set a seed
set_determinism(42)

# # ### Setup a data directory and download dataset
# # Specify a MONAI_DATA_DIRECTORY variable, where the data will be downloaded. If not specified a temporary directory will be used.

# # directory = os.environ.get("MONAI_DATA_DIRECTORY")
# # root_dir = tempfile.mkdtemp() if directory is None else directory
# # print(root_dir)

# # ### Prepare data loader for the training set
# # Here we will download the Brats dataset using MONAI's `DecathlonDataset` class, and we prepare the data loader for the training set.

# # +
# # batch_size = 2
# # channel = 0  # 0 = Flair
# # assert channel in [0, 1, 2, 3], "Choose a valid channel"

# # train_transforms = transforms.Compose(
# #     [
# #         transforms.LoadImaged(keys=["image"]),
# #         transforms.EnsureChannelFirstd(keys=["image"]),
# #         transforms.Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
# #         transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
# #         transforms.EnsureTyped(keys=["image"]),
# #         transforms.Orientationd(keys=["image"], axcodes="RAS"),
# #         transforms.Spacingd(keys=["image"], pixdim=(2.4, 2.4, 2.2), mode=("bilinear")),
# #         transforms.CenterSpatialCropd(keys=["image"], roi_size=(96, 96, 64)),
# #         transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
# #     ]
# # )
# # train_ds = DecathlonDataset(
# #     root_dir=root_dir,
# #     task="Task01_BrainTumour",
# #     section="training",  # validation
# #     cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
# #     num_workers=8,
# #     download=True,  # Set download to True if the dataset hasnt been downloaded yet
# #     seed=0,
# #     transform=train_transforms,
# # )
# # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
# # print(f'Image shape {train_ds[0]["image"].shape}')
# # -

# # ### Visualise examples from the training set

# # +
# # Plot axial, coronal and sagittal slices of a training sample
# # check_data = first(train_loader)
# # idx = 0

# # img = check_data["image"][idx, 0]
# # fig, axs = plt.subplots(nrows=1, ncols=3)
# # for ax in axs:
# #     ax.axis("off")
# # ax = axs[0]
# # ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
# # ax = axs[1]
# # ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
# # ax = axs[2]
# # ax.imshow(img[img.shape[0] // 2, ...], cmap="gray")
# # plt.savefig("training_examples.png")
# # -

# --------------------------------------------------
# 1) Prepare dataset / dataloader
# --------------------------------------------------
# Example usage: you can choose if you want to load just BCP or a combined dataset.
use_combined = True  # set True if you want to combine two datasets

if use_combined:
    # Example: combine BCP and CP
    print('Using combined dataset')
    bcp_dataset = BCPDataset(tsv_path="/home/andim/projects/def-bedelb/andim/hc-bcp/participants.tsv")
    cp_dataset = CPDataset(tsv_path="/home/andim/projects/def-bedelb/andim/hc-calgary-preschool/participants.tsv")
    
    combined_dataset = CombinedDataset(bcp_dataset, cp_dataset) 
    train_loader = DataLoader(combined_dataset, batch_size=3, shuffle=True, num_workers=8)
else:
    # BCP only
    bcp_dataset = BCPDataset(tsv_path="/home/andim/projects/def-bedelb/andim/hc-bcp/participants.tsv")
    train_loader = DataLoader(bcp_dataset, batch_size=3, shuffle=True, num_workers=8)

# Print size for sanity check
print("Dataset length:", len(train_loader.dataset))

# ## Autoencoder KL
#
# ### Define Autoencoder KL network
#
# In this section, we will define an autoencoder with KL-regularization for the LDM. The autoencoder's primary purpose is to transform input images into a latent representation that the diffusion model will subsequently learn. By doing so, we can decrease the computational resources required to train the diffusion component, making this approach suitable for learning high-resolution medical images.
#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# +
    # autoencoder = AutoencoderKL(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=1,
    #     num_channels=(32, 64, 64),
    #     latent_channels=3,
    #     num_res_blocks=1,
    #     norm_num_groups=16,
    #     attention_levels=(False, False, True),
    # )
autoencoder = AutoencoderKL(spatial_dims=3, 
                                in_channels=1, 
                                out_channels=1, 
                                latent_channels=3,
                                num_channels=(64, 128, 128, 128),
                                num_res_blocks=2, 
                                norm_num_groups=32,
                                norm_eps=1e-06,
                                attention_levels=(False, False, False, False), 
                                with_decoder_nonlocal_attn=False, 
                                with_encoder_nonlocal_attn=False)
autoencoder.to(device)


discriminator = PatchDiscriminator(spatial_dims=3, num_layers_d=3, num_channels=32, in_channels=1, out_channels=1)
discriminator.to(device)
# -

# ### Defining Losses
#
# We will also specify the perceptual and adversarial losses, including the involved networks, and the optimizers to use during the training process.

# +
l1_loss = L1Loss()
adv_loss = PatchAdversarialLoss(criterion="least_squares")
loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
loss_perceptual.to(device)


def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
    return torch.sum(kl_loss) / kl_loss.shape[0]


adv_weight = 0.01
perceptual_weight = 0.001
kl_weight = 1e-6
# -

optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=1e-4)
optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4)

adv_weight = 0.01
perceptual_weight = 0.001
kl_weight = 1e-6
autoencoder_warm_up_n_epochs = 5

# --------------------------------------------------
# 3) Initialize wandb
# --------------------------------------------------
today_str = datetime.now().strftime("%Y%m%d_%H%M%S") 
exp_name = 'exp_vq_vae'
scratch_dir = os.environ.get("SCRATCH", "/scratch")  # Use $SCRATCH environment variable
original_root_dir = os.path.join(scratch_dir, 'COMBINED', exp_name)
wandb_dir = os.path.join(original_root_dir, "wandb_logs")
os.makedirs(wandb_dir, exist_ok=True)
wandb.init(
    project="MONAI_3D_LDM-VQ-VAE", 
    name=f"experiment-{today_str}",
    mode = 'offline',
    dir=wandb_dir,
    config={
        "batch_size": 3,
        "lr": 1e-4,
        "adv_weight": adv_weight,
        "perceptual_weight": perceptual_weight,
        "kl_weight": kl_weight,
    }
)
config = wandb.config

# --------------------------------------------------
# 4) Training loop
# --------------------------------------------------
n_epochs = 100
epoch_recon_loss_list = []
epoch_gen_loss_list = []
epoch_disc_loss_list = []

# Initialize best reconstruction loss to a very high value
best_recon_loss = float("inf")

for epoch in range(n_epochs):
    autoencoder.train()
    discriminator.train()

    recon_loss_sum = 0.0
    gen_loss_sum = 0.0
    disc_loss_sum = 0.0
    step_count = 0

    pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{n_epochs}]", ncols=120)
    for batch in pbar:
        images = batch["data"].to(device)  # shape [B, 1, X, Y, Z]
        subject_id = batch["subject_id"][0]  # Assuming batch size > 0
        session_id = batch["session_id"][0]

        # ----- Generator / Autoencoder step -----
        optimizer_g.zero_grad(set_to_none=True)
        reconstruction, z_mu, z_sigma = autoencoder(images)
        kl = KL_loss(z_mu, z_sigma)
        recons_loss = l1_loss(reconstruction, images)
        p_loss = loss_perceptual(reconstruction, images)
        g_loss = recons_loss + kl_weight * kl + perceptual_weight * p_loss

        if epoch > autoencoder_warm_up_n_epochs:
            # Adversarial component
            logits_fake = discriminator(reconstruction.float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            g_loss += adv_weight * generator_loss
        else:
            generator_loss = torch.tensor(0.0, device=device)

        g_loss.backward()
        optimizer_g.step()

        # ----- Discriminator step -----
        if epoch > autoencoder_warm_up_n_epochs:
            optimizer_d.zero_grad(set_to_none=True)
            # Fake
            logits_fake = discriminator(reconstruction.detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            # Real
            logits_real = discriminator(images)[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            d_loss = 0.5 * (loss_d_fake + loss_d_real)
            d_loss_total = adv_weight * d_loss
            d_loss_total.backward()
            optimizer_d.step()
        else:
            d_loss = torch.tensor(0.0, device=device)

        recon_loss_sum += recons_loss.item()
        gen_loss_sum += generator_loss.item()
        disc_loss_sum += d_loss.item()
        step_count += 1

        pbar.set_postfix({
            "recons_loss": recon_loss_sum / step_count,
            "gen_loss": gen_loss_sum / step_count,
            "disc_loss": disc_loss_sum / step_count
        })

    # -- compute epoch losses
    epoch_recon_loss = recon_loss_sum / max(step_count, 1)
    epoch_gen_loss = gen_loss_sum / max(step_count, 1)
    epoch_disc_loss = disc_loss_sum / max(step_count, 1)
    epoch_recon_loss_list.append(epoch_recon_loss)
    epoch_gen_loss_list.append(epoch_gen_loss)
    epoch_disc_loss_list.append(epoch_disc_loss)

    # -- wandb logging
    wandb.log({
        "epoch": epoch,
        "recons_loss": epoch_recon_loss,
        "gen_loss": epoch_gen_loss,
        "disc_loss": epoch_disc_loss,
    })

    scratch_dir = os.environ.get("SCRATCH", "/scratch")  # Use $SCRATCH environment variable
    original_root_dir = os.path.join(scratch_dir, 'COMBINED', exp_name)
    # fold_specific_dir = os.path.join(original_root_dir, f"fold_{fold_idx}")
    os.makedirs(original_root_dir, exist_ok=True)
    
    # Every 5 epochs, save a checkpoint
    if epoch % 5 == 0 and epoch > 0:
        # Build the filename path in that directory
        ckpt_filename = os.path.join(original_root_dir, f"autoencoder_epoch_{epoch}.pth")

        torch.save(autoencoder.state_dict(), ckpt_filename)
        
        # (Optional) If you want wandb to store this file too:
        wandb.save(ckpt_filename)

        # Save images 
        save_and_log_image(epoch, images, reconstruction, subject_id, session_id, original_root_dir)

    # Save best model based on lowest reconstruction loss
    if epoch_recon_loss < best_recon_loss:
        best_recon_loss = epoch_recon_loss  # Update best loss
        best_ckpt_filename = os.path.join(original_root_dir, "best.pth")
        torch.save(autoencoder.state_dict(), best_ckpt_filename)
        wandb.save(best_ckpt_filename)
        print(f"New best model saved with recon loss {best_recon_loss:.6f} at epoch {epoch}")


# --------------------------------------------------
# 5) Save final model(s)
# --------------------------------------------------
# fold_specific_dir = os.path.join(original_root_dir, f"fold_{fold_idx}")

# Build the filename path in that directory
ckpt_filename_final = os.path.join(original_root_dir, f"autoencoder_final.pth")
torch.save(autoencoder.state_dict(), ckpt_filename_final)
wandb.save("autoencoder_final.pth")

wandb.finish()
print("Training complete!")

# # ### Train model

# # +
# # n_epochs = 100
# # autoencoder_warm_up_n_epochs = 5
# # val_interval = 10
# # epoch_recon_loss_list = []
# # epoch_gen_loss_list = []
# # epoch_disc_loss_list = []
# # val_recon_epoch_loss_list = []
# # intermediary_images = []
# # n_example_images = 4

# # for epoch in range(n_epochs):
# #     autoencoder.train()
# #     discriminator.train()
# #     epoch_loss = 0
# #     gen_epoch_loss = 0
# #     disc_epoch_loss = 0
# #     progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
# #     progress_bar.set_description(f"Epoch {epoch}")
# #     for step, batch in progress_bar:
# #         images = batch["data"].to(device)  # choose only one of Brats channels

# #         # Generator part
# #         optimizer_g.zero_grad(set_to_none=True)
# #         reconstruction, z_mu, z_sigma = autoencoder(images)
# #         kl_loss = KL_loss(z_mu, z_sigma)

# #         recons_loss = l1_loss(reconstruction.float(), images.float())
# #         p_loss = loss_perceptual(reconstruction.float(), images.float())
# #         loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

# #         if epoch > autoencoder_warm_up_n_epochs:
# #             logits_fake = discriminator(reconstruction.contiguous().float())[-1]
# #             generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
# #             loss_g += adv_weight * generator_loss

# #         loss_g.backward()
# #         optimizer_g.step()

# #         if epoch > autoencoder_warm_up_n_epochs:
# #             # Discriminator part
# #             optimizer_d.zero_grad(set_to_none=True)
# #             logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
# #             loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
# #             logits_real = discriminator(images.contiguous().detach())[-1]
# #             loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
# #             discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

# #             loss_d = adv_weight * discriminator_loss

# #             loss_d.backward()
# #             optimizer_d.step()

# #         epoch_loss += recons_loss.item()
# #         if epoch > autoencoder_warm_up_n_epochs:
# #             gen_epoch_loss += generator_loss.item()
# #             disc_epoch_loss += discriminator_loss.item()

# #         progress_bar.set_postfix(
# #             {
# #                 "recons_loss": epoch_loss / (step + 1),
# #                 "gen_loss": gen_epoch_loss / (step + 1),
# #                 "disc_loss": disc_epoch_loss / (step + 1),
# #             }
# #         )
# #     epoch_recon_loss_list.append(epoch_loss / (step + 1))
# #     epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
# #     epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))

# # del discriminator
# # del loss_perceptual
# # torch.cuda.empty_cache()
# # # -

# # plt.style.use("ggplot")
# # plt.title("Learning Curves", fontsize=20)
# # plt.plot(epoch_recon_loss_list)
# # plt.yticks(fontsize=12)
# # plt.xticks(fontsize=12)
# # plt.xlabel("Epochs", fontsize=16)
# # plt.ylabel("Loss", fontsize=16)
# # plt.legend(prop={"size": 14})
# # plt.show()

# # plt.title("Adversarial Training Curves", fontsize=20)
# # plt.plot(epoch_gen_loss_list, color="C0", linewidth=2.0, label="Generator")
# # plt.plot(epoch_disc_loss_list, color="C1", linewidth=2.0, label="Discriminator")
# # plt.yticks(fontsize=12)
# # plt.xticks(fontsize=12)
# # plt.xlabel("Epochs", fontsize=16)
# # plt.ylabel("Loss", fontsize=16)
# # plt.legend(prop={"size": 14})
# # plt.show()

# # # ### Visualise reconstructions

# # # Plot axial, coronal and sagittal slices of a training sample
# # idx = 0
# # img = reconstruction[idx, channel].detach().cpu().numpy()
# # fig, axs = plt.subplots(nrows=1, ncols=3)
# # for ax in axs:
# #     ax.axis("off")
# # ax = axs[0]
# # ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
# # ax = axs[1]
# # ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
# # ax = axs[2]
# # ax.imshow(img[img.shape[0] // 2, ...], cmap="gray")

# # ## DIFFUSION MODEL
# #
# # ### Define diffusion model and scheduler
# #
# # In this section, we will define the diffusion model that will learn data distribution of the latent representation of the autoencoder. Together with the diffusion model, we define a beta scheduler responsible for defining the amount of noise tahat is added across the diffusion's model Markov chain.

# # +
# unet = DiffusionModelUNet(
#     spatial_dims=3,
#     in_channels=3,
#     out_channels=3,
#     num_res_blocks=1,
#     num_channels=(32, 64, 64),
#     attention_levels=(False, True, True),
#     num_head_channels=(0, 64, 64),
# )
# unet.to(device)


# scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)
# # -

# # ### Scaling factor
# #
# # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM, if the standard deviation of the latent space distribution drifts too much from that of a Gaussian. For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
# #
# # _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one, and the results will not differ from those obtained when it is not used._
# #

# # +
# with torch.no_grad():
#     with autocast(enabled=True):
#         z = autoencoder.encode_stage_2_inputs(check_data["image"].to(device))

# print(f"Scaling factor set to {1/torch.std(z)}")
# scale_factor = 1 / torch.std(z)
# # -

# # We define the inferer using the scale factor:

# inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

# optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=1e-4)

# # ### Train diffusion model

# # +
# n_epochs = 150
# epoch_loss_list = []
# autoencoder.eval()
# scaler = GradScaler()

# first_batch = first(train_loader)
# z = autoencoder.encode_stage_2_inputs(first_batch["image"].to(device))

# for epoch in range(n_epochs):
#     unet.train()
#     epoch_loss = 0
#     progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
#     progress_bar.set_description(f"Epoch {epoch}")
#     for step, batch in progress_bar:
#         images = batch["image"].to(device)
#         optimizer_diff.zero_grad(set_to_none=True)

#         with autocast(enabled=True):
#             # Generate random noise
#             noise = torch.randn_like(z).to(device)

#             # Create timesteps
#             timesteps = torch.randint(
#                 0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
#             ).long()

#             # Get model prediction
#             noise_pred = inferer(
#                 inputs=images, autoencoder_model=autoencoder, diffusion_model=unet, noise=noise, timesteps=timesteps
#             )

#             loss = F.mse_loss(noise_pred.float(), noise.float())

#         scaler.scale(loss).backward()
#         scaler.step(optimizer_diff)
#         scaler.update()

#         epoch_loss += loss.item()

#         progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
#     epoch_loss_list.append(epoch_loss / (step + 1))
# # -

# plt.plot(epoch_loss_list)
# plt.title("Learning Curves", fontsize=20)
# plt.plot(epoch_loss_list)
# plt.yticks(fontsize=12)
# plt.xticks(fontsize=12)
# plt.xlabel("Epochs", fontsize=16)
# plt.ylabel("Loss", fontsize=16)
# plt.legend(prop={"size": 14})
# plt.show()

# # ### Plotting sampling example
# #
# # Finally, we generate an image with our LDM. For that, we will initialize a latent representation with just noise. Then, we will use the `unet` to perform 1000 denoising steps. In the last step, we decode the latent representation and plot the sampled image.

# # +
# autoencoder.eval()
# unet.eval()

# noise = torch.randn((1, 3, 24, 24, 16))
# noise = noise.to(device)
# scheduler.set_timesteps(num_inference_steps=1000)
# synthetic_images = inferer.sample(
#     input_noise=noise, autoencoder_model=autoencoder, diffusion_model=unet, scheduler=scheduler
# )
# # -

# # ### Visualise synthetic data

# idx = 0
# img = synthetic_images[idx, channel].detach().cpu().numpy()  # images
# fig, axs = plt.subplots(nrows=1, ncols=3)
# for ax in axs:
#     ax.axis("off")
# ax = axs[0]
# ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
# ax = axs[1]
# ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
# ax = axs[2]
# ax.imshow(img[img.shape[0] // 2, ...], cmap="gray")

# # ## Clean-up data

# if directory is None:
#     shutil.rmtree(root_dir)
