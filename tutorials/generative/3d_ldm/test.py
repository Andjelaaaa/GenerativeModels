import os
import torch
from torch.utils.data import DataLoader, random_split
import nibabel as nib
import numpy as np
import argparse

# Example: Adjust these imports to your actual code structure
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from datasets.datasets import BCPDataset, CPDataset
from datasets.combined_dataset import CombinedDataset

def save_nifti_reconstruction(
    input_tensor: torch.Tensor,
    recon_tensor: torch.Tensor,
    subject_id: str,
    session_id: str,
    save_dir: str
):
    """
    Saves the (input_tensor, recon_tensor) as NIfTI files, naming them with subject/session.
    Both are shape [B, C, D, H, W]. We only save the first sample [0, 0].
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert to NumPy
    input_np = input_tensor[0, 0].cpu().numpy()  # [D, H, W]
    recon_np = recon_tensor[0, 0].cpu().numpy()  # [D, H, W]

    # Identity affine (if you have an original affine, you can store that instead)
    input_nii = nib.Nifti1Image(input_np, affine=np.eye(4))
    recon_nii = nib.Nifti1Image(recon_np, affine=np.eye(4))

    # Filenames
    input_path = os.path.join(save_dir, f"{subject_id}_{session_id}_input_network.nii.gz")
    recon_path = os.path.join(save_dir, f"{subject_id}_{session_id}_reconstructed_network.nii.gz")

    nib.save(input_nii, input_path)
    nib.save(recon_nii, recon_path)

    print(f"Saved input NIfTI:       {input_path}")
    print(f"Saved reconstruction:   {recon_path}")

def reconstruct_combined_dataset(
    exp_name: str,
    dataset_name: str,
    fold_idx: int,
    checkpoint_path: str,
    scratch_dir: str,
    dataset_base_path: str,
    fraction: float = 0.1,
    batch_size: int = 1,
    device: str = "cuda:0",
):
    """
    Reconstructs ~10% of images from the combined dataset (BCP + CP) using an AutoencoderKL checkpoint.
    Saves input & recon NIfTI in the directory:
      {scratch_dir}/evaluation/{dataset_name}/{exp_name}/fold_{fold_idx}_reconstructions

    Args:
        exp_name (str): Name of your experiment (e.g., "exp_vq_vae").
        dataset_name (str): Name of dataset folder (e.g., "COMBINED").
        fold_idx (int): Fold index (0, 1, etc.).
        checkpoint_path (str): Path to the model .pth checkpoint.
        scratch_dir (str): HPC scratch or base output directory.
        dataset_base_path (str): Directory containing BCP/CP data.
        fraction (float): Fraction of dataset to reconstruct (0.1 => 10%).
        batch_size (int): Batch size for inference.
        device (str): "cuda:0" or "cpu".

    Returns:
        None. Writes NIfTI files to disk.
    """
    # ------------------------------------------------------------------
    # 1) Prepare output directory
    # ------------------------------------------------------------------
    save_path = os.path.join(
        scratch_dir,
        "evaluation",
        dataset_name,
        exp_name,
        f"fold_{fold_idx}_reconstructions"
    )
    os.makedirs(save_path, exist_ok=True)
    print(f"Will save reconstructions to: {save_path}")

    # ------------------------------------------------------------------
    # 2) Build/Load Model
    # ------------------------------------------------------------------
    # model = AutoencoderKL(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=1,
    #     num_channels=(32, 64, 64),  # Must match training!
    #     latent_channels=3,
    #     num_res_blocks=1,
    #     norm_num_groups=16,
    #     attention_levels=(False, False, True),
    # )
    model = AutoencoderKL(spatial_dims=3, 
                                in_channels=1, 
                                out_channels=1, 
                                latent_channels=3,
                                num_channels=(64, 128, 128, 128),
                                num_res_blocks=1, 
                                norm_num_groups=16,
                                norm_eps=1e-06,     
                                attention_levels=(False, False, False, False), 
                                with_decoder_nonlocal_attn=False, 
                                with_encoder_nonlocal_attn=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 3) Build Combined Dataset & Subset (10%)
    # ------------------------------------------------------------------
    bcp_tsv = os.path.join(dataset_base_path, "hc-bcp/participants.tsv")
    cp_tsv  = os.path.join(dataset_base_path, "hc-calgary-preschool/participants.tsv")

    bcp_ds = BCPDataset(tsv_path=bcp_tsv)
    cp_ds  = CPDataset(tsv_path=cp_tsv)
    full_dataset = CombinedDataset(bcp_ds, cp_ds)

    # We'll only reconstruct fraction=10%
    subset_len = int(len(full_dataset) * fraction)
    remainder_len = len(full_dataset) - subset_len

    # random_split shuffles by default
    subset_dataset, _ = random_split(full_dataset, [subset_len, remainder_len])
    print(f"Full combined dataset size: {len(full_dataset)}")
    print(f"Using subset of size:       {subset_len} (~{fraction*100:.1f}%)")

    dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ------------------------------------------------------------------
    # 4) Inference Loop
    # ------------------------------------------------------------------
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # batch has "data", "subject_id", "session_id"
            images = batch["data"].to(device)  # [B, 1, D, H, W]
            sub_id = batch["subject_id"][0]
            ses_id = batch["session_id"][0]

            # Reconstruct
            recon = model.reconstruct(images)  # [B, 1, D, H, W]

            print(f"[{i}] sub={sub_id}, ses={ses_id}")
            print(f"    Input shape: {images.shape}, Recon shape: {recon.shape}")

            # Save input & recon
            save_nifti_reconstruction(
                input_tensor=images,
                recon_tensor=recon,
                subject_id=sub_id,
                session_id=ses_id,
                save_dir=save_path
            )

    print("Done! Network-space reconstructions saved in:")
    print(f"  {save_path}")

# Example usage if calling this directly:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reconstructions with trained model.")
    parser.add_argument("--exp_name", required=True, type=str, help="Experiment name")
    parser.add_argument("--dataset_name", default="COMBINED", type=str, help="Dataset name")
    # parser.add_argument("--fold_idx", default=0, type=int, help="Fold index")
    # parser.add_argument("--checkpoint_path", required=True, type=str, help="Path to model checkpoint (.pth)")
    # parser.add_argument("--scratch_dir", required=True, type=str, help="Base directory for saving results")
    # parser.add_argument("--dataset_base_path", required=True, type=str, help="Path to the dataset root")
    # parser.add_argument("--fraction", default=0.1, type=float, help="Fraction of dataset to reconstruct")
    # parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    # parser.add_argument("--device", default="cuda:0", type=str, help="Device for inference")

    args = parser.parse_args()

    exp_name = args.exp_name
    dataset_name = args.dataset_name
    # reconstruct_combined_dataset(
    #     exp_name=args.exp_name,
    #     dataset_name=args.dataset_name,
    #     fold_idx=args.fold_idx,
    #     checkpoint_path=args.checkpoint_path,
    #     scratch_dir=args.scratch_dir,
    #     dataset_base_path=args.dataset_base_path,
    #     fraction=args.fraction,
    #     batch_size=args.batch_size,
    #     device=args.device
    # )
    reconstruct_combined_dataset(
        exp_name=args.exp_name,
        dataset_name=args.dataset_name,
        fold_idx=1,
        # checkpoint_path="/home/andim/scratch/COMBINED/exp_vq_vae_fold0/autoencoder_epoch_65.pth",
        checkpoint_path=f"/home/andim/scratch/{dataset_name}/{exp_name}/best.pth",
        scratch_dir="/home/andim/scratch",
        dataset_base_path="/home/andim/projects/def-bedelb/andim",
        fraction=0.1,       # 10%
        batch_size=1,
        device="cuda:0"
    )
