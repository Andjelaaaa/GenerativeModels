import os
import matplotlib.pyplot as plt
import numpy as np

from datasets import BCPDataset  # or from wherever your dataset classes are
from monai import transforms

def extract_middle_slices(image_3d):
    """
    image_3d: NumPy array with shape [D, H, W] (or [C, D, H, W] if you squeeze one channel).
    Returns a dict with 'axial', 'coronal', 'sagittal' slices at the center.
    """
    D, H, W = image_3d.shape

    mid_d = D // 2
    mid_h = H // 2
    mid_w = W // 2

    # Axial = fix 'D', so [H, W]
    axial_slice = image_3d[mid_d, :, :]

    # Coronal = fix 'H', so [D, W], then transpose to [W, D] if you prefer
    coronal_slice = image_3d[:, mid_h, :]

    # Sagittal = fix 'W', so [D, H]
    sagittal_slice = image_3d[:, :, mid_w]

    return {
        "axial": axial_slice,
        "coronal": coronal_slice,
        "sagittal": sagittal_slice
    }

def main():
    # Example: define a custom transform pipeline for testing
    test_transforms_1 = Compose([
        transforms.LoadImaged(keys="data"),
        transforms.EnsureChannelFirstd(keys="data"),
        transforms.Orientationd(keys="data", axcodes="RAS"),
        transforms.Spacingd(keys="data", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        transforms.CropForegroundd(keys="data", source_key="data", select_fn=lambda x: x>0),
        transforms.Resized(keys="data", spatial_size=(64, 128, 128), mode=["area"]),
        transforms.ScaleIntensityRangePercentilesd(keys="data", lower=0.0, upper=99.5, b_min=0.0, b_max=1.0, clip=True)
    ])

    
    test_transforms_2 = transforms.Compose([
        transforms.LoadImageD(image_only=True, keys=['data']),
        transforms.EnsureChannelFirstD(keys=['data']), 
        transforms.SpacingD(pixdim=1.5, keys=['data']),
        transforms.ResizeWithPadOrCropD(spatial_size=(120, 144, 120), mode='minimum', keys=['data']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['data'])
    ])

    # 3) Build the combined dataset *records*, but do NOT pass transforms yet
    #    so we can apply them in separate dataset objects below.
    print("Building combined dataset records WITHOUT transforms (for debug).")
    bcp_dataset_no_xform = BCPDataset(
        tsv_path="/home/andim/projects/def-bedelb/andim/hc-bcp/participants.tsv",
        transform=None  # we handle transforms separately
    )
    cp_dataset_no_xform = CPDataset(
        tsv_path="/home/andim/projects/def-bedelb/andim/hc-calgary-preschool/participants.tsv",
        transform=None
    )
    combined_no_xform = CombinedDataset(bcp_dataset_no_xform, cp_dataset_no_xform)

    # We'll test only the first few records
    num_to_test = 3
    indices_to_test = list(range(num_to_test))
    # Alternatively, you could pick random indices

    # 4) Create two separate dataset objects, each referencing the SAME data_dicts but different transforms
    #    We do this by copying the .data_dicts from combined_no_xform and applying new transforms in __getitem__.
    class CombinedDatasetWithTransform(CombinedDataset):
        def __init__(self, base_dataset, transform):
            # base_dataset is the combined_no_xform
            # We keep its data_dicts, but we override the transform
            self.dataset1 = base_dataset.dataset1
            self.dataset2 = base_dataset.dataset2
            self.len1 = self.dataset1.__len__()
            self.len2 = self.dataset2.__len__()
            self.transform = transform

        def __getitem__(self, idx):
            if idx < self.len1:
                data_item = self.dataset1.data_dicts[idx]
            else:
                data_item = self.dataset2.data_dicts[idx - self.len1]

            # apply transform here
            if self.transform is not None:
                return self.transform(data_item)
            return data_item

    # Make dataset for transform_1
    combined_xform_1 = CombinedDatasetWithTransform(combined_no_xform, test_transforms_1)
    # Make dataset for transform_2
    combined_xform_2 = CombinedDatasetWithTransform(combined_no_xform, test_transforms_2)

    # 5) Evaluate dataset #1, save slices
    output_dir_1 = "transform_debug_images_v1"
    os.makedirs(output_dir_1, exist_ok=True)
    for i in indices_to_test:
        sample = combined_xform_1[i]
        data_tensor = sample["data"]  # shape [C, D, H, W]
        subject_id = sample["subject_id"]
        session_id = sample["session_id"]

        data_np = data_tensor.numpy().squeeze()  # shape [D, H, W]
        slices_dict = extract_middle_slices(data_np)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        for ax_i, plane in enumerate(["axial", "coronal", "sagittal"]):
            axs[ax_i].imshow(slices_dict[plane], cmap="gray")
            axs[ax_i].axis("off")
            axs[ax_i].set_title(plane.capitalize())
        fig.suptitle(f"[v1] subj={subject_id}, sess={session_id}\n shape={data_np.shape}")
        save_name = f"sample_{i}_{subject_id}_{session_id}_v1.png"
        plt.savefig(os.path.join(output_dir_1, save_name), bbox_inches="tight")
        plt.close(fig)

    # 6) Evaluate dataset #2, save slices
    output_dir_2 = "transform_debug_images_v2"
    os.makedirs(output_dir_2, exist_ok=True)
    for i in indices_to_test:
        sample = combined_xform_2[i]
        data_tensor = sample["data"]
        subject_id = sample["subject_id"]
        session_id = sample["session_id"]

        data_np = data_tensor.numpy().squeeze()
        slices_dict = extract_middle_slices(data_np)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        for ax_i, plane in enumerate(["axial", "coronal", "sagittal"]):
            axs[ax_i].imshow(slices_dict[plane], cmap="gray")
            axs[ax_i].axis("off")
            axs[ax_i].set_title(plane.capitalize())
        fig.suptitle(f"[v2] subj={subject_id}, sess={session_id}\n shape={data_np.shape}")
        save_name = f"sample_{i}_{subject_id}_{session_id}_v2.png"
        plt.savefig(os.path.join(output_dir_2, save_name), bbox_inches="tight")
        plt.close(fig)

    print(f"Done! Check {output_dir_1} and {output_dir_2} for outputs.")

if __name__ == "__main__":
    main()