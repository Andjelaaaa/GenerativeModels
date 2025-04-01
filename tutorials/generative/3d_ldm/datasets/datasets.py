import os
import pandas as pd
import glob
import torch
from torch.utils.data import Dataset

# === MONAI/generative imports
from monai import transforms
from monai.data import DataLoader

def create_bcp_records(tsv_path: str):
    """
    Parse the participants.tsv to collect NIfTI file paths plus subject/session info,
    but only for sessions older than ~12 months.
    
    Returns a list of dicts, each with:
      {
        "data": "/path/to/some_stripped.nii.gz",
        "subject_id": "sub-XXX",
        "session_id": "ses-YYY",
      }
    """
    # Helper function to parse sessions like 'ses-9mo', 'ses-2wk' into approximate months
    def session_to_months(session_str):
        # remove leading 'ses-'
        s = session_str[4:]  # e.g. '12mo', '2wk'
        # extract the numeric portion and the trailing alpha (mo or wk)
        import re
        match = re.match(r'(\d+)([a-zA-Z]+)', s)
        if match:
            num = int(match.group(1))
            unit = match.group(2)
            if unit == 'mo':
                return num
            elif unit == 'wk':
                # ~4.345 weeks in a month on average; 
                return num / 4.345
        return None

    participants = pd.read_csv(tsv_path, sep="\t")
    data_dicts = []

    base_dir = os.path.dirname(tsv_path)
    runs_to_include_pth = os.path.join(base_dir, 'runs_to_include.csv')
    runs_to_include = pd.read_csv(runs_to_include_pth)

    for _, row in participants.iterrows():
        subject_id = row["participant_id"]
        all_sessions = row["sessions"].split(",")

        # Filter for sessions that are >= 12 months
        sessions_older_than_12 = [
            sess.strip()
            for sess in all_sessions
            if session_to_months(sess.strip()) and session_to_months(sess.strip()) >= 12
        ]

        # If subject is in the runs_to_include list...
        if subject_id in runs_to_include['participant_id'].to_list():
            # Loop over each session that is >= 12 months
            for session_id in sessions_older_than_12:
                # Check if that session is also in runs_to_include
                if session_id in runs_to_include[
                    runs_to_include['participant_id'] == subject_id
                ]['session_id'].to_list():
                    anat_dir = os.path.join(
                        base_dir,
                        f"n4_bias_correction/{subject_id}/{session_id}/anat/"
                    )
                    if not os.path.exists(anat_dir):
                        print(f"Missing directory: {anat_dir}")
                        continue

                    stripped_files = glob.glob(os.path.join(anat_dir, "*_T1w_stripped_n4.nii.gz"))
                    if not stripped_files:
                        print(f"No stripped files found in {anat_dir}")
                        continue

                    # Sort so that if multiple runs exist, we pick the "latest"
                    latest_file = sorted(
                        stripped_files,
                        key=lambda x: int(x.split("_run-")[1].split("_")[0])
                        if "_run-" in x else 0
                    )[-1]

                    data_dicts.append({
                        "data": latest_file,
                        "subject_id": subject_id,
                        "session_id": session_id
                    })

    print(f"Found {len(data_dicts)} valid entries in {tsv_path}")
    return data_dicts


# def create_bcp_records(tsv_path: str):
#     """
#     Parse the participants.tsv to collect NIfTI file paths plus subject/session info.
#     Returns a list of dicts, each with:
#       {
#         "image": "/path/to/some_stripped.nii.gz",
#         "subject_id": "sub-XXX",
#         "session_id": "ses-YYY",
#       }
#     """
#     participants = pd.read_csv(tsv_path, sep="\t")

#     data_dicts = []
#     base_dir = os.path.dirname(tsv_path)
#     runs_to_include_pth = os.path.join(base_dir, 'runs_to_include.csv')
    
#     runs_to_include = pd.read_csv(runs_to_include_pth)

#     for _, row in participants.iterrows():
#         subject_id = row["participant_id"]
#         # Handle multiple sessions if the cell is comma-separated
#         sessions = row["sessions"].split(",")
#         # print(subject_id)
#         # print(sessions)
#         if subject_id in runs_to_include['participant_id'].to_list():
#             for session in sessions:
#                 session_id = session.strip()
#                 if session_id in runs_to_include[runs_to_include['participant_id']==subject_id]['session_id'].to_list():
#                     anat_dir = os.path.join(
#                         base_dir,
#                         f"n4_bias_correction/{subject_id}/{session_id}/anat/"
#                     )
#                     if not os.path.exists(anat_dir):
#                         print(f"Missing directory: {anat_dir}")
#                         continue

#                     # Collect all stripped T1 files
#                     stripped_files = glob.glob(os.path.join(anat_dir, "*_T1w_stripped_n4.nii.gz"))
#                     if not stripped_files:
#                         print(f"No stripped files found in {anat_dir}")
#                         continue

#                     # Choose the latest run (if run is used in naming)
#                     latest_file = sorted(
#                         stripped_files,
#                         key=lambda x: int(x.split("_run-")[1].split("_")[0])
#                         if "_run-" in x else 0
#                     )[-1]

#                     data_dicts.append({
#                         "data": latest_file,
#                         "subject_id": subject_id,
#                         "session_id": session_id
#                     })

#     print(f"Found {len(data_dicts)} valid entries in {tsv_path}")
#     return data_dicts

def create_cp_records(tsv_path: str):
    """
    Parse the participants.tsv to collect NIfTI file paths plus subject/session info.
    Returns a list of dicts, each with:
      {
        "image": "/path/to/some_stripped_n4.nii.gz",
        "subject_id": "sub-XXX",
        "session_id": "ses-YYY",
      }
    """
    participants = pd.read_csv(tsv_path, sep="\t")

    data_dicts = []
    base_dir = os.path.dirname(tsv_path)
    for _, row in participants.iterrows():
        subject_id = row["participant_id"]
        # Handle multiple sessions if the cell is comma-separated
        sessions = row["sessions"].split(",")

        for session in sessions:
            session_id = session.strip()
            anat_dir = os.path.join(
                base_dir,
                f"n4_bias_correction/{subject_id}/{session_id}/anat/"
            )
            if not os.path.exists(anat_dir):
                print(f"Missing directory: {anat_dir}")
                continue

            # Collect all stripped T1 files
            stripped_files = glob.glob(os.path.join(anat_dir, "*_T1w_stripped_n4.nii.gz"))
            if not stripped_files:
                print(f"No stripped files found in {anat_dir}")
                continue

            # Choose the latest run (if run is used in naming)
            latest_file = sorted(
                stripped_files,
                key=lambda x: int(x.split("_run-")[1].split("_")[0])
                if "_run-" in x else 0
            )[-1]

            data_dicts.append({
                "data": latest_file,
                "subject_id": subject_id,
                "session_id": session_id
            })

    print(f"Found {len(data_dicts)} valid entries in {tsv_path}")
    return data_dicts

def threshold_at_zero(x):
    return x > 0

# RESOLUTION = 1.5
# INPUT_SHAPE_AE = (120, 144, 120)
# transforms_fn = transforms.Compose([
#         transforms.CopyItemsD(keys={'image_path'}, names=['image']),
#         transforms.LoadImageD(image_only=True, keys=['image']),
#         transforms.EnsureChannelFirstD(keys=['image']), 
#         transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
#         transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
#         transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
#     ])

# def get_monai_transforms(final_size=(64, 128, 128)):
#     return Compose([
#         LoadImaged(keys="data"),
#         EnsureChannelFirstd(keys="data"),
#         Orientationd(keys="data", axcodes="RAS"),
#         Spacingd(keys="data", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
#         CropForegroundd(keys="data", source_key="data", select_fn=threshold_at_zero),
#         Resized(keys="data", spatial_size=final_size, mode=["area"]),
#         ScaleIntensityRangePercentilesd(keys="data", lower=0, upper=99.5, b_min=0, b_max=1),
#     ])

def get_monai_transforms():
    return transforms.Compose([
        transforms.LoadImaged(keys="data"),
        transforms.EnsureChannelFirstD(keys=['data']),
        transforms.Orientationd(keys="data", axcodes="RAS"),
        transforms.SpacingD(pixdim=1, keys=['data']),
        transforms.CropForegroundd(keys="data", source_key="data", select_fn=lambda x: x>0, margin=20),
        transforms.ResizeWithPadOrCropD(spatial_size=(160, 160, 168), mode='minimum', keys=['data']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['data']),
    ])

class BCPDataset(Dataset):
    """
    Simple BCP dataset that yields a dict {"data": Tensor, "subject_id": str, "session_id": str}.
    """
    def __init__(self, tsv_path, transform=None):
        super().__init__()
        self.data_dicts = create_bcp_records(tsv_path)
        self.transform = transform if transform else get_monai_transforms()

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx):
        data_item = self.data_dicts[idx]
        output = self.transform(data_item)
        # Check for NaNs in the output tensor
        assert not torch.isnan(output["data"]).any(), f"NaN values found in 'data' for subject {data_item['subject_id']}"
        return self.transform(self.data_dicts[idx])
    
class CPDataset(Dataset):
    """
    Simple BCP dataset that yields a dict {"data": Tensor, "subject_id": str, "session_id": str}.
    """
    def __init__(self, tsv_path, transform=None):
        super().__init__()
        self.data_dicts = create_cp_records(tsv_path)
        self.transform = transform if transform else get_monai_transforms()

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx):
        data_item = self.data_dicts[idx]
        output = self.transform(data_item)
        # Check for NaNs in the output tensor
        assert not torch.isnan(output["data"]).any(), f"NaN values found in 'data' for subject {data_item['subject_id']}"
        return self.transform(self.data_dicts[idx])