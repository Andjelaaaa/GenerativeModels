from torch.utils.data import Dataset

class CombinedDataset(Dataset):
    """
    Combine two separate Dataset objects into a single dataset.
    """
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.len1 = len(self.dataset1)
        self.len2 = len(self.dataset2)

    def __len__(self):
        return self.len1 + self.len2

    def __getitem__(self, idx):
        if idx < self.len1:
            return self.dataset1[idx]
        else:
            return self.dataset2[idx - self.len1]