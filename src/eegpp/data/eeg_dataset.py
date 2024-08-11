from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, is_infer=False):
        self.data = []
        self.is_infer = is_infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
