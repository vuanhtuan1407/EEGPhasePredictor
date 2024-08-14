import joblib
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, dump_path: str):
        # data = (start_datetime, eeg, emg, mot, [lbs], mxs)
        data = joblib.load(dump_path)
        if len(data) == 6:
            self.start_datetime, self.eeg, self.emg, self.mot, self.lbs, self.mxs = data
        else:
            self.start_datetime, self.eeg, self.emg, self.mot, self.mxs = data
            self.lbs = []

    def __len__(self):
        return len(self.start_datetime)

    def __getitem__(self, idx):
        pass
