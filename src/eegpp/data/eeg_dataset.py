import joblib
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
        if len(self.lbs) != 0:
            return self.eeg[idx] / self.mxs[0], self.emg[idx] / self.mxs[1], self.mot[idx] / self.mxs[2], self.lbs[idx]
        else:
            return self.eeg[idx] / self.mxs[0], self.emg[idx] / self.mxs[1], self.mot[idx] / self.mxs[2]
