from typing import Literal

import joblib
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, dump_path: str, window_size=3, contain_side: Literal['left', 'right', 'both', 'none'] = 'both',
                 is_infer=False):
        """
        EEG Dataset
        :param dump_path:
        :param window_size: must be an odd if contain == 'both'
        :param contain_side:
        """
        # data = (start_datetime, eeg, emg, mot, [lbs], mxs)
        self.is_infer = is_infer
        self.window_size = window_size
        self.contain_side = contain_side
        if not is_infer:
            self.start_datetime, self.eeg, self.emg, self.mot, self.lbs, self.mxs = joblib.load(dump_path)
        else:
            self.start_datetime, self.eeg, self.emg, self.mot, self.mxs = joblib.load(dump_path)
            self.lbs = []

    def __len__(self):
        return len(self.start_datetime)

    def __getitem__(self, idx):
        seqs = []
        lbs = []
        if self.contain_side == 'none':
            seqs.append(self._getseq_idx(idx))
            lbs.append(self._getlb_idx(idx))
            seqs = torch.stack(seqs)
        else:
            if self.contain_side == 'right':
                for i in range(int(idx), int(idx + self.window_size) + 1):
                    seqs.append(self._getseq_idx(i))
                    lbs.append(self._getlb_idx(i))
            elif self.contain_side == 'left':
                for i in range(int(idx - self.window_size), int(idx) + 1):
                    seqs.append(self._getseq_idx(i))
                    lbs.append(self._getlb_idx(i))
            elif self.contain_side == 'both':
                for i in range(int(idx - self.window_size / 2), int(idx + self.window_size / 2) + 1):
                    seqs.append(self._getseq_idx(i))
                    lbs.append(self._getlb_idx(i))
            seqs = torch.concat(seqs, dim=-1)
        return seqs, lbs

    def _getseq_idx(self, idx):
        print(idx)
        if idx < 0 or idx >= self.__len__():
            seq = [0.0, 0.0, 0.0]
        else:
            seq = [self.eeg[idx] / self.mxs[0], self.emg[idx] / self.mxs[1], self.mot[idx] / self.mxs[2]]
        return torch.tensor(seq, dtype=torch.float32)

    def _getlb_idx(self, idx):
        lb = torch.zeros(self.window_size, dtype=torch.float32)
        if not self.is_infer:
            if idx < 0 or idx >= self.__len__():
                lb_idx = -1
            else:
                lb_idx = self.lbs[idx]
            lb[lb_idx] = 1.0
            return lb
        else:
            for v in lb:
                v = -1
            return lb
