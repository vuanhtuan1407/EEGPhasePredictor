import os
from typing import Union

from lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset

from src.eegpp.data import DUMP_DATA_FILES
from src.eegpp.data.data_utils import dump_seq_with_labels
from src.eegpp.data.data_utils import split_dataset
from src.eegpp.data.eeg_dataset import EEGDataset
from src.eegpp import params


# from src.eegpp import params


class EEGDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size=8,
            num_workers=1,
            dataset_file_idx: Union[list[int], str] = 'all',
    ):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_file_idx = dataset_file_idx

        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.predict_dataset = None

    def prepare_data(self):
        force_dump = False
        for dump_file in DUMP_DATA_FILES['train']:
            if not os.path.exists(dump_file):
                print('Cannot find dump file {}. Set force re-dump dats'.format(dump_file))
                force_dump = True
                break

        if force_dump:
            dump_seq_with_labels()

    def setup(self, stage=None):
        train_dts, val_dts, test_dts = [[], [], []]
        if isinstance(self.dataset_file_idx, list):
            for idx in self.dataset_file_idx:
                dump_file = DUMP_DATA_FILES['train'][idx]
                print("Loading dump file {}".format(dump_file))
                i_dataset = EEGDataset(dump_file, window_size=params.W_OUT)
                train_set, val_set, test_set = split_dataset(i_dataset)
                train_dts.append(train_set)
                val_dts.append(val_set)
                test_dts.append(test_set)

        else:
            for dump_file in DUMP_DATA_FILES['train']:
                print("Loading dump file {}".format(dump_file))
                i_dataset = EEGDataset(dump_file, window_size=params.W_OUT)
                train_set, val_set, test_set = split_dataset(i_dataset)
                train_dts.append(train_set)
                val_dts.append(val_set)
                test_dts.append(test_set)

        if stage == 'fit' or stage is None:
            self.train_dataset = ConcatDataset(train_dts)
            self.val_dataset = ConcatDataset(val_dts)
        elif stage == 'test' or stage == 'predict':
            self.test_dataset = ConcatDataset(test_dts)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        pass
