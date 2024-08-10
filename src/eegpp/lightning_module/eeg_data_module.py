import os
import shutil
from pathlib import Path

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.eegpp.data import DATA_DIR
from src.eegpp.data.data_utils import split_dataset_into_dirs
from src.eegpp.data.eeg_dataset import EEGDataset


# from src.eegpp import params


class EEGDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size=8,
            num_workers=1,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None

    def prepare_data(self):
        dir_names = ['train', 'val', 'test']
        os.makedirs(str(Path(DATA_DIR) / 'eeg'), exist_ok=True)
        # download_data_from_ggdrive()
        reset_dataset = False
        for dir_name in dir_names:
            if not os.path.exists(str(Path(DATA_DIR) / f'eeg/{dir_name}')):
                print(f'{dir_name} dataset does not exist. Set reset dataset to True')
                reset_dataset = True
                break
        if reset_dataset:
            print('Train-Val-Test (TVT) dataset is incomplete. Force create new TVT dataset')
            for dir_name in dir_names:
                shutil.rmtree(str(Path(DATA_DIR) / f'eeg/{dir_name}'))
                os.makedirs(str(Path(DATA_DIR) / f'eeg/{dir_name}'), exist_ok=True)
            split_dataset_into_dirs()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = EEGDataset()
            self.val_dataset = EEGDataset()
        elif stage == 'test':
            self.test_dataset = EEGDataset()

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
