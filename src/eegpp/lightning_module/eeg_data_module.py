from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.eegpp.data.eeg_dataset import EEGDataset


# from src.eegpp import params


class EEGDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size=8,
            num_workers=1,
            combine_all_datasets=False,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.combine_all_datasets = combine_all_datasets

        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.predict_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = EEGDataset()
            self.val_dataset = EEGDataset()
        elif stage == 'test':
            self.test_dataset = EEGDataset()
        elif stage == "predict":
            self.predict_dataset = EEGDataset(is_infer=True)

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
