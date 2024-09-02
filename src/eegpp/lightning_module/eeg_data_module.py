import os
from typing import Union

from lightning import LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler, Subset

from src.eegpp import params
from src.eegpp.data import DUMP_DATA_FILES
from src.eegpp.data.data_utils import dump_seq_with_labels
# from src.eegpp.data.data_utils import split_dataset
from src.eegpp.data.eeg_dataset import EEGDataset


# from src.eegpp import params


class EEGDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size=8,
            num_workers=0,
            dataset_file_idx: Union[list[int], str] = 'all',
            n_splits=5
    ):
        super().__init__()
        self.n_splits = n_splits  # number of split datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_file_idx = dataset_file_idx
        self.k = None  # KFold id

        self.datasets = None
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.predict_dataset = None
        self.all_splits = None
        self.train_ids = None
        self.val_ids = None

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
        # train_dts, val_dts, test_dts = [[], [], []]
        datasets = []
        if isinstance(self.dataset_file_idx, list):
            for idx in self.dataset_file_idx:
                dump_file = DUMP_DATA_FILES['train'][idx]
                print("Loading dump file {}".format(dump_file))
                i_dataset = EEGDataset(dump_file, window_size=params.W_OUT)
                datasets.append(i_dataset)
                # train_set, val_set, test_set = split_dataset(i_dataset)
                # train_dts.append(train_set)
                # val_dts.append(val_set)
                # test_dts.append(test_set)

        else:
            for dump_file in DUMP_DATA_FILES['train']:
                print("Loading dump file {}".format(dump_file))
                i_dataset = EEGDataset(dump_file, window_size=params.W_OUT)
                datasets.append(i_dataset)
                # train_set, val_set, test_set = split_dataset(i_dataset)
                # train_dts.append(train_set)
                # val_dts.append(val_set)
                # test_dts.append(test_set)
        # self.datasets = ConcatDataset(datasets)
        self.datasets = datasets
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=params.RD_SEED)
        # self.all_splits = [subset for subset in kf.split(datasets)]
        all_splits = []
        for dataset in datasets:
            splits = [subset for subset in kf.split(dataset)]
            all_splits.append(splits)
        self.all_splits = all_splits

        # if stage == 'fit' or stage is None:
        #     self.train_dataset = ConcatDataset(train_dts)
        #     self.val_dataset = ConcatDataset(val_dts)
        # elif stage == 'test' or stage == 'predict':
        #     self.test_dataset = ConcatDataset(test_dts)

    def setup_train_val_k(self, k):
        if self.datasets is None or self.all_splits is None:
            self.setup()
        self.k = k
        ret = self._check_k()
        if ret:
            train_dts, val_dts = [[], []]
            for i, splits in enumerate(self.all_splits):
                train_ids, val_ids = splits[k]
                train_dt = Subset(self.datasets[i], train_ids)
                val_dt = Subset(self.datasets[i], val_ids)
                train_dts.append(train_dt)
                val_dts.append(val_dt)
            self.train_dataset = ConcatDataset(train_dts)
            self.val_dataset = ConcatDataset(val_dts)

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
        pass

    def predict_dataloader(self):
        pass

    def _check_k(self):
        if self.k is None or self.k < 0 or self.k >= self.n_splits:
            print(f"k must be in [0, {self.n_splits})")
            return False
        return True
