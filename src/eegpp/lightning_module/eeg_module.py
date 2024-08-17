import torch
from lightning import LightningModule
from torch.optim import AdamW, SGD
from torchmetrics.classification import MulticlassF1Score
from src.eegpp import params

from src.eegpp.data.data_utils import LABEL_DICT
from src.eegpp.models.get_model import get_model


class EEGModule(LightningModule):
    def __init__(
            self,
            model_type,
            lr=1e-5,
            optimizer_type='SGD',
            loss_fn_type='CrossEntropyLoss',
    ):
        super().__init__()
        # self.save_hyperparameters()

        self.hparams.optimizer_type = optimizer_type
        self.hparams.lr = lr
        self.hparams.loss_fn_type = loss_fn_type
        self.hparams.model_type = model_type

        self.model = get_model(model_type)
        self.loss_fn = self.configure_loss_fn()
        self.f1score = MulticlassF1Score(num_classes=len(LABEL_DICT))

    def forward(self, x):
        return self.model(x)

    def configure_loss_fn(self):
        if self.hparams.loss_fn_type == 'MSELoss':
            loss_fn = torch.nn.MSELoss()
        else:
            loss_weight = None
            loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight)
        return loss_fn

    def configure_optimizers(self):
        if self.hparams.optimizer_type == 'AdamW':
            optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr)
        else:
            optimizer = SGD(self.model.parameters(), lr=self.hparams.lr)
        return optimizer

    def compute_loss(self, pred, true, window_size=params.W_OUT):
        return self.loss_fn(pred, true)

    def base_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.compute_loss(pred, y)
        print(x.shape, '\n', y.shape, '\n', loss.shape)
        return x, y, pred, loss

    def training_step(self, batch, batch_idx):
        x, y, pred, loss = self.base_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, pred, loss = self.base_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, lb, pred, loss = self.base_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def on_test_epoch_end(self) -> None:
        pass

    def on_test_end(self) -> None:
        pass
