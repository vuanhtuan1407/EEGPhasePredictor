from jinja2.compiler import F
from lightning import LightningModule
from torch.optim import AdamW


class EEGModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()

    def forward(self, x):
        return x

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def base_step(self, batch, batch_idx):
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def on_test_epoch_end(self) -> None:
        pass

    def on_test_end(self) -> None:
        pass
