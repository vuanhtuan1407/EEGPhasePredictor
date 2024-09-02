import torch
from lightning.fabric import Fabric
from torch.optim import Adam
from tqdm import tqdm

from src.eegpp.lightning_module.eeg_data_module import EEGDataModule
from src.eegpp.models.baseline.cnn_model import CNN1DModel

# from wandb.integration.lightning.fabric import WandbLogger

fabric = Fabric()
fabric.launch()

# model = CNN1DModel().to('cuda:0')
model = CNN1DModel()
optimizer = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

eeg_data_module = EEGDataModule()
eeg_data_module.setup()
eeg_data_module.setup_train_val_k(0)
train_dataloader, val_dataloader = eeg_data_module.train_dataloader(), eeg_data_module.val_dataloader()

model, optimizer = fabric.setup(model, optimizer)
train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

epochs = 1


def train_fabric():
    for epoch in range(epochs):
        model.train()
        for batch_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                                     desc=f"Training epoch {epoch}"):
            optimizer.zero_grad()
            x, y = batch
            pred = model(x)
            loss = loss_fn(pred, y)
            fabric.backward(loss)
            optimizer.step()
            fabric.log('train_loss', loss.item())

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader),
                                         desc=f"Validation epoch {epoch}"):
                x, y = batch
                pred = model(x)
                val_loss += loss_fn(pred, y).item()

        fabric.log('val_loss', val_loss / len(val_dataloader))


def train_pytorch():
    for epoch in range(epochs):
        model.train()
        for batch_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                                     desc=f"Training epoch {epoch}"):
            optimizer.zero_grad()
            x, y = batch
            pred = model(x.to('cuda:0'))
            loss = loss_fn(pred, y.to('cuda:0'))
            loss.backward()
            optimizer.step()
            fabric.log('train_loss', loss.item())

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader),
                                         desc=f"Validation epoch {epoch}"):
                x, y = batch
                pred = model(x.to('cuda:0'))
                val_loss += loss_fn(pred, y.to('cuda:0')).item()

        fabric.log('val_loss', val_loss / len(val_dataloader))


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    train_fabric()
    # train_pytorch()
