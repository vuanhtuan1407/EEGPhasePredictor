import torch
from tqdm import tqdm


def train_dataloader():
    pass


def val_dataloader():
    pass


def test_dataloader():
    pass


def train_loop(epoch, fabric, dataloader, model, optimizer, loss_fn, arg):
    model.train()
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Training:'):
        optimizer.zero_grad()
        x, y = batch
        pred = model(x)
        loss = loss_fn(pred, y)
        fabric.backward(loss)
        optimizer.step()

        if i % arg.log_interval == 0:
            print(f'Train loss: {loss.item():.4f}')

        if arg.num_sanity_check is not None and i == arg.num_sanity_check:
            break


@torch.no_grad()
def val_loop(epoch, fabric, dataloader, model, optimizer, loss_fn, metrics, arg):
    model.eval()
    val_loss = 0
    accuracy = metrics['accuracy']
    f1score = metrics['f1']
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Training:'):
        x, y = batch
        pred = model(x)
        val_loss += loss_fn(pred, y)
        accuracy.update(pred, y)
        f1score.update(pred, y)


@torch.no_grad()
def test_loop(fabric, dataloader, model, optimizer, loss_fn, metric, arg):
    pass
