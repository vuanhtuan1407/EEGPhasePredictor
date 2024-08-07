import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
import wandb

import params
from callbacks.callback_utils import early_stopping, model_checkpoint
from lightning_module.eeg_data_module import EEGDataModule
from lightning_module.eeg_module import EEGModule


def train():
    torch.set_float32_matmul_precision('medium')

    # # CLI parsing arguments
    # args = parse_arguments()
    if params.ENABLE_LOGGING:
        logger = WandbLogger(save_dir=params.LOG_DIR, project='EEGPhasePredictor')
        # ======================
        # set logger name
        # ======================

        logger.experiment.config['batch_size'] = params.BATCH_SIZE
    else:
        logger = False

    # ===================
    # resume_checkpoint
    # ===================

    checkpoint = None

    trainer_callbacks = [early_stopping]
    if params.ENABLE_CHECKPOINTING:
        trainer_callbacks = [model_checkpoint, early_stopping]

    eeg_module = EEGModule()
    eeg_data_module = EEGDataModule()

    trainer = L.Trainer(
        devices=params.DEVICES,
        accelerator=params.ACCELERATOR,
        max_epochs=params.NUM_EPOCHS,
        logger=logger,
        enable_checkpointing=params.ENABLE_CHECKPOINTING,
        val_check_interval=1.0,
        reload_dataloaders_every_n_epochs=1,
        callbacks=trainer_callbacks,
    )

    trainer.fit(eeg_module, datamodule=eeg_data_module, ckpt_path=checkpoint)

    if logger:  # turn off wandb quiet if logger is not False
        wandb.finish(quiet=True)
