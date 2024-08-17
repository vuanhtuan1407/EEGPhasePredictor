import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger

from lightning_module.eeg_data_module import EEGDataModule
from lightning_module.eeg_module import EEGModule
from src.eegpp import params
from src.eegpp.callbacks.callback_utils import early_stopping, model_checkpoint


def train(
        model_type=params.MODEL_TYPE,
        enable_logging=params.ENABLE_LOGGING,
        log_dir=params.LOG_DIR,
        enable_checkpointing=params.ENABLE_CHECKPOINTING,
        accelerator=params.ACCELERATOR,
        device=params.DEVICES,
        num_epochs=params.NUM_EPOCHS,
        resume_ckpt=params.RESUME_CKPT,
        batch_size=params.BATCH_SIZE,
        num_workers=params.NUM_WORKERS,
        dataset_file_idx=params.DATASET_FILE_IDX,
        lr=params.LEARNING_RATE
):
    torch.set_float32_matmul_precision('medium')

    # # CLI parsing arguments
    # args = parse_arguments()
    if enable_logging:
        logger = WandbLogger(save_dir=log_dir, project='EEGPhasePredictor')

        # ======================
        # set logger name
        # ======================

        logger.experiment.config['batch_size'] = batch_size
    else:
        logger = False

    # ===================
    # resume_checkpoint
    # ===================

    if resume_ckpt:
        checkpoint = ''
    else:
        checkpoint = None

    if enable_checkpointing:
        trainer_callbacks = [model_checkpoint, early_stopping]
    else:
        trainer_callbacks = [early_stopping]

    eeg_module = EEGModule(
        model_type=model_type,
        lr=lr
    )
    eeg_data_module = EEGDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_file_idx=dataset_file_idx
    )

    trainer = L.Trainer(
        devices=device,
        accelerator=accelerator,
        max_epochs=num_epochs,
        logger=logger,
        enable_checkpointing=enable_checkpointing,
        val_check_interval=1.0,
        reload_dataloaders_every_n_epochs=1,
        callbacks=trainer_callbacks,
    )

    trainer.fit(eeg_module, datamodule=eeg_data_module, ckpt_path=checkpoint)

    if logger:  # turn off wandb quiet if logger is not False
        wandb.finish(quiet=True)


if __name__ == '__main__':
    train()
