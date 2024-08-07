from lightning.pytorch.loggers import WandbLogger

import params
import torch

def train():
    torch.set_float32_matmul_precision('medium')

    # # CLI parsing arguments
    # args = parse_arguments()
    logger = False
    if params.USE_LOGGER:
        logger = WandbLogger(save_dir=params.LOG_DIR, project='EEGPhasePredictor')
        # ======================
        # set logger name
        # ======================

        logger.experiment.config['batch_size'] = params.BATCH_SIZE

    # resume_ckpt = f'{params.MODEL_TYPE}-{params.DATA_TYPE}-{params.CONF_TYPE}-{int(params.USE_ORGANISM)}_epochs={params.EPOCHS}.ckpt'
    # checkpoint = ut.abspath(f'checkpoints/{resume_ckpt}')
    # if not os.path.exists(checkpoint):
    #     checkpoint = None

    checkpoint = None

    trainer_callbacks = [early_stopping]
    if params.ENABLE_CHECKPOINTING:
        trainer_callbacks = [model_checkpoint, early_stopping]

    sp_module = SPModule(
        model_type=params.MODEL_TYPE,
        data_type=params.DATA_TYPE,
        conf_type=params.CONF_TYPE,
        use_organism=params.USE_ORGANISM,
        batch_size=params.BATCH_SIZE,
        lr=params.LEARNING_RATE,
    )

    sp_data_module = SPDataModule(
        data_type=params.DATA_TYPE,
        batch_size=params.BATCH_SIZE,
        num_workers=params.NUM_WORKERS,
    )

    trainer = L.Trainer(
        devices=params.DEVICES,
        accelerator=params.ACCELERATOR,
        max_epochs=params.EPOCHS,
        logger=logger,
        enable_checkpointing=params.ENABLE_CHECKPOINTING,
        val_check_interval=1.0,
        reload_dataloaders_every_n_epochs=1,
        callbacks=trainer_callbacks,
    )

    trainer.fit(sp_module, datamodule=sp_data_module, ckpt_path=checkpoint)

    if logger:  # turn off wandb quiet if logger is not False
        wandb.finish(quiet=True)