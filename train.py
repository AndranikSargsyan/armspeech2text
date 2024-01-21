import json
import os

import lightning as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from datamodule import CommonVoiceDataModule, alphabet
from lit_conformer import LitConformer
from lit_rnn import ArmSpeechRNN


def train():
    # TODO read from config file or thorough cli args
    seed = 0
    version = 1
    model_id = 'conformer'
    ckpt_path = None

    seed_everything(seed)

    if model_id == 'conformer':
        model = LitConformer(labels=alphabet)
    elif model_id == 'armspeechrnn':
        model = ArmSpeechRNN(labels=alphabet)
    else:
        raise Exception('Model is not supported.')

    checkpoint_callback = ModelCheckpoint(
        monitor='cer',
        dirpath=f'checkpoints/{model_id}_v{version}',
        filename='armspeech2text-{epoch:02d}-{cer:.2f}'
    )
       
    data_loader = CommonVoiceDataModule(
        data_root='./data',
        batch_size=32
    )

    logger = TensorBoardLogger(save_dir=os.getcwd(), version=f'{model_id}_v{version}', name=f"lightning_logs")

    trainer = pl.Trainer(
        enable_checkpointing=True,
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=10000
    )
    trainer.fit(model, data_loader, ckpt_path=ckpt_path)


if __name__ == '__main__':
    train()
