import lightning as pl
import torch
import torch.nn as nn
from torch.nn import CTCLoss
from torchmetrics.text import CharErrorRate, WordErrorRate

from conformer import Conformer
from datamodule.datasets import ids2transcript
from decoder import GreedyDecoder
from scheduler import WarmupLRScheduler


class LitConformer(pl.LightningModule):
    def __init__(self, labels):
        super().__init__()

        self.labels = labels
        num_classes = len(self.labels)
        input_dim = 80

        self.model = Conformer(
            num_classes=num_classes, 
            input_dim=input_dim, 
            encoder_dim=144, 
            num_encoder_layers=16,
            num_attention_heads=4,
        )

        self.criterion = CTCLoss(blank=self.labels.index('_'), reduction='mean', zero_infinity=True)
        self.evaluation_decoder = GreedyDecoder(self.labels)  # Decoder used for validation
        self.wer = WordErrorRate()
        self.cer = CharErrorRate()

    def forward(self, x, input_lengths, hs=None):
        x, output_lengths = self.model(x, input_lengths)
        return x, output_lengths

    def training_step(self, batch, batch_idx):
        inputs, targets, input_sizes, target_sizes = batch
        inputs = inputs.transpose(0, 1) # BxTxC
        input_sizes = torch.LongTensor(input_sizes)
        target_sizes = torch.LongTensor(target_sizes)
        model_out, output_lengths = self(inputs, input_sizes)
        out = model_out.transpose(0, 1)  # TxBxC
        loss = self.criterion(out, targets, output_lengths, target_sizes)
        self.log("train_loss", loss.item(), batch_size=1, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, input_sizes, target_sizes = batch
        inputs = inputs.transpose(0, 1)
        input_sizes = torch.LongTensor(input_sizes)
        target_sizes = torch.LongTensor(target_sizes)
        out, output_lengths = self(inputs, input_sizes)
        decoded_output, _ = self.evaluation_decoder.decode(out, output_lengths)
        target_sentences = [ids2transcript(target[:t_size]) for target, t_size in zip(targets, target_sizes)]
        self.wer(decoded_output, target_sentences)
        self.cer(decoded_output, target_sentences)

        self.log('wer', self.wer.compute(), prog_bar=True, on_epoch=True)
        self.log('cer', self.cer.compute(), prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=2e-5,
            betas=(0.9, 0.98),
            eps=1e-8
        )

        scheduler = WarmupLRScheduler(
            optimizer,
            warmup_steps=200,
            init_lr=1e-5,
            peak_lr=1e-4,
            total_steps=5000    
        )
        return [optimizer], [scheduler]
