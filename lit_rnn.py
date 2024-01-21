import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CTCLoss
from torchmetrics.text import CharErrorRate, WordErrorRate

from decoder import GreedyDecoder
from datamodule.datasets import ids2transcript


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=False):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths, h=None):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths, enforce_sorted=False)
        x, h = self.rnn(x, h)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x, h


class ArmSpeechRNN(pl.LightningModule):
    def __init__(self, labels):
        super().__init__()
        self.bidirectional = True
        self.hidden_layers = 5
        self.hidden_size = 512

        self.labels = labels
        self.num_classes = len(self.labels)
        self.input_size = 80

        self.rnns = nn.Sequential(
            BatchRNN(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                rnn_type=nn.GRU,
                bidirectional=self.bidirectional,
                batch_norm=False
            ),
            *(
                BatchRNN(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size,
                    rnn_type=nn.GRU,
                    bidirectional=self.bidirectional
                ) for x in range(self.hidden_layers - 1)
            )
        )

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size),
            nn.Linear(self.hidden_size, self.num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()

        self.criterion = CTCLoss(blank=self.labels.index('_'), reduction='mean', zero_infinity=True)
        self.evaluation_decoder = GreedyDecoder(self.labels)  # Decoder used for validation
        self.wer = WordErrorRate()
        self.cer = CharErrorRate()

    def forward(self, x, lengths, hs=None):
        lengths = torch.tensor(lengths).cpu().int()
        if hs is None:
            hs = [None] * len(self.rnns)
        for i, rnn in enumerate(self.rnns):
            x, h = rnn(x, lengths, hs[i])
        x = self.fc(x)
        x = x.transpose(0, 1)
        x = self.inference_softmax(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets, input_sizes, target_sizes = batch
        model_out = self(inputs, input_sizes)
        out = model_out.transpose(0, 1)  # TxNxH
        out = out.log_softmax(-1)

        loss = self.criterion(out, targets, input_sizes, target_sizes)
        self.log("train_loss", loss.item(), batch_size=1, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, input_sizes, target_sizes = batch
        inputs = inputs.to(self.device)
        out = self(inputs, input_sizes)
        decoded_output, _ = self.evaluation_decoder.decode(out, input_sizes)
        target_sentences = [ids2transcript(target[:t_size]) for target, t_size in zip(targets, target_sizes)]
        self.wer(decoded_output, target_sentences)
        self.cer(decoded_output, target_sentences)
        self.log('wer', self.wer.compute(), prog_bar=True, on_epoch=True)
        self.log('cer', self.cer.compute(), prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=5e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.99
        )
        return [optimizer], [scheduler]
