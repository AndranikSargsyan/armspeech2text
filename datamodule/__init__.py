import lightning as pl
from torch.utils.data import DataLoader

from .datasets import alphabet, CommonVoiceDataset, _collate_fn


class CommonVoiceDataModule(pl.LightningDataModule):
    def __init__(self, data_root='./data', batch_size=32):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Define train/val/test datasets here
        self.train_dataset = CommonVoiceDataset(data_root=self.data_root, split='train', augment=True)
        self.val_dataset = CommonVoiceDataset(data_root=self.data_root, split='val')
        self.test_dataset = CommonVoiceDataset(data_root=self.data_root, split='val')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_collate_fn,
            num_workers=16
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=_collate_fn, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=_collate_fn, num_workers=16)