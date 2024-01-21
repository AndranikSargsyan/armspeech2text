import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import lightning as pl
from torch.nn.utils.rnn import pad_sequence

from .augmentations import augment

# alphabet = ['_', ' ', '(', ')', ',', '-', '.', ':', '`', '«', '´', '»', '՚', '՛', '՜', '՝', '՞',
#     'ա', 'բ', 'գ', 'դ', 'ե', 'զ', 'է', 'ը', 'թ', 'ժ', 'ի', 'լ', 'խ', 'ծ', 'կ', 'հ', 'ձ',
#     'ղ', 'ճ', 'մ', 'յ', 'ն', 'շ', 'ո', 'չ', 'պ', 'ջ', 'ռ', 'ս', 'վ', 'տ', 'ր', 'ց', 'ւ',
#     'փ', 'ք', 'օ', 'ֆ', 'և', '։', '֊', '’', '…']

alphabet = ['_', ' ',
    'ա', 'բ', 'գ', 'դ', 'ե', 'զ', 'է', 'ը', 'թ', 'ժ', 'ի', 'լ', 'խ', 'ծ', 'կ', 'հ', 'ձ',
    'ղ', 'ճ', 'մ', 'յ', 'ն', 'շ', 'ո', 'չ', 'պ', 'ջ', 'ռ', 'ս', 'վ', 'տ', 'ր', 'ց', 'ւ',
    'փ', 'ք', 'օ', 'ֆ', 'և', '։', '֊']

alphabet_map = (dict([(alphabet[i], i) for i in range(len(alphabet))]))

alphabet_set = set(alphabet[1:])
def transcript2ids(transcript):
    return [alphabet_map.get(x, 0) for x in list(transcript)]


def ids2transcript(ids):
    return ''.join([alphabet[x] for x in list(ids)])


class CommonVoiceDataset(Dataset):
    def __init__(
        self, 
        data_root: str,
        normalize: bool = True,
        target_sample_rate: int = 16000,
        split='train',
        augment=False
    ):
        self.data_root = data_root
        self.mp3_dir = os.path.join(self.data_root, 'clips')
        self.target_sample_rate = target_sample_rate
        self.normalize = normalize
        self.augment = augment

        if split == 'train':
            split_file = 'my_train.tsv'
        else:
            split_file = 'my_val.tsv'

        self.df = pd.read_csv(os.path.join(self.data_root, split_file), sep='\t')

        self.mp3_paths = self.df['path']
        self.transcripts = self.df['sentence'].str.lower()

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate = target_sample_rate,
            n_fft = int(25 / 1000 * target_sample_rate), # 25 ms
            n_mels = 80, # num of mel filter banks
            hop_length = int(10 / 1000 * target_sample_rate) # 10 ms
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        mp3_path, transcript = self.mp3_paths[index], self.transcripts[index]
        transcript = ''.join(list(filter(lambda x: x in alphabet_set, transcript)))
        audio_sample_path = os.path.join(self.mp3_dir, mp3_path)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr)
        sr = self.target_sample_rate
        if self.augment:
            try:
                signal = torch.tensor(augment(samples=np.array(signal), sample_rate=sr)) # augment
            except Exception:
                print('augmentation exception')
        mel = torch.log(self.mel_transform(signal) + 1e-9)
        mel = (mel - mel.mean()) / (mel.std())
        transcript_ids = transcript2ids(transcript)
        return mel.squeeze().permute(1, 0), transcript_ids

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal


def _collate_fn(batch):
    features = [item[0] for item in batch]
    transcripts = [torch.tensor(item[1], dtype=torch.long).reshape(-1, 1) for item in batch]
    input_sizes = [item.shape[0] for item in features]
    target_sizes = [len(item) for item in transcripts]
    mfcc_padded = pad_sequence(features, batch_first=False)
    transcripts_padded = pad_sequence(transcripts, batch_first=True)
    return mfcc_padded, transcripts_padded.squeeze(), input_sizes, target_sizes


if __name__ == '__main__':
    ds = CommonVoiceDataset(data_root='./data')
    features, transcript_ids = ds[403]
    print(torch.max(features), torch.min(features))
    import matplotlib.pyplot as plt
    plt.imshow(features.permute(1, 0).detach().numpy(), cmap='magma')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig('sample.png')

    print(torch.mean(features))
    print(transcript_ids)
    print(ids2transcript(transcript_ids))

    train_loader = DataLoader(ds, batch_size=16, collate_fn=_collate_fn)
    print('Dataset length', len(ds))
    for batch in train_loader:
        inputs, targets, input_sizes, target_sizes = batch
        print(inputs.shape)
        print(targets.shape)
        print(input_sizes)
        print(target_sizes)
        exit()
