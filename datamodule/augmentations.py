import numpy as np
import torch
import torchaudio
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, RoomSimulator

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.2),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    RoomSimulator(p=0.6)
], p=0.7)


if __name__ == '__main__':
    audio_sample_path = './data/clips/common_voice_hy-AM_26058867.mp3'
    signal, sr = torchaudio.load(audio_sample_path)
    augmented_samples = torch.tensor(augment(samples=np.array(signal), sample_rate=sr))
    torchaudio.save('augmented1.wav', augmented_samples, sample_rate=sr)
