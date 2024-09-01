import torch
import torchaudio
import pandas as pd
import os
from torch.utils.data import Dataset


class UrbanSoundDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, sample_size):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate 
        self.sample_size = sample_size
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        audio_path = self._get_audio_path(index)
        label = self._get_audio_label(index)
        signal, sr = torchaudio.load(audio_path)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_down_size_if_necessary(signal)
        signal = self._right_padding_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label
    
    def _cut_down_size_if_necessary(self, signal):
        if signal.shape[1] > self.sample_size:
            signal = signal[:,:self.sample_size]
        return signal

    def _right_padding_if_necessary(self, signal):
        if signal.shape[1] < self.sample_size:
            signal = torch.nn.functional.pad(signal, (0, self.sample_size-signal.shape[1]))    
        return signal
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _get_audio_path(self, index):
        fold = self.annotations.iloc[index, 5]
        path1 = self.audio_dir + f"fold{fold}"
        path2 = self.annotations.iloc[index, 0]
        path = os.path.join(path1, path2)
        return path 
    
    def _get_audio_label(self, index):
        label = self.annotations.iloc[index, 6]
        return label     
    
    
if __name__ == "__main__":
    ANNOTATIONS_FILE = r'UrbanSound8K\metadata\UrbanSound8K.csv'
    AUDIO_DIR = 'UrbanSound8K/audio/'
    SAMPLE_RATE = 22050
    SAMPLE_SIZE = 22050

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024, 
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, SAMPLE_SIZE)
    signal1, label1 = usd[0]
    signal2, label2 = usd[100]
    print(signal1.shape)
    print(signal2.shape)
