import torch
import torchaudio
import pandas as pd
import os
from torch.utils.data import Dataset


class UrbanSoundDataset(Dataset):
    def __init__(self, annotations_file, audio_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        audio_path = self._get_audio_path(index)
        label = self._get_audio_label(index)
        signal, sr = torchaudio.load(audio_path)
        return signal, label
    
    def _get_audio_path(self, index):
        fold = self.annotations.iloc[index, 5]
        path1 = self.audio_dir + f"fold{fold}"
        path2 = self.annotations.iloc[index, 0]
        path = os.path.join(path1, path2)
        return path 
    
    def _get_audio_label(self, index):
        label = self.annotations.iloc[index, 6]
        return label     
    
    
# if __name__ == "__main__":
#     audio_dir_m = 'UrbanSound8K/audio/'
#     annotations_file_m = r'UrbanSound8K\metadata\UrbanSound8K.csv'
    
#     usd = UrbanSoundDataset(annotations_file_m, audio_dir_m)
#     signal, label = usd[0]
#     print(signal.size())
#     print(label)
    
