from torch.utils.data import Dataset
from config import Config
import numpy as np

class TransformerTTSDataset(Dataset):
    def __init__(self):
        self.phoneme_to_index = {}
        with open(Config.metadata_path, 'r') as f:
            lines = f.readlines()
            self.wav_names = [line.split('|')[0] for line in lines]

    def __len__(self):
        return len(self.wav_names)

    def __getitem__(self, idx):
        wav_name = self.wav_names[idx]


        return (
            np.load(f'{Config.phoneme_paths}/{wav_name}.npy', encoding='utf-8'),
            np.load(f'{Config.spec_paths}/{wav_name}.npy'),
            np.load(f'{Config.mel_path}/{wav_name}.npy'),
        )