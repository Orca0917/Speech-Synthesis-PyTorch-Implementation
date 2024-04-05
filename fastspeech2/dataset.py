import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from utils import log_config
import logging


log_config.setup_logging()
logger      = logging.getLogger(__name__)

class fastspeech2Dataset(Dataset):
    def __init__(self, args, mode: str):
        metadata_path   = os.path.join(args.path, 'metadata.csv')
        metadata_df     = pd.read_csv(metadata_path, delimiter='|', header=None)
        self.file_names: list[str] = metadata_df[0].values

        if mode == 'train':
            self.file_names = [f_name for f_name in self.file_names if f_name.split("-")[0] not in ['LJ001', 'LJ002', 'LJ003']]
        elif mode == 'valid':
            self.file_names = [f_name for f_name in self.file_names if f_name.startswith('LJ001') or f_name.startswith('LJ002')]
        elif mode == 'test':
            self.file_names = [f_name for f_name in self.file_names if f_name.startswith('LJ003')]
        else:
            logger.warning(f"Unknown mode {mode} while creating dataset")

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        # X: text phoneme sequence
        # y: waveform mel spectrogram
        base_path       = "./preprocessed/"
        phoneme_seq     = os.path.join(base_path, "phoneme_seq", self.file_names[index] + ".npy")
        melspectrogram  = os.path.join(base_path, "melspectrogram", self.file_names[index] + ".npy")
        text_grid       = os.path.join(base_path, "textgrid", self.file_names[index] + ".TextGrid")

        phoneme_seq     = np.load(phoneme_seq)      # 전처리 완료된 phoneme sequence
        melspectrogram  = np.load(melspectrogram)   # 전처리 완료된 mel spectrogram

        phoneme_seq     = torch.tensor(phoneme_seq, dtype=torch.int32)
        melspectrogram  = torch.tensor(melspectrogram, dtype=torch.float32)
        return phoneme_seq, melspectrogram



def pad_sequence1D(seq):
    return nn.utils.rnn.pad_sequence(sequences=seq, batch_first=True, padding_value=0)


def pad_sequence2D(mels, max_len):
    B                   = len(mels)
    padded_mel          = torch.zeros(B, 128, max_len) # 멜 스펙트로그램 차원 맞춰주기
    for i, spectrogram in enumerate(mels):
        padded_mel[i, :spectrogram.size(0), :spectrogram.size(1)] = spectrogram
    return padded_mel


def collate_fn(batch):

    phoneme_sequences   = [item[0] for item in batch]
    melspectrograms     = [item[1] for item in batch]

    max_mel_timestep    = max([len(mel[0]) for mel in melspectrograms])

    padded_phoneme_seq  = pad_sequence1D(phoneme_sequences)     # 음소 시퀀스 차원 맞춰주기
    padded_mel          = pad_sequence2D(melspectrograms, max_mel_timestep)

    return padded_phoneme_seq, padded_mel