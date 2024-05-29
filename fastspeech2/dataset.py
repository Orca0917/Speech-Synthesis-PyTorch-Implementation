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
        # y: waveform mel seq
        base_path       = "./preprocessed/"
        phoneme_seq     = os.path.join(base_path, "phoneme_seq", self.file_names[index] + ".npy")
        melspectrogram  = os.path.join(base_path, "melspectrogram", self.file_names[index] + ".npy")
        duration        = os.path.join(base_path, "duration", self.file_names[index] + ".npy")
        pitch           = os.path.join(base_path, "pitch", self.file_names[index] + ".npy")
        energy          = os.path.join(base_path, "energy", self.file_names[index] + ".npy")

        phoneme_seq     = np.load(phoneme_seq)      # 전처리 완료된 phoneme sequence
        melspectrogram  = np.load(melspectrogram)   # 전처리 완료된 mel seq
        duration        = np.load(duration)
        pitch           = np.load(pitch)
        energy          = np.load(energy)

        phoneme_seq     = torch.tensor(phoneme_seq, dtype=torch.int32)
        melspectrogram  = torch.tensor(melspectrogram, dtype=torch.float32)
        duration        = torch.tensor(duration, dtype=torch.int32)
        pitch           = torch.tensor(pitch, dtype=torch.float32)
        energy          = torch.tensor(energy, dtype=torch.float32)

        return phoneme_seq, melspectrogram, duration, pitch, energy



def pad_sequence1D(seq):
    return nn.utils.rnn.pad_sequence(sequences=seq, batch_first=True, padding_value=0)


def pad_sequence2D(seqs):
    B                   = len(seqs)
    T                   = seqs[0].shape[1]
    max_len             = max([len(seq[0]) for seq in seqs])
    padded_mel          = torch.zeros(B, T, max_len) # 멜 스펙트로그램 차원 맞춰주기
    for i, seq in enumerate(seqs):
        padded_mel[i, :seq.size(0), :seq.size(1)] = seq
    return padded_mel


def collate_fn(batch):

    phoneme_sequences   = [item[0] for item in batch]
    melspectrograms     = [item[1] for item in batch]
    durations           = [item[2] for item in batch]
    pitches             = [item[3] for item in batch]
    energy              = [item[4] for item in batch]

    padded_phoneme_seq  = pad_sequence1D(phoneme_sequences)     # 음소 시퀀스 차원 맞춰주기
    padded_mel          = pad_sequence2D(melspectrograms)
    padded_duration     = pad_sequence1D(durations)
    padded_pitch        = pad_sequence1D(pitches)
    padded_energy       = pad_sequence1D(energy)


    return padded_phoneme_seq, padded_mel, padded_duration, padded_pitch, padded_energy