import os
import pandas as pd
import numpy as np
import argparse
from g2p_en import G2p
from tqdm import tqdm
from utils import log_config
import librosa
import logging


log_config.setup_logging()
logger      = logging.getLogger(__name__)
g2p         = G2p()
phoneme_set = set()


def to_phoneme(text):
    phoneme_sequence = g2p(text)
    phoneme_sequence = [ '.' if p == '..' else p for p in phoneme_sequence ]
    
    for phoneme in phoneme_sequence:
        phoneme_set.add(phoneme)

    return phoneme_sequence


def to_melspectrogram(file, args):
    try:
        f_path = os.path.join(args.path, 'wavs', file) + '.wav'
        y, sr = librosa.load(f_path, sr=args.sr)
        mel = librosa.feature.melspectrogram(y=y, 
                                             sr=sr,
                                             n_fft=args.frame_size,
                                             hop_length=args.hop_size)
        return mel
    
    except Exception:
        logger.error(f"cannot read file name: {f_path}")
        return None


def phoneme_to_label(text, ph2index):
    sequence = [ ph2index[p] for p in text ]
    return sequence


def preprocess(args):
    metadata_path   = os.path.join(args.path, 'metadata.csv')
    metadata_df     = pd.read_csv(metadata_path, header=None, delimiter='|')

    file_name       = metadata_df[0].values                                 # wav file 명
    text            = metadata_df[2].values                                 # transcript

    logger.info("Start to convert text to phoneme.")
    phoneme = [to_phoneme(t) for t in tqdm(text)]                           # 텍스트를 음소로 변환

    logger.info("Start to make mel-spectrogram.")
    mel_spectrogram = [to_melspectrogram(f, args) for f in tqdm(file_name)] # 멜 스펙트로그램 생성

    logger.info("Start to make phoneme sequence.")
    logger.info(f"vocab size = {len(phoneme_set)}")

    ph2index = {ph: idx for idx, ph in enumerate(phoneme_set)}              # phoneme to index
    os.makedirs("./preprocessed/phoneme_seq", exist_ok=True)                # make directory
    os.makedirs("./preprocessed/melspectrogram", exist_ok=True)             # make directory

    tqdm_bar = tqdm(zip(phoneme, mel_spectrogram, file_name), total=len(phoneme))
    for ph, mel, f in tqdm_bar:
        phoneme_sequence = phoneme_to_label(ph, ph2index)                   # convert to index
        np.save(f"./preprocessed/phoneme_seq/{f}.npy", phoneme_sequence)    # save phoneme seq
        np.save(f"./preprocessed/melspectrogram/{f}.npy", mel)              # save mel


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--path', default='./LJSpeech-1.1')
    args.add_argument('--sr', default=22050, help='sampling rate')
    args.add_argument('--frame_size', default=1024)
    args.add_argument('--hop_size', default=256)
    args = args.parse_args()

    preprocess(args)
