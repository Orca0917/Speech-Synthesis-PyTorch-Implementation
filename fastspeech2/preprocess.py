import os
import pandas as pd
import numpy as np
import argparse
from g2p_en import G2p
from tqdm import tqdm
from utils import log_config
import librosa
import logging
import tgt
import pyworld as pw

log_config.setup_logging()
logger      = logging.getLogger(__name__)
phoneme_set = set()


def to_melspectrogram(wav_path, start_t, end_t):
    try:
        y, sr   = librosa.load(wav_path, sr=args.sr)
        y       = y[int(start_t * args.sr) : int(end_t * args.sr)].astype(np.float32)
        mel     = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=args.frame_size, hop_length=args.hop_size)
        return y, mel
    
    except Exception:
        logger.error(f"Cannot read or convert file name: {wav_path}")
        return None


def phoneme_to_label(text, ph2index):
    sequence = [ ph2index[p] for p in text ]
    return sequence


def get_pitch(spectrogram, duration, args):
    pitch, t = pw.dio(spectrogram.astype(np.float64),
                      args.sr,
                      frame_period=args.hop_size / args.sr * 1000)
    
    pitch = pw.stonemask(spectrogram.astype(np.float64), pitch, t, args.sr)
    pitch = pitch[: sum(duration)]

    return pitch


def get_energy(spectrogram):
    energy = np.sum(np.abs(spectrogram) ** 2, axis=0)
    return energy


def get_duration_phoneme_from_textgrid(textgrid_path: str, args):
    silent_phonemes = ['sil', 'sp', 'spn']
    try:
        textgrid = tgt.io.read_textgrid(textgrid_path)
    except Exception:
        logger.error(f"Cannot read textgrid file!")
        raise FileNotFoundError(f"Cannot read file: {textgrid_path}")
    
    phonemes, duration = [], []
    textgrid_object = textgrid.get_tier_by_name("phones")._objects

    for phoneme_information in textgrid_object:
        start_time  = phoneme_information.start_time
        end_time    = phoneme_information.end_time
        phoneme     = phoneme_information.text

        # -- 초기 시작지점 찾기 (앞에 무성 제외)
        if len(phonemes) == 0:
            if phoneme in silent_phonemes:
                continue
            else:
                wav_start_time = start_time
            
        phonemes += [phoneme]

        # -- 음성 종료지점 찾기
        if phoneme not in silent_phonemes:
            wav_end_time = end_time
            last_idx = len(phonemes)

        duration += [
            int(
                round(end_time * args.sr / args.hop_size) -
                round(start_time * args.sr / args.hop_size)
            )
        ]

    phonemes = phonemes[:last_idx]
    duration = duration[:last_idx]

    return phonemes, duration, wav_start_time, wav_end_time

def preprocess(args):
    metadata_path   = os.path.join(args.path, 'metadata.csv')
    metadata_df     = pd.read_csv(metadata_path, header=None, delimiter='|')

    file_name       = metadata_df[0].values                                   # wav file 명
    # text            = metadata_df[2].values                                 # transcript


    all_wavfile_phoneme         = []    # 입력 phoneme
    all_wavfile_melspectrogram  = []    # 정답 melsepctrogram
    all_wavfile_pitch           = []    # 정답 pitch
    all_wavfile_energy          = []    # 정답 energy
    all_wavfile_duration        = []    # 정답 duration

    for fname in tqdm(file_name):
        textgrid_path   = os.path.join("./preprocessed/textgrid", fname) + '.TextGrid'
        wavfile_path    = os.path.join(args.path, 'wavs', fname) + '.wav'
        (
            one_wavfile_phoneme,
            one_wavfile_duration,
            one_wavfile_start_time,
            one_wavfile_end_time
        ) = get_duration_phoneme_from_textgrid(textgrid_path, args)

        (
            one_wavfile_spectrogram,
            one_wavfile_melspectrogram
        ) = to_melspectrogram(wav_path=wavfile_path, 
                              start_t=one_wavfile_start_time, 
                              end_t=one_wavfile_end_time)
        
        one_wavfile_pitch = get_pitch(spectrogram=one_wavfile_spectrogram,
                                      duration=one_wavfile_duration,
                                      args=args)
        
        one_wavfile_energy = get_energy(melspectrogram=one_wavfile_spectrogram)

        all_wavfile_phoneme         += [one_wavfile_phoneme]
        all_wavfile_melspectrogram  += [one_wavfile_melspectrogram]
        all_wavfile_pitch           += [one_wavfile_pitch]
        all_wavfile_energy          += [one_wavfile_energy]
        all_wavfile_duration        += [one_wavfile_duration]

        for ph in one_wavfile_phoneme:
            phoneme_set.add(ph)

    print(phoneme_set)
    logger.info("Start to make phoneme sequence.")
    logger.info(f"vocab size = {len(phoneme_set)}")

    ph2index = {ph: idx for idx, ph in enumerate(phoneme_set)}              # phoneme to index
    os.makedirs("./preprocessed/phoneme_seq", exist_ok=True)                # make directory
    os.makedirs("./preprocessed/melspectrogram", exist_ok=True)             # make directory
    os.makedirs("./preprocessed/duration", exist_ok=True)                   # make directory
    os.makedirs("./preprocessed/energy", exist_ok=True)                     # make directory
    os.makedirs("./preprocessed/pitch", exist_ok=True)                      # make directory

    tqdm_bar = tqdm(zip(all_wavfile_phoneme, 
                        all_wavfile_melspectrogram, 
                        all_wavfile_duration,
                        all_wavfile_energy,
                        all_wavfile_pitch,
                        file_name), total=len(file_name))
    for ph, mel, dur, energy, pitch, f in tqdm_bar:
        phoneme_sequence = phoneme_to_label(ph, ph2index)                   # convert to index
        np.save(f"./preprocessed/phoneme_seq/{f}.npy", phoneme_sequence)    # save phoneme seq
        np.save(f"./preprocessed/melspectrogram/{f}.npy", mel)              # save mel
        np.save(f"./preprocessed/duration/{f}.npy", dur)                    # save mel
        np.save(f"./preprocessed/energy/{f}.npy", energy)                   # save mel
        np.save(f"./preprocessed/pitch/{f}.npy", pitch)                     # save mel


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--path', default='../LJSpeech-1.1')
    args.add_argument('--sr', default=22050, help='sampling rate')
    args.add_argument('--frame_size', default=1024)
    args.add_argument('--hop_size', default=256)
    args = args.parse_args()

    preprocess(args)
