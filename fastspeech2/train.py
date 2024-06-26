import torch
import logging
import librosa
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from utils import log_config
from textgrid import TextGrid
from dataset import fastspeech2Dataset, collate_fn
from torch.utils.data import DataLoader
from model.fastspeech2 import FastSpeech2


log_config.setup_logging()
logger = logging.getLogger(__name__)


# def learning_rate_scheduler(step):
#     warmup_steps = 4000
#     learning_rate = model_dim ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)
#     return learning_rate

def show_melspectrogram(melspectrogram):
    fig, ax = plt.subplots(1, 1)
    S_dB    = librosa.power_to_db(melspectrogram, ref=np.max)
    img     = librosa.display.specshow(S_dB, ax=ax, x_axis='time', y_axis='mel', hop_length=256, n_fft=1024, sr=22050)
    
    ax.set_title(f'Mel-frequency spectrogram')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    plt.tight_layout()
    plt.show()


def train(args):
    train_dataset    = fastspeech2Dataset(args, mode='train')
    valid_dataset    = fastspeech2Dataset(args, mode='valid')
    test_dataset     = fastspeech2Dataset(args, mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, collate_fn=collate_fn)
    test_dataloader  = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

    model = FastSpeech2(args=args)

    print(model.__dict__)
    for epoch in range(args.epoch):

        model.train()
        for seq, y_mel, y_dur, y_pitch, y_energy in tqdm(train_dataloader, total=len(train_dataloader)):

            pred_dur, pred_pitch, pred_energy, pred_mel = model(seq)
            logger.info(f"Phoneme sequence's shape  : {seq.shape}")

            logger.info(f"(model out) duration.shape : {pred_dur.shape}")
            logger.info(f"(answer) duration.shape : {y_dur.shape}")

            logger.info(f"(model out) melspectrogram.shape : {pred_mel.shape}")
            logger.info(f"(answer) mel-spectrogram's shape   : {y_mel.shape}")

            logger.info(f"(model out) pitch.shape : {pred_pitch.shape}")
            logger.info(f"(answer) pitch.shape : {y_pitch.shape}")

            logger.info(f"(model out) energy.shape : {pred_energy.shape}")
            logger.info(f"(answer) energy.shape : {y_energy.shape}")

            break
            # for mel in y:
            #     show_melspectrogram(mel)

            # break

        model.eval()
        with torch.no_grad():
            for x, y in tqdm(valid_dataloader, total=len(valid_dataloader)):
                pass

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--path', type=str, default='../LJSpeech-1.1')
    args.add_argument('--epoch', type=int, default=1)
    args.add_argument('--phoneme_embedding_dimension', type=int, default=256)
    args.add_argument('--encoder_layers', type=int, default=4)
    args.add_argument('--encoder_hidden', type=int, default=256)
    args.add_argument('--encoder_conv1d_kernel', type=list, default=[9, 1])
    args.add_argument('--encoder_conv1d_filter_size', type=int, default=1024)
    args.add_argument('--encoder_attention_heads', type=int, default=2)
    args.add_argument('--encoder_decoder_dropout', type=float, default=0.1)
    args.add_argument('--variance_predictor_conv1d_kernel', type=list, default=[3, 3])
    args.add_argument('--variance_predictor_conv1d_filter_size', type=int, default=256)
    args.add_argument('--variance_predictor_dropout', type=float, default=0.5)
    args.add_argument('--decoder_layers', type=int, default=4)
    args.add_argument('--decoder_hidden', type=int, default=256)
    args.add_argument('--decoder_conv1d_kernel', type=list, default=[9, 1])
    args.add_argument('--decoder_conv1d_filter_size', type=int, default=1024)
    args.add_argument('--decoder_attention_heads', type=int, default=2)
    args.add_argument('--mel_spectrogram_dim', type=int, default=80)
    args = args.parse_args()

    args.logger = logger
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.logger.info(f"Current device: {args.device}")
    train(args=args)