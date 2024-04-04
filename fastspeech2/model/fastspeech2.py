import torch.nn as nn
from utils import log_config
from model.transformer import PositionalEncoding, FeedForwardTransformer, Conv1D
import logging


logger      = logging.getLogger(__name__)


# duratino predictor, pitch predictor, energy predictor
"""
consists of a 2-layer 1D-convolutional network with ReLU activation,
each followed by the layer normalization and the dropout layer, and an extra linear layer to project
the hidden states into the output sequence.
"""
class Predictor(nn.Module):
    def __init__(self, args):
        super(Predictor, self).__init__()
        self.args = args
        self.logger = args.logger

        self.conv_1d = Conv1D()


    def forward(self, encoder_out):
        pass


class LengthRegulator(nn.Moduel):
    def __init__(self, args):
        self.args = args
        self.logger = args.logger

    def forward(self, encoder_out):
        pass


class VarianceAdaptor(nn.Module):
    def __init__(self, args):
        super(VarianceAdaptor, self).__init__()
        self.args = args
        self.logger = args.logger
        self.duration_predictor = Predictor()
        self.energy_predictor = Predictor()
        self.pitch_predictor = Predictor()

    def forward(self, encoder_out):
        pass


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args   = args
        self.logger = args.logger
        self.fft    = FeedForwardTransformer(args=args)

    def forward(self, positional_encoded_phoneme):
        self.logger.info(f"Class: Encoder :: positional_encoded_phoneme.shape = {positional_encoded_phoneme.shape}")

        after_fft = positional_encoded_phoneme  # initial value

        # -- 총 encoder_layers(=4) 번의 FFT Block을 적용
        for encoder_layer_index in range(self.args.encoder_layers):
            self.logger.info(f"{encoder_layer_index + 1}'th FFT layer")
            after_fft = self.fft(after_fft)

        self.logger.info(f"Encoder out: {after_fft.shape}")

        return None


class FastSpeech2(nn.Module):
    def __init__(self, args):
        super(FastSpeech2, self).__init__()
        self.args                   = args
        self.logger                 = args.logger
        self.phoneme_embedding      = nn.Embedding(76, args.phoneme_embedding_dimension)
        self.positional_encoding    = PositionalEncoding(args)
        self.encoder                = Encoder(args)
        self.variance_adaptor       = None
        self.melspectrogram_decoder = None
    
    def forward(self, phoneme_sequence):
        self.logger.info(f"Class: FastSpeech2 :: phoneme_sequence.shape = {phoneme_sequence.shape}")

        # 음소(phoneme)을 임베딩을 거쳐 256차원으로 변경
        embedded_phoneme = self.phoneme_embedding(phoneme_sequence)
        self.logger.info(f"Class: FastSpeech2 :: embedded_phoneme.shape = {embedded_phoneme.shape}")

        # 음소임베딩에 positional encoding을 추가
        positional_encoded_phoneme = self.positional_encoding(embedded_phoneme)
        self.logger.info(f"Class: FastSpeech2 :: positional_encoded_phoneme.shape = {positional_encoded_phoneme.shape}")

        # Feed Forward Transformer Block으로 구성되어 있는 encoder로 전달
        encoder_out = self.encoder(positional_encoded_phoneme)
        # self.logger.info(f"Class: FastSpeech2 :: encoder_out.shape = {encoder_out.shape}")


        return None
