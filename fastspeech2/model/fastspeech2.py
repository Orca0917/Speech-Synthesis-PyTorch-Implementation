import torch
import torch.nn as nn
from model.transformer import PositionalEncoding, FeedForwardTransformer, Conv1D

# duratino predictor, pitch predictor, energy predictor
"""
consists of a 2-layer 1D-convolutional network with ReLU activation,
each followed by the layer normalization and the dropout layer, and an extra linear layer to project
the hidden states into the output sequence.
"""
class Predictor(nn.Module):
    def __init__(self, args):
        super(Predictor, self).__init__()
        self.args                   = args
        self.logger                 = args.logger
        self.conv_1d_1              = Conv1D(args=args,
                                             in_dim=args.encoder_hidden,
                                             out_dim=args.variance_predictor_conv1d_filter_size,
                                             kernel_size=args.variance_predictor_conv1d_kernel[0])
        self.conv_1d_2              = Conv1D(args=args,
                                             in_dim=args.variance_predictor_conv1d_filter_size,
                                             out_dim=args.variance_predictor_conv1d_filter_size,
                                             kernel_size=args.variance_predictor_conv1d_kernel[1])
        self.relu                   = nn.ReLU()
        self.layer_normalization    = nn.LayerNorm(normalized_shape=args.variance_predictor_conv1d_filter_size)
        self.dropout                = nn.Dropout(p=args.variance_predictor_dropout)
        self.linear_layer           = nn.Linear(args.variance_predictor_conv1d_filter_size, 1)


    def forward(self, encoder_out):

        conv_out    = self.relu(self.conv_1d_1(encoder_out))
        ln_dropout  = self.dropout(self.layer_normalization(conv_out))
        conv_out    = self.relu(self.conv_1d_2(ln_dropout))
        ln_dropout  = self.dropout(self.layer_normalization(conv_out))
        linear_out  = self.linear_layer(ln_dropout)

        return linear_out


class LengthRegulator(nn.Module):
    def __init__(self, args):
        super(LengthRegulator, self).__init__()
        self.args   = args
        self.logger = args.logger

    def pad(self, expanded_phoneme_sequence: list):
        # phoneme sequence의 길이를 모두 맞춰주는 함수
        # 부족한 공간은 모두 zero pad!!

        # expanded_phoneme_sequence의 각 원소는 2D Tensor
        max_len = max([batch.shape[0] for batch in expanded_phoneme_sequence])

        padded_batch = []
        for batch in expanded_phoneme_sequence:
            batch = torch.nn.functional.pad(input=batch,
                                        pad=(0, 0, 0, max_len - batch.shape[0]),
                                        mode='constant')
            padded_batch += [batch]

        padded_batch = torch.stack(padded_batch)
        return padded_batch

    def forward(self, pred_duration, phoeneme_hidden_sequence):
        self.logger.info(f"Predicted duration: {pred_duration.shape}")
        self.logger.info(f"Phoneme hidden sequence: {phoeneme_hidden_sequence.shape}")

        length_regulated_seq = []
        for B_hidden_seq, B_duration in zip(phoeneme_hidden_sequence, pred_duration):
            batch = []
            for hidden_seq, duration in zip(B_hidden_seq, B_duration):
                hidden_seq = hidden_seq.expand(int(duration.item()), hidden_seq.shape[-1])
                batch += [hidden_seq]

            batch_cat = torch.cat(batch)
            length_regulated_seq += [batch_cat]

        length_regulated_padded_seq = self.pad(length_regulated_seq)
        return length_regulated_padded_seq


class VarianceAdaptor(nn.Module):
    def __init__(self, args):
        super(VarianceAdaptor, self).__init__()
        self.args               = args
        self.logger             = args.logger
        self.duration_predictor = Predictor(args)
        self.energy_predictor   = Predictor(args)
        self.pitch_predictor    = Predictor(args)
        self.length_regulator   = LengthRegulator(args)

    def forward(self, encoder_out):

        predicted_duration = self.duration_predictor(encoder_out)

        # 논문에서는 logarithm 함수를 사용했다고 했는데 왜 다른 구현에서는 exp를 썻을까.
        predicted_duration = torch.clamp(torch.round(torch.exp(predicted_duration) - 1), min=0)
        # self.logger.info(predicted_duration)
        self.logger.info(f"Class: VarianceAdaptor :: predicted duration {predicted_duration.shape}")
        
        length_regulator_out = self.length_regulator(predicted_duration, encoder_out)
        self.logger.info(f"Class: VarianceAdaoptr :: length_regulator_out {length_regulator_out.shape}")
        
        predicted_pitch = self.pitch_predictor(length_regulator_out)
        self.logger.info(f"Class: VarianceAdaoptr :: predicted_pitch {predicted_pitch.shape}")

        predicted_energy = self.energy_predictor(length_regulator_out)
        self.logger.info(f"Class: VarianceAdaoptr :: predicted_energy {predicted_energy.shape}")
        
        variance_adaptor_out = length_regulator_out + predicted_pitch + predicted_energy
        self.logger.info(f"Class: VarianceAdaoptr :: variance_adaptor_out {variance_adaptor_out.shape}")

        return predicted_duration, predicted_pitch, predicted_energy, variance_adaptor_out



class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args   = args
        self.logger = args.logger
        self.n_layers = args.encoder_layers
        self.fft    = FeedForwardTransformer(args=args,
                                             hidden=args.encoder_hidden,
                                             kernel=args.encoder_conv1d_kernel,
                                             filter=args.encoder_conv1d_filter_size,
                                             n_head=args.encoder_attention_heads,
                                             p=args.encoder_decoder_dropout)

    def forward(self, positional_encoded_phoneme):
        self.logger.info(f"Class: Encoder :: positional_encoded_phoneme.shape = {positional_encoded_phoneme.shape}")

        after_fft = positional_encoded_phoneme  # initial value

        # -- 총 encoder_layers(=4) 번의 FFT Block을 적용
        for encoder_layer_index in range(self.n_layers):
            self.logger.info(f"{encoder_layer_index + 1}'th FFT layer")
            after_fft = self.fft(after_fft)

        self.logger.info(f"Encoder out: {after_fft.shape}")

        return after_fft
    

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        self.logger = args.logger
        self.n_layers = args.decoder_layers
        self.fft = FeedForwardTransformer(args=args,
                                             hidden=args.decoder_hidden,
                                             kernel=args.decoder_conv1d_kernel,
                                             filter=args.decoder_conv1d_filter_size,
                                             n_head=args.decoder_attention_heads,
                                             p=args.encoder_decoder_dropout)
        
    def forward(self, variance_adaptor_out):
        self.logger.info(f"Class: Decoder :: variance_adaptor_out.shape = {variance_adaptor_out.shape}")

        after_fft = variance_adaptor_out  # initial value

        # -- 총 encoder_layers(=4) 번의 FFT Block을 적용
        for decoder_layer_index in range(self.n_layers):
            self.logger.info(f"{decoder_layer_index + 1}'th FFT layer")
            after_fft = self.fft(after_fft)

        self.logger.info(f"Decoder out: {after_fft.shape}")

        return after_fft


class FastSpeech2(nn.Module):
    def __init__(self, args):
        super(FastSpeech2, self).__init__()
        self.args                   = args
        self.logger                 = args.logger
        self.phoneme_embedding      = nn.Embedding(76, args.phoneme_embedding_dimension)
        self.positional_encoding    = PositionalEncoding(args)
        self.encoder                = Encoder(args)
        self.variance_adaptor       = VarianceAdaptor(args)
        self.melspectrogram_decoder = Decoder(args)
        self.output_linear_layer    = nn.Linear(args.phoneme_embedding_dimension, 
                                                args.mel_spectrogram_dim)
    
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
        self.logger.info(f"Class: FastSpeech2 :: encoder_out.shape = {encoder_out.shape}")

        # phoneme sequence에 발화 길이, 높낮이, 세기 정보를 추가
        (
            pred_duration, 
            pred_pitch,
            pred_energy,
            variance_adaptor_out
        ) = self.variance_adaptor(encoder_out)
        self.logger.info(f"Class: FastSpeech2 :: variance_adaptor_out.shape = {variance_adaptor_out.shape}")

        # positional encoding transformer 이전 한번 더 추가
        positional_encoded_va_out = self.positional_encoding(variance_adaptor_out)
        self.logger.info(f"Class: FastSpeech2 :: positional_encoded_va_out.shape = {positional_encoded_va_out.shape}")

        # 최종 mel spectrogram 예측
        decoder_out = self.melspectrogram_decoder(positional_encoded_va_out)
        self.logger.info(f"Class: FastSpeech2 :: decoder_out.shape = {decoder_out.shape}")

        # mel channel 80개로 생성
        pred_mel = self.output_linear_layer(decoder_out)
        self.logger.info(f"Class: FastSpeech2 :: pred_mel.shape = {pred_mel.shape}")


        return pred_duration, pred_pitch, pred_energy, pred_mel
