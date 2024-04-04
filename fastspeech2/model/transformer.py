import torch
import torch.nn as nn

from model.attention import MultiHeadAttention


class Conv1D(nn.Module):
    def __init__(self, args):
        super(Conv1D, self).__init__()
        self.args = args
        self.logger = args.logger

        self.conv_1d_1 = nn.Conv1d(in_channels=args.encoder_hidden, 
                                   out_channels=args.encoder_conv1d_filter_size,
                                   kernel_size=args.encoder_conv1d_kernel[0],
                                   padding=(args.encoder_conv1d_kernel[0]-1) // 2)
        
        self.conv_1d_2 = nn.Conv1d(in_channels=args.encoder_conv1d_filter_size,
                                   out_channels=args.encoder_hidden,
                                   kernel_size=args.encoder_conv1d_kernel[1],
                                   padding=(args.encoder_conv1d_kernel[1]-1) // 2)

    def forward(self, input):
        self.logger.info(f"Class: Conv1D :: input: {input.shape}")  # (16, 139, 256)
        input = input.transpose(1, 2)
        input = self.conv_1d_1(input)
        input = self.conv_1d_2(input)
        input = input.transpose(1, 2)
        return input


class FeedForwardTransformer(nn.Module):
    def __init__(self, args):
        super(FeedForwardTransformer, self).__init__()
        self.args                   = args
        self.logger                 = args.logger
        self.multi_head_attention   = MultiHeadAttention(args=args)
        self.conv_1d                = Conv1D(args=args)
        self.layer_norm             = nn.LayerNorm(args.encoder_hidden)
        self.dropout                = nn.Dropout(p=0.5)

    def forward(self, input):
        self.logger.info(f"Class: FeedForwardTransformer :: input.shape = {input.shape}")
        
        residual                    = input
        multi_head_attention_out    = self.multi_head_attention(input)          # Multi-Head Attention
        after_dropout               = self.dropout(multi_head_attention_out)    # Dropout
        after_skip_connection       = after_dropout + residual                  # Add (skip connection)
        after_layer_normalization   = self.layer_norm(after_skip_connection)    # Norm: Layer normalization 

        residual                    = after_layer_normalization
        after_conv                  = self.conv_1d(after_layer_normalization)   # Conv1D
        after_dropout               = self.dropout(after_conv)                  # Dropout
        after_skip_connection       = after_dropout + residual                  # Add (skip connection)
        after_layer_normalization   = self.layer_norm(after_skip_connection)    # Norm: Layer normalization

        return after_layer_normalization


class PositionalEncoding(nn.Module):
    def __init__(self, args):
        super(PositionalEncoding, self).__init__()
        self.device = args.device

    # 나중에 매 배치마다 positional encoding 테이블을 생성하지 않고,
    # 가장 긴 음성을 기준으로 T를 설정하고 재사용해도 좋을 듯
    def forward(self, embedded_phoneme):

        # 입력 음소 임베딩의 차원 확인
        B, T, H  = embedded_phoneme.shape

        # Positional encoding 테이블 생성
        encoding = torch.zeros(T, H, device=self.device, requires_grad=False)
        pos      = torch.arange(0, T, device=self.device, dtype=torch.float32).unsqueeze(dim=1)
        _2i      = torch.arange(0, H, 2, device=self.device, dtype=torch.float32)

        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / H)))
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / H)))
        
        # 원래 음소 임베딩에 positional encoding 을 더한 값 반환
        return embedded_phoneme + encoding
