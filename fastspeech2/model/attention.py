import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    scaled dot product attention
    """
    def __init__(self, args):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.logger = args.logger

    def forward(self, query, key, value):
        B, T, d_k       = key.shape     # (16, 139, 128)
        key_transpose   = key.transpose(1, 2)
        QK_transpose    = torch.matmul(query, key_transpose) / torch.sqrt(torch.tensor(d_k))
        softmax_out     = self.softmax(QK_transpose)
        attention_out   = torch.matmul(softmax_out, value)
        

        return attention_out


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.args   = args
        self.logger = args.logger

        # 음소 임베딩 크기는 head의 개수로 나누어 떨어져야만 함 (concat 때문)
        if args.phoneme_embedding_dimension % args.encoder_attention_heads != 0:
            self.logger.error(f"phoeneme_embedding_size({args.phoneme_embedding_dimension}) must be divided by number of heads({args.encoder_attention_heads})")
            raise ValueError("Phoneme embedding dimension must be divisible by the number of attention heads.")
         

        self.h_dim = args.phoneme_embedding_dimension // args.encoder_attention_heads
        self.heads = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(args.phoneme_embedding_dimension, self.h_dim, bias=False),  # query
                nn.Linear(args.phoneme_embedding_dimension, self.h_dim, bias=False),  # key
                nn.Linear(args.phoneme_embedding_dimension, self.h_dim, bias=False),  # value
            ])
            for _ in range(args.encoder_attention_heads)
        ])
        self.O          = nn.Linear(args.phoneme_embedding_dimension, args.phoneme_embedding_dimension, bias=False)
        self.attention  = ScaledDotProductAttention(args)
            
    def forward(self, input):
        self.logger.info(f"Class: MultiHeadAttention :: input.shape = {input.shape}")   # (B, T, H=256)

        attention_results = []
        for head_idx, (Q, K, V) in enumerate(self.heads):
            self.logger.info(f"\nHead index #{head_idx + 1}")
            query, key, value = Q(input), K(input), V(input)
            self.logger.info(f"query shape: {query.shape}")
            self.logger.info(f"key shape: {key.shape}")
            self.logger.info(f"value shape: {value.shape}")

            attention_result = self.attention(query, key, value)
            self.logger.info(f"attention result: {attention_result.shape}")
            attention_results = attention_results + [attention_result]

        concat_result = torch.cat(attention_results, dim=-1)
        self.logger.info(f"concat_result.shape : {concat_result.shape}")
        mha_out = self.O(concat_result)
        self.logger.info(f"multi-head attentino out: {mha_out.shape}")

        return mha_out
