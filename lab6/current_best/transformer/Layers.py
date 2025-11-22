''' Define the Layers '''
import torch.nn as nn
from transformer.SubLayers import PositionwiseFeedForward, MultiHeadCrossAttention_Flash, MultiHeadSelfAttention_Flash
class DecoderLayer_Flash(nn.Module):
    ''' Compose with three layers using Flash Attention '''
    def __init__(self, d_model, d_inner, n_head, d_qkv, dropout=0.1):
        super(DecoderLayer_Flash, self).__init__()
        self.slf_attn = MultiHeadSelfAttention_Flash(n_head=n_head, d_model=d_model, d_qkv=d_qkv, dropout=dropout, causal=True)
        self.enc_attn = MultiHeadCrossAttention_Flash(n_head=n_head, d_model=d_model, d_qkv=d_qkv, dropout=dropout, causal=False)
        self.pos_ffn = PositionwiseFeedForward(d_in=d_model, d_hid=d_inner, dropout=dropout)

    def forward(self, dec_input, dec_seq_lens, enc_output, enc_seq_lens):
        #################YOUR CODE HERE#################
        # 1. Self-Attention
        # 2. Encoder-Decoder Attention
        # 3. Position-wise Feed-Forward Network
        ################################################
        x = self.slf_attn(dec_input, dec_seq_lens)
        x = self.enc_attn(x, enc_output, dec_seq_lens, enc_seq_lens)
        dec_output = self.pos_ffn(x)
        return dec_output