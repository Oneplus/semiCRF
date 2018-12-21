#!/usr/bin/env python
import torch
from .input_encoder_base import InputEncoderBase
from .gal_lstm import GalLSTM


class GalLSTMInputEncoder(InputEncoderBase):
    def __init__(self, inp_dim: int,
                 hidden_dim: int,
                 n_layer: int,
                 dropout: float = 0.):
        super(GalLSTMInputEncoder, self).__init__()
        self.encoder_ = GalLSTM(inp_dim, hidden_dim,
                                bidirectional=True, num_layers=n_layer,
                                wdrop=dropout, idrop=dropout, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: (batch_size, seq_len, dim)
        batch_size, seq_len, _ = x.size()
        raw_output, _ = self.encoder_(x)
        # raw_output: (batch_size, seq_len, hidden_dim)
        return raw_output

    def encoding_dim(self):
        return self.hidden_dim * 2


class LSTMInputEncoder(InputEncoderBase):
    def __init__(self, inp_dim: int,
                 hidden_dim: int,
                 n_layer: int,
                 dropout: float = 0.):
        super(LSTMInputEncoder, self).__init__()
        self.encoder_ = torch.nn.LSTM(inp_dim, hidden_dim,
                                      num_layers=n_layer, dropout=dropout,
                                      bidirectional=True, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: (batch_size, seq_len, dim)
        batch_size, seq_len, _ = x.size()
        raw_output, _ = self.encoder_(x)
        # raw_output: (batch_size, seq_len, hidden_dim)
        return raw_output

    def encoding_dim(self):
        return self.hidden_dim * 2
