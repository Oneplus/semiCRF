#!/usr/bin/env python
import torch
from .input_encoder_base import InputEncoderBase


class ProjectedInputEncoder(InputEncoderBase):
    def __init__(self, inp_dim: int,
                 hidden_dim: int,):
        super(ProjectedInputEncoder, self).__init__()
        self.encoder_ = torch.nn.Linear(inp_dim, hidden_dim, bias=False)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return self.encoder_(x)

    def encoding_dim(self):
        return self.hidden_dim
