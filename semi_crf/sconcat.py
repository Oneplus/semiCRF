#!/usr/bin/env python
import torch
from semi_crf.segment_encoder_base import SegmentEncoderBase


class SegmentalConcatenate(SegmentEncoderBase):
    def __init__(self, max_seg_len, dim, dropout, use_cuda):
        super(SegmentalConcatenate, self).__init__(max_seg_len, use_cuda)
        self.dim = dim
        self.dropout = dropout
        self.paddings_ = torch.nn.ModuleList(
            [torch.nn.ConstantPad2d((0, 0, length, 0), 0) for length in range(max_seg_len)])
        self.projects_ = torch.nn.ModuleList(
            [torch.nn.Linear(dim, dim, bias=False) for _ in range(max_seg_len)])
        self.encoder_ = torch.nn.Sequential(torch.nn.ReLU(),
                                            torch.nn.Dropout(p=dropout),
                                            torch.nn.Linear(dim, dim))

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # input_: (batch_size, seq_len, dim)
        batch_size, seq_len, dim = input_.size()

        encoding_ = torch.FloatTensor(batch_size, seq_len, self.max_seg_len, dim).fill_(0.)
        hidden_ = torch.FloatTensor(batch_size, seq_len, dim).fill_(0.)
        if self.use_cuda:
            encoding_ = encoding_.cuda()
            hidden_ = hidden_.cuda()

        for length in range(self.max_seg_len):
            hidden_ = hidden_ + self.projects_[length](self.paddings_[length](input_)[:, :seq_len, :])
            encoding_[:, :, length, :] = self.encoder_(hidden_)

        return encoding_

    def encoding_dim(self):
        return self.dim

    def numeric_input(self):
        return True


if __name__ == "__main__":
    seq_len = 5
    dim = 7
    batch_size = 2
    max_seg_len = 3

    encoder = SegmentalConcatenate(max_seg_len, dim, dropout=0.1, use_cuda=False)
    print(encoder)
    print(encoder.encoding_dim())

    input_ = torch.arange(0, batch_size * seq_len * dim).view(batch_size, seq_len, dim).float()
    print(input_)
    print(encoder.forward(input_))
