#!/usr/bin/env python
import torch
from semi_crf.segment_encoder_base import SegmentEncoderBase


class SegmentalConcatenate(SegmentEncoderBase):
    def __init__(self, max_seg_len, dim, dropout, use_cuda):
        super(SegmentalConcatenate, self).__init__(max_seg_len, use_cuda)
        self.dim = dim
        self.dropout = dropout
        output_dim = dim
        encoders = [torch.nn.Sequential(torch.nn.Linear(dim, output_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(p=dropout),
                                        torch.nn.Linear(output_dim, output_dim))
                    for _ in range(max_seg_len)]
        self.encoders = torch.nn.ModuleList(encoders)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # input_: (batch_size, seq_len, dim)
        batch_size, seq_len, dim = input_.size()

        encoding_ = torch.FloatTensor(batch_size, seq_len, self.max_seg_len, dim).fill_(0.)
        if self.use_cuda:
            encoding_ = encoding_.cuda()

        encoding_[:, 0, 0, :] = self.encoders[0](input_[:, 0, :])
        for ending_pos in range(seq_len):
            for starting_pos in range(max(ending_pos - self.max_seg_len, -1) + 1, ending_pos + 1):
                # the starting_pos and ending_pos are inclusive
                length = ending_pos - starting_pos
                for i, pos in enumerate(range(starting_pos, ending_pos + 1)):
                    encoding_[:, ending_pos, length, :] += self.encoders[i](input_[:, pos, :])

        # output_: (batch_size, seq_len, max_seg_len, dim)
        return encoding_

    def encoding_dim(self):
        return self.dim


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
