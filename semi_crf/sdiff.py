#!/usr/bin/env python
import torch
from semi_crf.segment_encoder_base import SegmentEncoderBase


class SegmentalDifference(SegmentEncoderBase):
    def __init__(self, max_seg_len, inp_dim, use_cuda):
        super(SegmentalDifference, self).__init__(max_seg_len, use_cuda)
        self.inp_dim = inp_dim

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # input_: (batch_size, seq_len, dim)
        batch_size, seq_len, dim = input_.size()

        encoding_ = torch.FloatTensor(batch_size, seq_len, self.max_seg_len, dim).fill_(0.)
        if self.use_cuda:
            encoding_ = encoding_.cuda()

        for ending_pos in range(seq_len):
            for starting_pos in range(max(ending_pos - self.max_seg_len, -1) + 1, ending_pos + 1):
                # the starting_pos and ending_pos are inclusive
                length = ending_pos - starting_pos
                if starting_pos == 0:
                    encoding_[:, ending_pos, length, :] = input_[:, ending_pos, :]
                else:
                    encoding_[:, ending_pos, length, :] = input_[:, ending_pos, :] - input_[:, starting_pos - 1, :]

        # output_: (batch_size, seq_len, max_seg_len, dim)
        return encoding_

    def encoding_dim(self):
        return self.inp_dim


if __name__ == "__main__":
    seq_len = 5
    dim = 7
    batch_size = 2
    max_seg_len = 3

    encoder = SegmentalDifference(max_seg_len, False)
    print(encoder)
    print(encoder.encoding_dim())

    input_ = torch.arange(0, batch_size * seq_len * dim).view(batch_size, seq_len, dim)
    print(input_)
    print(encoder.forward(input_))
