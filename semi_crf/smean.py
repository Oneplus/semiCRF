#!/usr/bin/env python
import torch
from semi_crf.segment_encoder_base import SegmentEncoderBase


class SegmentalMean(SegmentEncoderBase):
    def __init__(self, max_seg_len, dim, use_cuda):
        super(SegmentalMean, self).__init__(max_seg_len, use_cuda)
        self.dim = dim

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # input_: (batch_size, seq_len, dim)
        batch_size, seq_len, dim = input_.size()

        encoding_ = torch.FloatTensor(batch_size, seq_len, self.max_seg_len, dim).fill_(0.)
        if self.use_cuda:
            encoding_ = encoding_.cuda()

        for ending_pos in range(seq_len):
            for starting_pos in range(max(ending_pos - self.max_seg_len, -1) + 1, ending_pos + 1):
                length = ending_pos - starting_pos
                encoding_[:, ending_pos, length, :] = input_[:, starting_pos: ending_pos + 1, :].mean(dim=-2)

        # output_: (batch_size, seq_len, max_seg_len, dim)
        return encoding_

    def encoding_dim(self):
        return self.dim

    def numeric_input(self):
        return True


if __name__ == "__main__":
    seq_len = 5
    dim = 8
    batch_size = 2
    max_seg_len = 3

    encoder = SegmentalMean(max_seg_len, dim, False)
    print(encoder)
    print(encoder.encoding_dim())

    input_ = torch.arange(0, batch_size * seq_len * dim).view(batch_size, seq_len, dim).float()
    print(input_)
    print(encoder.forward(input_))
