#!/usr/bin/env python
import torch
from semi_crf.segment_encoder_base import SegmentEncoderBase


class SegmentalDifference(SegmentEncoderBase):
    def __init__(self, max_seg_len, dim, use_cuda):
        super(SegmentalDifference, self).__init__(max_seg_len, use_cuda)
        self.dim = dim
        assert dim % 2 == 0
        self.forward_paddings_ = torch.nn.ModuleList(
            [torch.nn.ConstantPad2d((0, 0, length, 0), 0) for length in range(1, max_seg_len + 1)])
        self.backward_paddings_ = torch.nn.ModuleList(
            [torch.nn.ConstantPad2d((0, 0, 0, length), 0) for length in range(1, max_seg_len + 1)])

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # input_: (batch_size, seq_len, dim)
        batch_size, seq_len, dim = input_.size()

        encoding_ = torch.FloatTensor(batch_size, seq_len, self.max_seg_len, dim).fill_(0.)
        if self.use_cuda:
            encoding_ = encoding_.cuda()

        half_dim = dim // 2
        for length in range(self.max_seg_len):
            encoding_[:, :, length, :half_dim] = \
                input_[:, :, :half_dim] - self.forward_paddings_[length](input_)[:, :seq_len, :half_dim]
            encoding_[:, :, length, half_dim:] = \
                input_[:, :, half_dim:] - self.backward_paddings_[length](input_)[:, length + 1:, half_dim:]

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

    encoder = SegmentalDifference(max_seg_len, dim, False)
    print(encoder)
    print(encoder.encoding_dim())

    input_ = torch.arange(0, batch_size * seq_len * dim).view(batch_size, seq_len, dim)
    print(input_)
    print(encoder.forward(input_))
