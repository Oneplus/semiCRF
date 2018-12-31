#!/usr/bin/env python
import torch
from semi_crf.segment_encoder_base import SegmentEncoderBase


class RecursiveCell(torch.nn.Module):
    def __init__(self, dim):
        super(RecursiveCell, self).__init__()
        self.dim = dim
        self.gate = torch.nn.Linear(dim * 2, dim * 3)
        self.project = torch.nn.Linear(dim * 2, dim)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        batch_size = input_.size(0)
        flattened_input_ = input_.contiguous().view(batch_size, -1)

        z = self.project(flattened_input_).view(batch_size, 1, -1)
        stacked_input_ = torch.cat([input_, z], dim=1)
        g = self.gate(flattened_input_).view(batch_size, 3, -1)
        g = torch.nn.functional.softmax(g, dim=1)
        ret = stacked_input_.mul(g).sum(dim=1)
        return ret


class SegmentalRecursive(SegmentEncoderBase):
    def __init__(self, max_seg_len, dim, use_cuda):
        super(SegmentalRecursive, self).__init__(max_seg_len, use_cuda)
        self.dim = dim
        self.cell = RecursiveCell(dim)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # input_: (batch_size, seq_len, dim)
        batch_size, seq_len, dim = input_.size()

        encoding_ = torch.FloatTensor(batch_size, seq_len, self.max_seg_len, dim).fill_(0.)
        if self.use_cuda:
            encoding_ = encoding_.cuda()

        layer_ = input_
        encoding_[:, :, 0, :] = input_
        for length in range(1, self.max_seg_len):
            for ending_pos in range(length, seq_len):
                encoding_[:, ending_pos, length, :] = self.cell(layer_[:, ending_pos - 1: ending_pos + 1, :])
            layer_ = encoding_[:, :, length, :].squeeze(2)

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

    input_ = torch.arange(0, batch_size * seq_len * dim).view(batch_size, seq_len, dim).float()
    print(input_)
    cell = RecursiveCell(dim)
    cell_input_ = input_[:, 0: 2, :]
    print(cell_input_)
    print(cell(cell_input_))

    encoder = SegmentalRecursive(max_seg_len, dim, False)
    print(encoder)
    print(encoder.encoding_dim())
    print(encoder.forward(input_))
