#!/usr/bin/env python
import torch
from semi_crf.highway import Highway
from semi_crf.segment_encoder_base import SegmentEncoderBase


class SegmentalConvolution(SegmentEncoderBase):
    def __init__(self, max_seg_len, dim, filters, n_highway, use_cuda):
        super(SegmentalConvolution, self).__init__(max_seg_len, use_cuda)

        self.convolutions = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(
                in_channels=dim,
                out_channels=num,
                kernel_size=width,
                bias=True
            )
            self.convolutions.append(conv)

        self.padding = torch.nn.ConstantPad1d((0, max_seg_len), 0)
        self.convolutions = torch.nn.ModuleList(self.convolutions)
        self.n_filters = sum(f[1] for f in filters)
        self.n_highway = n_highway
        self.activation = torch.nn.ReLU()
        self.highways = Highway(self.n_filters, self.n_highway, activation=torch.nn.functional.relu)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = input_.size()

        encoding_ = torch.FloatTensor(batch_size, seq_len, self.max_seg_len, self.n_filters).fill_(0.)
        if self.use_cuda:
            encoding_ = encoding_.cuda()

        for ending_pos in range(seq_len):
            for starting_pos in range(max(ending_pos - self.max_seg_len, -1) + 1, ending_pos + 1):
                length = ending_pos - starting_pos
                block_input_ = input_.narrow(1, starting_pos, length + 1).transpose(1, 2)
                convs = []
                for i in range(len(self.convolutions)):
                    padded_block_input_ = self.padding(block_input_)
                    convolved = self.convolutions[i](padded_block_input_)
                    # (batch_size * sequence_length, n_filters for this width)
                    convolved, _ = torch.max(convolved, dim=-1)
                    convolved = self.activation(convolved)
                    convs.append(convolved)
                encoding_[:, ending_pos, length, :] = self.highways(torch.cat(convs, dim=-1))
        return encoding_

    def encoding_dim(self):
        return self.n_filters


if __name__ == "__main__":
    seq_len = 5
    dim = 7
    batch_size = 2
    max_seg_len = 3

    encoder = SegmentalConvolution(max_seg_len, dim, [[1, 32], [2, 32], [3, 64]], 1, False)
    print(encoder)
    print(encoder.encoding_dim())

    input_ = torch.arange(0, batch_size * seq_len * dim).view(batch_size, seq_len, dim).float()
    print(input_)
    print(encoder.forward(input_))
