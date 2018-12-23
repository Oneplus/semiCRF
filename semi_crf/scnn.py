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

        self.convolutions = torch.nn.ModuleList(self.convolutions)
        self.filters = filters
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
                    padded_block_input_ = torch.nn.functional.pad(block_input_,
                                                                  (0, self.filters[i][0]),
                                                                  'constant', 0.)
                    convolved = self.convolutions[i](padded_block_input_)
                    # (batch_size * sequence_length, n_filters for this width)
                    convolved, _ = torch.max(convolved, dim=-1)
                    convolved = self.activation(convolved)
                    convs.append(convolved)
                encoding_[:, ending_pos, length, :] = self.highways(torch.cat(convs, dim=-1))
        return encoding_

    def encoding_dim(self):
        return self.n_filters

    def numeric_input(self):
        return True


class FastSegmentalConvolution(SegmentEncoderBase):
    def __init__(self, max_seg_len, dim, filters, n_highway, use_cuda):
        super(FastSegmentalConvolution, self).__init__(max_seg_len, use_cuda)

        self.convolutions = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(
                in_channels=dim,
                out_channels=num,
                kernel_size=width,
                bias=True
            )
            self.convolutions.append(conv)

        max_width = max([f[0] for f in filters])
        self.padding = torch.nn.ConstantPad2d((0, 0, 0, max_width), 0)
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

        padded_input_ = self.padding(input_).transpose(1, 2)
        globel_convs = []
        for i in range(len(self.convolutions)):
            convolved = self.convolutions[i](padded_input_)[:, :, :seq_len]
            globel_convs.append(convolved)

        for starting_pos in range(seq_len):
            for ending_pos in range(starting_pos, min(starting_pos + self.max_seg_len, seq_len)):
                length = ending_pos - starting_pos
                convs = []
                for convolved in globel_convs:
                    convolved, _ = torch.max(convolved[:, :, starting_pos: ending_pos + 1], dim=-1)
                    convolved = self.activation(convolved)
                    convs.append(convolved)
                encoding_[:, ending_pos, length, :] = self.highways(torch.cat(convs, dim=-1))
        return encoding_

    def encoding_dim(self):
        return self.n_filters

    def numeric_input(self):
        return True


if __name__ == "__main__":
    seq_len = 5
    dim = 7
    batch_size = 2
    max_seg_len = 3
    torch.manual_seed(1)

    encoder = FastSegmentalConvolution(max_seg_len, dim, [[1, 32], [2, 32], [3, 64]], 1, False)
    print(encoder)
    print(encoder.encoding_dim())

    input_ = torch.arange(0, batch_size * seq_len * dim).view(batch_size, seq_len, dim).float()
    print(input_)
    print(encoder.forward(input_))
