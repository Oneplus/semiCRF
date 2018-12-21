#!/usr/bin/env python
import torch
from semi_crf.segment_encoder_base import SegmentEncoderBase


class SegmentalRNN(SegmentEncoderBase):
    def __init__(self, max_seg_len, dim, hidden_dim, dropout, use_cuda):
        super(SegmentalRNN, self).__init__(max_seg_len, use_cuda)
        self.encoder = torch.nn.LSTM(dim, hidden_dim,
                                     bidirectional=True, num_layers=1,
                                     dropout=dropout, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = input_.size()

        encoding_ = torch.FloatTensor(batch_size, seq_len, self.max_seg_len, self.hidden_dim * 2).fill_(0.)
        if self.use_cuda:
            encoding_ = encoding_.cuda()

        for ending_pos in range(seq_len):
            for starting_pos in range(max(ending_pos - self.max_seg_len, -1) + 1, ending_pos + 1):
                length = ending_pos - starting_pos
                block_input_ = input_.narrow(1, starting_pos, length + 1)
                output_, _ = self.encoder(block_input_)
                encoding_[:, ending_pos, length, :] = \
                    torch.cat([output_[:, -1, :self.hidden_dim], output_[:, 0, self.hidden_dim:]], dim=-1)
        return encoding_

    def encoding_dim(self):
        return self.hidden_dim * 2


if __name__ == "__main__":
    seq_len = 5
    dim = 7
    batch_size = 2
    max_seg_len = 3

    encoder = SegmentalRNN(max_seg_len, dim, dim, 0.1, False)
    print(encoder)
    print(encoder.encoding_dim())

    input_ = torch.arange(0, batch_size * seq_len * dim).view(batch_size, seq_len, dim).float()
    print(input_)
    print(encoder.forward(input_))
