#!/usr/bin/env python
import torch
from semi_crf.segment_encoder_base import SegmentEncoderBase


class SegmentalRNN(SegmentEncoderBase):
    def __init__(self, max_seg_len, dim, hidden_dim, dropout, use_cuda):
        super(SegmentalRNN, self).__init__(max_seg_len, use_cuda)
        self.forward_rnn = torch.nn.LSTM(dim, hidden_dim,
                                         bidirectional=False, num_layers=1,
                                         dropout=dropout, batch_first=True)
        self.backward_rnn = torch.nn.LSTM(dim, hidden_dim,
                                          bidirectional=False, num_layers=1,
                                          dropout=dropout, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = input_.size()

        encoding_ = torch.FloatTensor(batch_size, seq_len, self.max_seg_len, self.hidden_dim * 2).fill_(0.)
        if self.use_cuda:
            encoding_ = encoding_.cuda()

        fwd_cache_ = []
        bwd_cache_ = []
        for starting_pos in range(seq_len):
            ending_pos = min(starting_pos + self.max_seg_len, seq_len)
            indices = torch.arange(starting_pos, ending_pos)
            block_input_ = input_.index_select(1, indices)
            fwd_output_, _ = self.forward_rnn(block_input_)
            fwd_cache_.append(fwd_output_)
        for ending_pos in range(seq_len):
            starting_pos = max(-1, ending_pos - self.max_seg_len)
            indices = torch.arange(ending_pos, starting_pos, -1)
            block_input_ = input_.index_select(1, indices)
            bwd_output_, _ = self.backward_rnn(block_input_)
            bwd_cache_.append(bwd_output_)

        for starting_pos in range(seq_len):
            for ending_pos in range(starting_pos, min(starting_pos + self.max_seg_len, seq_len)):
                length = ending_pos - starting_pos
                encoding_[:, ending_pos, length, :self.hidden_dim] = fwd_cache_[starting_pos][:, length, :]
                encoding_[:, ending_pos, length, self.hidden_dim:] = bwd_cache_[ending_pos][:, length, :]

        return encoding_

    def encoding_dim(self):
        return self.hidden_dim * 2

    def numeric_input(self):
        return True


if __name__ == "__main__":
    seq_len = 5
    dim = 7
    batch_size = 2
    max_seg_len = 3
    torch.manual_seed(1)

    encoder = SegmentalRNN(max_seg_len, dim, dim, 0.1, False)
    print(encoder)
    print(encoder.encoding_dim())

    input_ = torch.arange(0, batch_size * seq_len * dim).view(batch_size, seq_len, dim).float()
    print(input_)
    print(encoder.forward(input_))
