#!/usr/bin/env python
import torch
from semi_crf.segment_encoder_base import SegmentEncoderBase


class DurationEmbeddings(SegmentEncoderBase):
    def __init__(self, max_seg_len, out_dim, use_cuda):
        super(DurationEmbeddings, self).__init__(max_seg_len, use_cuda)
        self.embeddings = torch.nn.Embedding(max_seg_len + 1, out_dim)
        self.out_dim = out_dim

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # input_: (batch_size, seq_len, dim)
        batch_size, seq_len, _ = input_.size()

        mask_ = torch.LongTensor(batch_size, seq_len, self.max_seg_len).fill_(0)
        for ending_pos in range(seq_len):
            for starting_pos in range(max(ending_pos - self.max_seg_len, -1) + 1, ending_pos + 1):
                length = ending_pos - starting_pos
                mask_[:, ending_pos, length] = length + 1
        if self.use_cuda:
            mask_ = mask_.cuda()

        return self.embeddings(mask_)

    def encoding_dim(self):
        return self.out_dim


def debug():
    seq_len = 5
    dim = 7
    batch_size = 2
    max_seg_len = 3

    encoder = DurationEmbeddings(max_seg_len, 4, False)
    print(encoder)
    print(encoder.encoding_dim())

    input_ = torch.arange(0, batch_size * seq_len * dim).view(batch_size, seq_len, dim)
    print(input_)
    print(encoder.forward(input_))


if __name__ == "__main__":
    debug()
