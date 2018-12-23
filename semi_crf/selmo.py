#!/usr/bin/env python
import torch
import h5py
import logging
from semi_crf.segment_encoder_base import SegmentEncoderBase
logger = logging.getLogger(__name__)


class SegmentalContextualizedEmbeddings(SegmentEncoderBase):
    def __init__(self, max_seg_len, lexicon_path, use_cuda):
        super(SegmentalContextualizedEmbeddings, self).__init__(max_seg_len, use_cuda)
        self.dim = dim
        self.lexicon = h5py.File(lexicon_path, 'r')
        self.dim = self.lexicon['#info'][0].item()
        self.n_layers = self.lexicon['#info'][1].item()

        logger.info('dim: {}'.format(dim))
        logger.info('number of layers: {}'.format(n_layers))

        weights = torch.randn(n_layers)
        self.weights = torch.nn.Parameter(weights, requires_grad=True)

    def forward(self, input_: List[List[str]]) -> torch.Tensor:
        # input_: (batch_size, seq_len, dim)
        batch_size = len(input_)
        seq_len = max([len(seq) for seq in input_])

        encoding_ = torch.FloatTensor(batch_size, seq_len, self.max_seg_len, self.dim).fill_(0.)
        if self.use_cuda:
            encoding_ = encoding_.cuda()

        for i, one_input_ in enumerate(input_):
            sentence_key = '\t'.join(one_input_).replace('.', '$period$').replace('/', '$backslash$')
            data = torch.from_numpy(self.lexicon[sentence_key][()]).transpose(0, 1)
            data = torch.autograd.Variable(data, requires_grad=False)
            data = data.transpose(-2, -1).matmul(self.weights)
            for ending_pos in range(seq_len):
                for starting_pos in range(max(ending_pos - self.max_seg_len, -1) + 1, ending_pos + 1):
                    # the starting_pos and ending_pos are inclusive
                    length = ending_pos - starting_pos
                    if ending_pos + 1 < seq_len:
                        forward = data[ending_pos + 1, :self.dim / 2] - data[starting_pos, :self.dim / 2]
                    else:
                        forward = -data[starting_pos, :self.dim / 2]
                    if starting_pos - 1 >= 0:
                        backward = data[starting_pos - 1, self.dim / 2:] - data[ending_pos, self.dim / 2:]
                    else:
                        backward = -data[ending_pos, self.dim / 2:]
                    encoding_[i, ending_pos, length, :] = torch.cat([forward, backward], dim=-1)

        # output_: (batch_size, seq_len, max_seg_len, dim)
        return encoding_

    def encoding_dim(self):
        return self.dim

    def numeric_input(self):
        return False


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)

    seq_len = 5
    dim = 7
    batch_size = 2
    max_seg_len = 3

    encoder = SegmentalContextualizedEmbeddings(max_seg_len, sys.argv[1], False)
    print(encoder)
    print(encoder.encoding_dim())

    input_ = torch.arange(0, batch_size * seq_len * dim).view(batch_size, seq_len, dim)
    print(input_)
    print(encoder.forward(input_))
