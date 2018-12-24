#!/usr/bin/env python
from typing import List
import torch
import h5py
import logging
from semi_crf.segment_encoder_base import SegmentEncoderBase
logger = logging.getLogger(__name__)


class SegmentalContextualizedEmbeddings(SegmentEncoderBase):
    def __init__(self, max_seg_len, lexicon_path, use_cuda):
        super(SegmentalContextualizedEmbeddings, self).__init__(max_seg_len, use_cuda)
        self.lexicon = h5py.File(lexicon_path, 'r')
        self.dim = self.lexicon['#info'][0].item()
        self.n_layers = self.lexicon['#info'][1].item()

        logger.info('dim: {}'.format(self.dim))
        logger.info('number of layers: {}'.format(self.n_layers))

        weights = torch.randn(self.n_layers)
        self.weights = torch.nn.Parameter(weights, requires_grad=True)
        self.forward_paddings_ = torch.nn.ModuleList(
            [torch.nn.ConstantPad2d((0, 0, length, 0), 0) for length in range(1, max_seg_len + 1)])
        self.backward_paddings_ = torch.nn.ModuleList(
            [torch.nn.ConstantPad2d((0, 0, 0, length), 0) for length in range(1, max_seg_len + 1)])

    def forward(self, input_: List[List[str]]) -> torch.Tensor:
        # input_: (batch_size, seq_len, dim)
        batch_size = len(input_)
        seq_len = max([len(seq) for seq in input_])

        encoding_ = torch.FloatTensor(batch_size, seq_len, self.max_seg_len, self.dim).fill_(0.)
        elmo_ = torch.FloatTensor(batch_size, seq_len, self.dim).fill_(0.)
        if self.use_cuda:
            encoding_ = encoding_.cuda()
            elmo_ = elmo_.cuda()

        half_dim = self.dim // 2
        for i, one_input_ in enumerate(input_):
            sentence_key = '\t'.join(one_input_).replace('.', '$period$').replace('/', '$backslash$')
            one_seq_len_ = len(one_input_)
            data_ = torch.from_numpy(self.lexicon[sentence_key][()]).transpose(0, 1)
            if self.use_cuda:
                data_ = data_.cuda()
            data_ = torch.autograd.Variable(data_, requires_grad=False)
            elmo_[i, :one_seq_len_, :] = data_.transpose(-2, -1).matmul(self.weights)

        for length in range(self.max_seg_len):
            encoding_[:, :, length, :half_dim] = \
                elmo_[:, :, :half_dim] - self.forward_paddings_[length](elmo_)[:, :seq_len, :half_dim]
            encoding_[:, :, length, half_dim:] = \
                elmo_[:, :, half_dim:] - self.backward_paddings_[length](elmo_)[:, length + 1:, half_dim:]

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

    input_ = [
        ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'],
        ['Peter', 'Blackburn'],
        ['BRUSSELS', '1996-08-22']]
    print(input_)
    print(encoder.forward(input_))
