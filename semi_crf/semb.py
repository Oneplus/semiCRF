#!/usr/bin/env python
from typing import List
import torch
import logging
from semi_crf.segment_encoder_base import SegmentEncoderBase
from semi_crf.embedding_layer import EmbeddingLayer, load_embedding_txt
logger = logging.getLogger(__name__)


class SegmentEmbeddings(SegmentEncoderBase):
    def __init__(self, max_seg_len: int,
                 filename: str,
                 has_header: bool,
                 fixed: bool,
                 normalize: bool,
                 use_cuda: bool):
        super(SegmentEmbeddings, self).__init__(max_seg_len, use_cuda)
        words, vals = load_embedding_txt(filename, has_header)
        self.dim = len(vals[0])
        self.mapping = {'<oov>': 0, '<pad>': 1}
        for word in words:
            if word not in self.mapping:
                self.mapping[word] = len(self.mapping)
            else:
                logger.info('{} occurs multiple times?'.format(word))
        self.embeddings = EmbeddingLayer('<semb_nil>', self.dim, self.mapping, embs=None,
                                         fix_emb=fixed, normalize=normalize)
        logger.info('loaded segment embeddings: {0} x {1}'.format(self.embeddings.n_V, self.embeddings.n_d))

        # hack the embedding initialization to speed up
        data = self.embeddings.embedding.weight.data
        # handle oov and pad, avoid effect of randomization.
        data[:2, :].fill_(0)
        data[2:, :].copy_(torch.from_numpy(vals))

        if normalize:
            norms = data.norm(2, 1)
            if norms.dim() == 1:
                norms = norms.unsqueeze(1)
            data.div_(norms.expand_as(data))

    def forward(self, input_: List[List[str]]) -> torch.Tensor:
        # input_: (batch_size, seq_len, dim)
        batch_size = len(input_)
        seq_len = max([len(seq) for seq in input_])
        mask_ = torch.LongTensor(batch_size, seq_len, self.max_seg_len).fill_(1)
        if self.use_cuda:
            mask_ = mask_.cuda()

        for ending_pos in range(seq_len):
            for starting_pos in range(max(ending_pos - self.max_seg_len, -1) + 1, ending_pos + 1):
                # the starting_pos and ending_pos are inclusive
                length = ending_pos - starting_pos
                for i in range(batch_size):
                    token = '_'.join(input_[i][starting_pos: ending_pos + 1])
                    mask_[i, ending_pos, length] = self.mapping.get(token, 0)

        # output_: (batch_size, seq_len, max_seg_len, dim)
        return self.embeddings(mask_)

    def encoding_dim(self):
        return self.dim

    def numeric_input(self):
        return False


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)

    seq_len = 5
    batch_size = 2
    max_seg_len = 3

    encoder = SegmentEmbeddings(max_seg_len, sys.argv[1], has_header=True,
                                fixed=True, normalize=False, use_cuda=False)
    print(encoder)
    print(encoder.encoding_dim())

    input_ = [['this', 'is', 'a', 'test', '.'],
              ['China', 'is', 'a', 'country'],
              ['China', 'Daily', 'is', 'a', 'paper']]
    print(encoder.forward(input_))
