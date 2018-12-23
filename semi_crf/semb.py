#!/usr/bin/env python
from typing import List
import torch
import codecs
import numpy as np
import logging
from semi_crf.segment_encoder_base import SegmentEncoderBase
from semi_crf.embedding_layer import EmbeddingLayer
logger = logging.getLogger(__name__)


def load_embedding_txt(path, has_header=False):
    words = []
    vals = []
    with codecs.open(path, 'r', encoding='utf-8') as fin:
        if has_header:
            fin.readline()

        for line in fin:
            fields = line.strip().split()
            words.append(fields[0])
            vals += [float(x) for x in fields[1:]]
    return words, np.asarray(vals).reshape(len(words), -1)


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
        self.embedings = EmbeddingLayer(self.dim, self.mapping, embs=None,
                                        fix_emb=fixed, normalize=normalize)
        logger.info('loaded segment embeddings: {0} x {1}'.format(self.embedings.n_V, self.embedings.n_d))

        # hack the embedding initialization to speed up
        data = self.embedings.embedding.weight.data
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
            mask_ = encoding_.cuda()

        for ending_pos in range(seq_len):
            for starting_pos in range(max(ending_pos - self.max_seg_len, -1) + 1, ending_pos + 1):
                # the starting_pos and ending_pos are inclusive
                length = ending_pos - starting_pos
                for i in range(batch_size):
                    token = '_'.join(input_[i][starting_pos: ending_pos + 1])
                    mask_[i, ending_pos, length] = self.mapping.get(token, 0)

        # output_: (batch_size, seq_len, max_seg_len, dim)
        return self.embedings(mask_)

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

    input_ = [['this', 'is', 'a', 'test', '.'], ['China', 'is', 'a', 'country']]
    print(encoder.forward(input_))
