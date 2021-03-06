import torch
import logging
import codecs
import numpy as np
from semi_crf.input_layer_base import InputLayerBase
logger = logging.getLogger(__name__)


class EmbeddingLayer(InputLayerBase):
    def __init__(self, input_field_name,
                 n_d, word2id, embs=None, fix_emb=True, oov='<oov>', pad='<pad>', normalize=False):
        super(EmbeddingLayer, self).__init__(input_field_name)
        self.word2id = word2id
        self.id2word = {i: word for word, i in word2id.items()}
        self.n_V, self.n_d = len(word2id), n_d
        self.oovid = word2id[oov]
        self.padid = word2id[pad]
        self.embedding = torch.nn.Embedding(self.n_V, n_d, padding_idx=self.padid)
        self.embedding.weight.data.uniform_(-0.25, 0.25)

        if embs is not None:
            emb_words, emb_vecs = embs
            weight = self.embedding.weight
            for emb_word, emb_vec in zip(emb_words, emb_vecs):
                if emb_word not in word2id:
                    continue
                i = word2id[emb_word]
                weight.data[i].copy_(torch.from_numpy(emb_vec))

        if normalize:
            weight = self.embedding.weight
            norms = weight.data.norm(2, 1)
            if norms.dim() == 1:
                norms = norms.unsqueeze(1)
            weight.data.div_(norms.expand_as(weight.data))

        if fix_emb:
            self.embedding.weight.requires_grad = False

    def forward(self, input_):
        return self.embedding(input_)

    def encoding_dim(self):
        return self.n_d


def load_embedding_txt(path, has_header=False):
    words = []
    vals = []
    with codecs.open(path, 'r', encoding='utf-8') as fin:
        if has_header:
            fin.readline()

        for line in fin:
            line = line.strip()
            if line:
                parts = line.split()
                words.append(parts[0])
                vals += [float(x) for x in parts[1:]]  # equal to append
    return words, np.asarray(vals).reshape(len(words), -1)  # reshape
