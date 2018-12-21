import torch
import logging
import codecs
import numpy as np
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, n_d, word2id, embs=None, fix_emb=True, oov='<oov>', pad='<pad>', normalize=True):
        super(EmbeddingLayer, self).__init__()
        if embs is not None:
            embwords, embvecs = embs
            # for word in embwords:
            #  assert word not in word2id, "Duplicate words in pre-trained embeddings"
            #  word2id[word] = len(word2id)

            logging.info("{} pre-trained word embeddings loaded.".format(len(word2id)))
            if n_d != len(embvecs[0]):
                logging.warning("[WARNING] n_d ({}) != word vector size ({}). Use {} for embeddings.".format(
                    n_d, len(embvecs[0]), len(embvecs[0])))
                n_d = len(embvecs[0])

        self.word2id = word2id
        self.id2word = {i: word for word, i in word2id.items()}
        self.n_V, self.n_d = len(word2id), n_d
        self.oovid = word2id[oov]
        self.padid = word2id[pad]
        self.embedding = torch.nn.Embedding(self.n_V, n_d, padding_idx=self.padid)
        self.embedding.weight.data.uniform_(-0.25, 0.25)

        if embs is not None:
            weight = self.embedding.weight
            weight.data[:len(embwords)].copy_(torch.from_numpy(embvecs))
            logging.info("embedding shape: {}".format(weight.size()))

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
