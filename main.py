#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, List, Tuple
import os
import errno
import sys
import codecs
import argparse
import time
import random
import logging
import json
import tempfile
import collections
import torch
import subprocess
import h5py
from semi_crf.embedding_layer import EmbeddingLayer
from semi_crf.embedding_layer import load_embedding_txt
from semi_crf.sdiff import SegmentalDifference
from semi_crf.sconcat import SegmentalConcatenate
from semi_crf.scnn import SegmentalConvolution
from semi_crf.srnn import SegmentalRNN
from semi_crf.dummy_inp import DummyInputEncoder
from semi_crf.lstm_inp import LSTMInputEncoder
from semi_crf.lstm_inp import GalLSTMInputEncoder
from semi_crf.semi_crf import ZeroOrderSemiCRFLayer

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


def dict2namedtuple(dic: Dict):
    return collections.namedtuple('Namespace', dic.keys())(**dic)


def read_corpus(path: str,
                max_seg_len: int,
                split_segment_exceeding_max_length: bool = True):
    """
    read segment format data.

    e.g. token_1, token_2, ... ||| start:length:label start:length:label
    """
    input_dataset_ = []
    segment_dataset_ = []

    with codecs.open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            inputs_, segments_ = line.strip().split('|||')

            segments_ = segments_.strip()
            fields = []
            for field in segments_.split():
                tokens = field.split(':')
                if len(tokens) == 3:
                    start, length, label = tokens
                elif len(tokens) == 2:
                    start, length, label = tokens[0], tokens[1], '-DUMMY-'
                else:
                    raise ValueError('Ill-formatted input data.')
                start, length = int(start), int(length)
                if split_segment_exceeding_max_length and length > max_seg_len:
                    while length > 0:
                        fields.append((start, min(max_seg_len, length), label))
                        length -= max_seg_len
                else:
                    fields.append((start, length, label))
            input_dataset_.append(inputs_.strip().split())
            segment_dataset_.append(fields)
    return input_dataset_, segment_dataset_


def create_one_batch(n_tags: int,
                     max_seg_len: int,
                     input_dataset_: List[List[str]],
                     segment_dataset_: List[List[Tuple[int, int, str]]],
                     word2id: Dict[str, int],
                     label2id: Dict[str, int],
                     oov: str = '<oov>',
                     pad: str = '<pad>',
                     sort: bool = True,
                     use_cuda: bool = False):
    batch_size = len(input_dataset_)
    lst = list(range(batch_size))
    if sort:
        lst.sort(key=lambda l: -len(input_dataset_[l]))

    sorted_input_dataset_ = [input_dataset_[i] for i in lst]
    sorted_segment_dataset_ = [segment_dataset_[i] for i in lst]
    sorted_seq_lens = [len(input_dataset_[i]) for i in lst]
    seq_len = max(sorted_seq_lens)

    oov_id, pad_id = word2id.get(oov, None), word2id.get(pad, None)
    batch_x = torch.LongTensor(batch_size, seq_len).fill_(pad_id)
    batch_y = torch.LongTensor(batch_size, seq_len).fill_(0)
    batch_lens = torch.LongTensor(sorted_seq_lens)

    assert oov_id is not None and pad_id is not None
    for i, input_ in enumerate(sorted_input_dataset_):
        for j, x_ij in enumerate(input_):
            batch_x[i, j] = word2id.get(x_ij, oov_id)
        segment_ = sorted_segment_dataset_[i]
        for start, length, label in segment_:
            label = label2id.get(label, 0)
            end = start + length - 1
            batch_y[i, end] = (length - 1) * n_tags + label

    if use_cuda:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        batch_lens = batch_lens.cuda()

    return batch_x, batch_y, batch_lens


# shuffle training examples and create mini-batches
def create_batches(n_tags: int,
                   max_seg_len: int,
                   input_dataset_: List[List[str]],
                   segment_dataset_: List[List[Tuple[int, int, str]]],
                   word2id: Dict[str, int],
                   label2id: Dict[str, int],
                   batch_size: int,
                   perm=None, shuffle=True, sort=True, keep_full=False, use_cuda=False):

    lst = perm or list(range(len(input_dataset_)))
    if shuffle:
        random.shuffle(lst)

    if sort:
        lst.sort(key=lambda l: -len(input_dataset_[l]))

    sorted_input_dataset_ = [input_dataset_[i] for i in lst]
    sorted_segment_dataset_ = [segment_dataset_[i] for i in lst]

    sum_len = 0.0
    batches_x, batches_y, batches_lens = [], [], []
    size = batch_size
    n_batch = (len(input_dataset_) - 1) // size + 1

    start_id = 0
    while start_id < len(input_dataset_):
        end_id = start_id + size
        if end_id > len(input_dataset_):
            end_id = len(input_dataset_)

        if keep_full and len(sorted_input_dataset_[start_id]) != len(sorted_input_dataset_[end_id - 1]):
            end_id = start_id + 1
            while end_id < len(input_dataset_) and \
                    len(sorted_input_dataset_[end_id]) == len(sorted_input_dataset_[start_id]):
                end_id += 1

        bx, by, blens = create_one_batch(n_tags,
                                         max_seg_len,
                                         sorted_input_dataset_[start_id: end_id],
                                         sorted_segment_dataset_[start_id: end_id],
                                         word2id,
                                         label2id,
                                         sort=sort, use_cuda=use_cuda)
        sum_len += sum(blens)
        batches_x.append(bx)
        batches_y.append(by)
        batches_lens.append(blens)
        start_id = end_id

    logging.info("{} batches, avg len: {:.1f}".format(n_batch, sum_len / len(input_dataset_)))
    order = [0] * len(lst)
    for i, l in enumerate(lst):
        order[l] = i
    return batches_x, batches_y, batches_lens, order


class Model(torch.nn.Module):
    def __init__(self, conf, word_emb_layer, max_seg_len, n_class, use_cuda):
        super(Model, self).__init__()
        self.use_cuda = use_cuda
        self.word_emb_layer = word_emb_layer
        self.dropout = torch.nn.Dropout(p=conf["dropout"])

        input_encoder_name = conf['input_encoder']['name'].lower()
        if input_encoder_name == 'gal_lstm':
            self.input_encoder = GalLSTMInputEncoder(conf['embeddings']['dim'],
                                                     conf['input_encoder']['hidden_dim'],
                                                     conf['input_encoder']['n_layers'])
            inp_dim = self.input_encoder.encoding_dim()
        elif input_encoder_name == 'lstm':
            self.input_encoder = LSTMInputEncoder(conf['embeddings']['dim'],
                                                  conf['input_encoder']['hidden_dim'],
                                                  conf['input_encoder']['n_layers'])
            inp_dim = self.input_encoder.encoding_dim()
        elif input_encoder_name == 'dummy':
            self.input_encoder = DummyInputEncoder()
            inp_dim = conf['embeddings']['dim']
        else:
            raise ValueError('Unknown input encoder: {}'.format(input_encoder_name))

        segment_encoders = []
        enc_dim = 0
        for c in conf['segment_encoders']:
            name = c['name'].lower()
            if name == 'sdiff':
                encoder = SegmentalDifference(max_seg_len, inp_dim, use_cuda)
            elif name == 'sconcat':
                encoder = SegmentalConcatenate(max_seg_len, inp_dim, conf["dropout"], use_cuda)
            elif name == 'scnn':
                encoder = SegmentalConvolution(max_seg_len, inp_dim, c["filters"], c["n_highway"], use_cuda)
            elif name == 'srnn':
                encoder = SegmentalRNN(max_seg_len, inp_dim, c["hidden_dim"], conf["dropout"], use_cuda)
            else:
                raise ValueError('unsupported segment encoders: {}'.format(name))
            segment_encoders.append(encoder)
            enc_dim += encoder.encoding_dim()

        assert len(segment_encoders) > 0

        self.segment_encoders = torch.nn.ModuleList(segment_encoders)

        self.segment_scorer = torch.nn.Sequential(torch.nn.Linear(enc_dim, enc_dim),
                                                  torch.nn.ReLU(),
                                                  torch.nn.Dropout(p=conf['dropout']),
                                                  torch.nn.Linear(enc_dim, n_class))

        if conf["semicrf"]["order"] == 0:
            self.classify_layer = ZeroOrderSemiCRFLayer(use_cuda)
        else:
            raise ValueError("Unsupported order: {}".format(conf["semicrf"]["order"]))

        self.train_time = 0
        self.eval_time = 0
        self.emb_time = 0
        self.classify_time = 0

    def forward(self, input_: torch.Tensor,
                output_: torch.Tensor,
                lens_: torch.Tensor):
        # input_: (batch_size, seq_len)
        input_ = self.word_emb_layer(torch.autograd.Variable(input_).cuda() if self.use_cuda
                                     else torch.autograd.Variable(input_))
        # input_: (batch_size, seq_len)

        encoded_input_ = self.input_encoder(input_)
        # encoded_input_: (batch_size, seq_len, dim)

        encoded_input_ = self.dropout(encoded_input_)

        segment_reprs_ = []
        for segment_encoder in self.segment_encoders:
            segment_reprs_.append(segment_encoder(encoded_input_))

        segment_repr_ = torch.cat(segment_reprs_, dim=-1)

        segment_repr_ = self.dropout(segment_repr_)

        transitions = self.segment_scorer(segment_repr_)
        start_time = time.time()

        output, loss = self.classify_layer.forward(transitions, output_, lens_)

        if not self.training:
            self.classify_time += time.time() - start_time
        return output, loss


def eval_model(model, valid_payload, ix2label, args, gold_path):
    if args.output is not None:
        path = args.output
        fpo = codecs.open(path, 'w', encoding='utf-8')
    else:
        descriptor, path = tempfile.mkstemp(suffix='.tmp')
        fpo = codecs.getwriter('utf-8')(os.fdopen(descriptor, 'w'))

    valid_x, valid_y, valid_lens, orders = valid_payload

    model.eval()
    tagset = []
    for x, y, lens in zip(valid_x, valid_y, valid_lens):
        output, _ = model.forward(x, y, lens)
        for bid in range(len(x)):
            tags = []
            output_data_ = output[bid]
            for start, length, label in output_data_:
                tag = ix2label[int(label)]
                tags.append((start, length, tag))
            tagset.append(tags)

    for order in orders:
        print(' '.join(['{0}:{1}:{2}'.format(start, length, tag) for start, length, tag in tagset[order]]), file=fpo)
    fpo.close()

    model.train()
    p = subprocess.Popen([args.script, gold_path, path], stdout=subprocess.PIPE)
    p.wait()
    f = 0
    for line in p.stdout.readlines():
        f = line.strip().split()[-1]
    #os.remove(path)
    return float(f)


def train_model(epoch: int,
                opt: argparse.Namespace,
                model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                train_payload: Tuple,
                valid_payload: Tuple,
                test_payload: Tuple,
                ix2label: Dict,
                best_valid,
                test_result):
    model.train()

    total_loss, total_tag = 0.0, 0
    cnt = 0
    start_time = time.time()

    train_x, train_y, train_lens, _ = train_payload

    lst = list(range(len(train_x)))
    random.shuffle(lst)
    train_x = [train_x[l] for l in lst]
    train_y = [train_y[l] for l in lst]
    train_lens = [train_lens[l] for l in lst]

    for x, y, lens in zip(train_x, train_y, train_lens):
        cnt += 1
        model.zero_grad()
        _, loss = model.forward(x, y, lens)
        total_loss += loss.item()
        n_tags = sum(lens)
        total_tag += n_tags
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip_grad)
        optimizer.step()

        if cnt % opt.report_steps == 0:
            logging.info("Epoch={} iter={} lr={:.6f} train_ave_loss={:.6f} time={:.2f}s".format(
                epoch, cnt, optimizer.param_groups[0]['lr'],
                1.0 * loss.data[0] / n_tags.float(), time.time() - start_time
            ))
            start_time = time.time()

        if cnt % opt.eval_steps == 0:
            valid_result = eval_model(model, valid_payload, ix2label, opt, opt.gold_valid_path)
            logging.info("Epoch={} iter={} lr={:.6f} train_loss={:.6f} valid_acc={:.6f}".format(
                epoch, cnt, optimizer.param_groups[0]['lr'], total_loss, valid_result))

            if valid_result > best_valid:
                torch.save(model.state_dict(), os.path.join(opt.model, 'model.pkl'))
                logging.info("New record achieved!")
                best_valid = valid_result
                if test is not None:
                    test_result = eval_model(model, test_payload, ix2label, opt, opt.gold_test_path)
                    logging.info("Epoch={} iter={} lr={:.6f} test_acc={:.6f}".format(
                        epoch, cnt, optimizer.param_groups[0]['lr'], test_result))

    return best_valid, test_result


def label_to_index(segment_data_: List[List[Tuple[int, int, str]]],
                   label2id: Dict[str, int],
                   incremental: bool = True):
    for segment_ in segment_data_:
        for _, _, label in segment_:
            if label not in label2id and incremental:
                label2id[label] = len(label2id)


def train():
    cmd = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    cmd.add_argument('--seed', default=1, type=int, help='the random seed.')
    cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    cmd.add_argument('--config', required=True, help='the config file.')
    cmd.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'],
                     help='the type of optimizer: valid options=[sgd, adam]')
    cmd.add_argument('--train_path', required=True, help='the path to the training file.')
    cmd.add_argument('--valid_path', required=True, help='the path to the validation file.')
    cmd.add_argument('--test_path', required=False, help='the path to the testing file.')
    cmd.add_argument('--gold_valid_path', type=str, help='the path to the validation file.')
    cmd.add_argument('--gold_test_path', type=str, help='the path to the testing file.')
    cmd.add_argument("--model", required=True, help="path to save model")
    cmd.add_argument("--batch_size", "--batch", type=int, default=32, help='the batch size.')
    cmd.add_argument("--max_seg_len", default=10, help="the max length of segment.")
    cmd.add_argument("--max_epoch", type=int, default=100, help='the maximum number of iteration.')
    cmd.add_argument("--word_cut", type=int, default=5, help='remove the words that is less frequent than')
    cmd.add_argument("--report_steps", type=int, default=1024, help='eval every x batches')
    cmd.add_argument("--eval_steps", type=int, help='eval every x batches')
    cmd.add_argument("--lr", type=float, default=0.01, help='the learning rate.')
    cmd.add_argument("--lr_decay", type=float, default=0, help='the learning rate decay.')
    cmd.add_argument("--clip_grad", type=float, default=1, help='the tense of clipped grad.')
    cmd.add_argument('--output', help='The path to the output file.')
    cmd.add_argument("--script", required=True, help="The path to the evaluation script")

    opt = cmd.parse_args(sys.argv[2:])

    print(opt)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    if opt.gpu >= 0:
        torch.cuda.set_device(opt.gpu)
        if opt.seed > 0:
            torch.cuda.manual_seed(opt.seed)

    conf = json.load(open(opt.config, 'r'))

    if opt.gold_valid_path is None:
        opt.gold_valid_path = opt.valid_path

    if opt.gold_test_path is None and opt.test_path is not None:
        opt.gold_test_path = opt.test_path

    use_cuda = opt.gpu >= 0 and torch.cuda.is_available()

    raw_training_input_data_, raw_training_segment_data_ = read_corpus(opt.train_path, opt.max_seg_len)
    raw_valid_input_data_, raw_valid_segment_data_ = read_corpus(opt.valid_path, opt.max_seg_len)
    if opt.test_path is not None:
        raw_test_input_data_, raw_test_segment_data_ = read_corpus(opt.test_path, opt.max_seg_len)
    else:
        raw_test_input_data_, raw_test_segment_data_ = [], []

    logging.info('training instance: {}, validation instance: {}, test instance: {}.'.format(
        len(raw_training_segment_data_), len(raw_valid_segment_data_), len(raw_test_segment_data_)))
    logging.info('training tokens: {}, validation tokens: {}, test tokens: {}.'.format(
        sum([len(seq) for seq in raw_training_segment_data_]),
        sum([len(seq) for seq in raw_valid_segment_data_]),
        sum([len(seq) for seq in raw_test_segment_data_])))

    label2id = {'<pad>': 0}
    label_to_index(raw_training_segment_data_, label2id)
    label_to_index(raw_valid_segment_data_, label2id, incremental=False)
    label_to_index(raw_test_segment_data_, label2id, incremental=False)
    logging.info('number of tags: {0}'.format(len(label2id)))

    word_count = collections.Counter()
    for x in raw_training_input_data_:
        for w in x:
            word_count[w] += 1

    if "pretrained" in conf["embeddings"]:
        embs_payload = load_embedding_txt(conf["embeddings"]["pretrained"], conf["embeddings"]["has_header"])
        word_lexicon = {word: i for i, word in enumerate(embs_payload[0])}
        logging.info('loaded {} entries from {}'.format(len(embs_payload[0]), conf["embeddings"]["pretrained"]))
    else:
        embs_payload = None
        word_lexicon = {}
    for w in word_count:
        if word_count[w] >= opt.word_cut and w not in word_lexicon:
            word_lexicon[w] = len(word_lexicon)

    for special_word in ['<oov>', '<pad>']:
        if special_word not in word_lexicon:
            word_lexicon[special_word] = len(word_lexicon)
    logging.info('training vocab size: {}'.format(len(word_lexicon)))

    word_emb_layer = EmbeddingLayer(conf["embeddings"]["dim"], word_lexicon, fix_emb=False, embs=embs_payload)
    logging.info('Word embedding size: {0}'.format(len(word_emb_layer.word2id)))

    n_tags = len(label2id)
    id2label = {ix: label for label, ix in label2id.items()}

    word2id = word_emb_layer.word2id

    training_payload = create_batches(n_tags, opt.max_seg_len,
                                      raw_training_input_data_, raw_training_segment_data_,
                                      word2id, label2id,
                                      opt.batch_size,
                                      use_cuda=use_cuda)

    if opt.eval_steps is None or opt.eval_steps > len(raw_training_input_data_):
        opt.eval_steps = len(training_payload[0])

    valid_payload = create_batches(n_tags, opt.max_seg_len,
                                   raw_valid_input_data_, raw_valid_segment_data_,
                                   word2id, label2id,
                                   opt.batch_size,
                                   shuffle=False, sort=True, keep_full=True,
                                   use_cuda=use_cuda)

    if opt.test_path is not None:
        test_payload = create_batches(n_tags, opt.max_seg_len,
                                      raw_test_input_data_, raw_test_segment_data_,
                                      word2id, label2id,
                                      opt.batch_size,
                                      shuffle=False, sort=True, keep_full=True,
                                      use_cuda=use_cuda)
    else:
        test_payload = None

    model = Model(conf, word_emb_layer, opt.max_seg_len, n_tags, use_cuda)

    logging.info(str(model))
    if use_cuda:
        model = model.cuda()

    need_grad = lambda x: x.requires_grad
    if opt.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(filter(need_grad, model.parameters()), lr=opt.lr)
    else:
        optimizer = torch.optim.SGD(filter(need_grad, model.parameters()), lr=opt.lr)

    try:
        os.makedirs(opt.model)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    with codecs.open(os.path.join(opt.model, 'word.dic'), 'w', encoding='utf-8') as fpo:
        for w, i in word_emb_layer.word2id.items():
            print('{0}\t{1}'.format(w, i), file=fpo)

    with codecs.open(os.path.join(opt.model, 'label.dic'), 'w', encoding='utf-8') as fpo:
        for label, i in label2id.items():
            print('{0}\t{1}'.format(label, i), file=fpo)

    json.dump(vars(opt), codecs.open(os.path.join(opt.model, 'config.json'), 'w', encoding='utf-8'))
    best_valid, test_result = -1e8, -1e8
    for epoch in range(opt.max_epoch):
        best_valid, test_result = train_model(epoch, opt, model, optimizer,
                                              training_payload, valid_payload, test_payload,
                                              id2label, best_valid, test_result)
        if opt.lr_decay > 0:
            optimizer.param_groups[0]['lr'] *= opt.lr_decay
        logging.info('Total encoder time: {:.2f}s'.format(model.eval_time / (epoch + 1)))
        logging.info('Total embedding time: {:.2f}s'.format(model.emb_time / (epoch + 1)))
        logging.info('Total classify time: {:.2f}s'.format(model.classify_time / (epoch + 1)))

    logging.info("best_valid_acc: {:.6f}".format(best_valid))
    logging.info("test_acc: {:.6f}".format(test_result))


def test():
    cmd = argparse.ArgumentParser('The testing components of')
    cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    cmd.add_argument("--input", help="the path to the test file.")
    cmd.add_argument('--output', help='the path to the output file.')
    cmd.add_argument("--models", required=True, help="path to save model")

    args = cmd.parse_args(sys.argv[2:])

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    lexicon = h5py.File(args.lexicon, 'r')
    dim, n_layers = lexicon['#info'][0].item(), lexicon['#info'][1].item()
    logging.info('dim: {}'.format(dim))
    logging.info('n_layers: {}'.format(n_layers))

    model_path = args.model

    args2 = dict2namedtuple(json.load(codecs.open(os.path.join(model_path, 'config.json'), 'r', encoding='utf-8')))

    word_lexicon = {}
    word_emb_layers = []
    with codecs.open(os.path.join(model_path, 'word.dic'), 'r', encoding='utf-8') as fpi:
        for line in fpi:
            tokens = line.strip().split('\t')
            if len(tokens) == 1:
                tokens.insert(0, '\u3000')
            token, i = tokens
            word_lexicon[token] = int(i)

    word_emb_layer = EmbeddingLayer(args2.word_dim, word_lexicon, fix_emb=False, embs=None)

    logging.info('word embedding size: ' + str(len(word_emb_layers[0].word2id)))

    label2id, id2label = {}, {}
    with codecs.open(os.path.join(model_path, 'label.dic'), 'r', encoding='utf-8') as fpi:
        for line in fpi:
            token, i = line.strip().split('\t')
            label2id[token] = int(i)
            id2label[int(i)] = token
    logging.info('number of labels: {0}'.format(len(label2id)))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    model = Model(args2, word_emb_layer, len(label2id), use_cuda)
    model.load_state_dict(torch.load(os.path.join(path, 'model.pkl'), map_location=lambda storage, loc: storage))
    if use_cuda:
        model = model.cuda()

    raw_test_data, raw_test_labels = read_corpus(args.input)
    label_to_index(raw_test_labels, label2id, incremental=False)

    test_data, test_labels, test_lens, order = create_batches(dim, n_layers,
                                                              raw_test_data, raw_test_labels,
                                                              word_lexicon,
                                                              label2id,
                                                              args2.batch_size,
                                                              shuffle=False, sort=True, keep_full=True,
                                                              use_cuda=use_cuda)

    if args.output is not None:
        fpo = codecs.open(args.output, 'w', encoding='utf-8')
    else:
        fpo = codecs.getwriter('utf-8')(sys.stdout)

    model.eval()
    tagset = []
    for x, p, y, lens in zip(test_data, test_labels, test_lens):
        output, loss = model.forward(x, p, y)
        output_data = output.data
        for bid in range(len(x)):
            tags = []
            for k in range(lens[bid]):
                tag = id2label[int(output_data[bid][k])]
                tags.append(tag)
            tagset.append(tags)

    for l in order:
        for tag in tagset[l]:
            print(tag, file=fpo)
        print(file=fpo)

    fpo.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train()
    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        test()
    else:
        print('Usage: {0} [train|test] [options]'.format(sys.argv[0]), file=sys.stderr)
