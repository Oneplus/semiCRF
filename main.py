#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, List
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
import shutil
import numpy as np
from semi_crf.batch import Batcher, SegmentBatch, TagBatch, LengthBatch, TextBatch, InputBatch
from semi_crf.embedding_layer import EmbeddingLayer
from semi_crf.embedding_layer import load_embedding_txt
from semi_crf.elmo import InputLayerBase, ContextualizedWordEmbeddings
from semi_crf.sdiff import SegmentalDifference
from semi_crf.smean import SegmentalMean
from semi_crf.srecur import SegmentalRecursive
from semi_crf.sconcat import SegmentalConcatenate
from semi_crf.scnn import SegmentalConvolution, FastSegmentalConvolution
from semi_crf.srnn import SegmentalRNN
from semi_crf.semb import SegmentEmbeddings
from semi_crf.dur_emb import DurationEmbeddings
from semi_crf.dummy_inp import DummyInputEncoder
from semi_crf.lstm_inp import LSTMInputEncoder, GalLSTMInputEncoder
from semi_crf.semi_crf import ZeroOrderSemiCRFLayer
from semi_crf.crf_layer import CRFLayer
from semi_crf.classify_layer import ClassifyLayer
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def dict2namedtuple(dic: Dict):
    return collections.namedtuple('Namespace', dic.keys())(**dic)


def read_segment_corpus(path: str,
                        max_seg_len: int,
                        split_segment_exceeding_max_length: bool = True):
    """
    read segment format data.

    e.g. field_1_1, field_1_2, ... ||| field_2_1, field_2_2, ... ||| start:length:label start:length:label
    """
    input_dataset_ = []
    segment_dataset_ = []

    n_input_fields = None
    with codecs.open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            inputs_, segments_ = line.strip().rsplit('|||', 1)

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

            input_fields_ = inputs_.split('|||')
            if n_input_fields is None:
                n_input_fields = len(input_fields_)
            input_dataset_.append([input_field_.strip().split() for input_field_ in input_fields_])
            segment_dataset_.append(fields)
    new_input_dataset_ = [[] for _ in range(n_input_fields)]
    for input_fields_ in input_dataset_:
        for n, input_field_ in enumerate(input_fields_):
            new_input_dataset_[n].append(input_field_)
    return new_input_dataset_, segment_dataset_


def read_tag_corpus(path: str):
    input_dataset_ = []
    tag_dataset_ = []

    n_input_fields = None
    with codecs.open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            inputs_, tags_ = line.strip().rsplit('|||', 1)

            tags_ = tags_.strip().split()
            input_fields_ = inputs_.split('|||')
            if n_input_fields is None:
                n_input_fields = len(input_fields_)
            input_dataset_.append([input_field_.strip().split() for input_field_ in input_fields_])
            tag_dataset_.append(tags_)
    new_input_dataset_ = [[] for _ in range(n_input_fields)]
    for input_fields_ in input_dataset_:
        for n, input_field_ in enumerate(input_fields_):
            new_input_dataset_[n].append(input_field_)
    return new_input_dataset_, tag_dataset_


class SemiCRFModel(torch.nn.Module):
    def __init__(self, conf: Dict,
                 input_layers: List[InputLayerBase],
                 max_seg_len: int,
                 n_class: int,
                 use_cuda: bool):
        super(SemiCRFModel, self).__init__()
        self.n_class = n_class
        self.use_cuda = use_cuda
        self.dropout = torch.nn.Dropout(p=conf["dropout"])

        self.input_layers = torch.nn.ModuleList(input_layers)
        input_dim = 0
        for input_layer in self.input_layers:
            # the last one in auxilary
            input_dim += input_layer.encoding_dim()

        input_encoder_name = conf['input_encoder']['type'].lower()
        if input_encoder_name == 'gal_lstm':
            self.input_encoder = GalLSTMInputEncoder(input_dim,
                                                     conf['input_encoder']['hidden_dim'],
                                                     conf['input_encoder']['n_layers'],
                                                     conf["dropout"])
            encoded_input_dim = self.input_encoder.encoding_dim()
        elif input_encoder_name == 'lstm':
            self.input_encoder = LSTMInputEncoder(input_dim,
                                                  conf['input_encoder']['hidden_dim'],
                                                  conf['input_encoder']['n_layers'],
                                                  conf["dropout"])
            encoded_input_dim = self.input_encoder.encoding_dim()
        elif input_encoder_name == 'dummy':
            self.input_encoder = DummyInputEncoder()
            encoded_input_dim = input_dim
        else:
            raise ValueError('Unknown input encoder: {}'.format(input_encoder_name))

        segment_encoders = []
        enc_dim = 0
        for c in conf['segment_encoders']:
            name = c['type'].lower()
            if name == 'sdiff':
                encoder = SegmentalDifference(max_seg_len, encoded_input_dim, use_cuda)
            elif name == 'sconcat':
                encoder = SegmentalConcatenate(max_seg_len, encoded_input_dim, conf["dropout"], use_cuda)
            elif name == 'scnn':
                encoder = SegmentalConvolution(max_seg_len, encoded_input_dim, c["filters"], c["n_highway"], use_cuda)
            elif name == 'fast_scnn':
                encoder = FastSegmentalConvolution(max_seg_len, encoded_input_dim,
                                                   c["filters"], c["n_highway"], use_cuda)
            elif name == 'srnn':
                encoder = SegmentalRNN(max_seg_len, encoded_input_dim, c["hidden_dim"], conf["dropout"], use_cuda)
            elif name == 'smean':
                encoder = SegmentalMean(max_seg_len, encoded_input_dim, use_cuda)
            elif name == 'srecur':
                encoder = SegmentalRecursive(max_seg_len, encoded_input_dim, use_cuda)
            elif name == 'semb':
                encoder = SegmentEmbeddings(max_seg_len, c['pretrained'], c['has_header'],
                                            c['fixed'], c['normalize'], use_cuda=use_cuda)
            elif name == 'dur_emb':
                encoder = DurationEmbeddings(max_seg_len, c["dim"], use_cuda)
            else:
                raise ValueError('unsupported segment encoders: {}'.format(name))
            segment_encoders.append(encoder)
            enc_dim += encoder.encoding_dim()

        assert len(segment_encoders) > 0

        self.segment_encoders = torch.nn.ModuleList(segment_encoders)

        label_dim = conf["label_dim"]
        label_embeddings = torch.FloatTensor(n_class, label_dim).uniform_(-1/np.sqrt(label_dim), 1/np.sqrt(label_dim))
        self.label_embeddings = torch.nn.Parameter(label_embeddings)

        score_dim = enc_dim + label_dim
        self.segment_scorer = torch.nn.Sequential(torch.nn.Linear(score_dim, score_dim),
                                                  torch.nn.ReLU(),
                                                  torch.nn.Dropout(p=conf['dropout']),
                                                  torch.nn.Linear(score_dim, 1))

        if conf["classifier"]["order"] == 0:
            self.classify_layer = ZeroOrderSemiCRFLayer(use_cuda)
        else:
            raise ValueError("Unsupported order: {}".format(conf["classifier"]["order"]))

        self.train_time = 0
        self.eval_time = 0
        self.emb_time = 0
        self.classify_time = 0

    def forward(self, input_: Dict[str, torch.Tensor],
                output_: torch.Tensor):
        # input_: (batch_size, seq_len)
        lens_ = input_['length']
        text_ = input_['text']
        embeddings_ = []
        for input_layer in self.input_layers:
            input_field_name = input_layer.input_field_name
            embeddings_.append(input_layer(input_[input_field_name]))

        if len(embeddings_) > 0:
            input_ = torch.cat(embeddings_, dim=-1)
            # input_: (batch_size, seq_len, input_dim)

            encoded_input_ = self.input_encoder(input_)
            # encoded_input_: (batch_size, seq_len, dim)

            encoded_input_ = self.dropout(encoded_input_)
        else:
            assert all([not segment_encoder_.numeric_input() for segment_encoder_ in self.segment_encoders])

        segment_reprs_ = []
        for segment_encoder in self.segment_encoders:
            # decide to use numeric input or not
            if segment_encoder.numeric_input():
                segment_reprs_.append(segment_encoder(encoded_input_))
            else:
                segment_reprs_.append(segment_encoder(text_))

        segment_repr_ = torch.cat(segment_reprs_, dim=-1)

        segment_repr_ = self.dropout(segment_repr_)

        segment_repr_ = segment_repr_.unsqueeze_(-2).expand(-1, -1, -1, self.n_class, -1)

        label_size, label_dim = self.label_embeddings.size()
        label_repr_ = self.label_embeddings.view(1, 1, 1, label_size, label_dim).expand(
            segment_repr_.size(0), segment_repr_.size(1), segment_repr_.size(2), -1, -1)
        segment_repr_ = torch.cat([segment_repr_, label_repr_], dim=-1)

        transitions = self.segment_scorer(segment_repr_).squeeze(-1)
        start_time = time.time()

        output, loss = self.classify_layer.forward(transitions, output_, lens_)

        if not self.training:
            self.classify_time += time.time() - start_time
        return output, loss


class SeqLabelModel(torch.nn.Module):
    def __init__(self, conf: Dict,
                 input_layers: List[InputLayerBase],
                 n_class: int,
                 use_cuda: bool):
        super(SeqLabelModel, self).__init__()
        self.n_class = n_class
        self.use_cuda = use_cuda
        self.dropout = torch.nn.Dropout(p=conf["dropout"])

        self.input_layers = torch.nn.ModuleList(input_layers)
        input_dim = 0
        for input_layer in self.input_layers:
            # the last one in auxiliary
            input_dim += input_layer.encoding_dim()

        input_encoder_name = conf['input_encoder']['type'].lower()
        if input_encoder_name == 'gal_lstm':
            self.input_encoder = GalLSTMInputEncoder(input_dim,
                                                     conf['input_encoder']['hidden_dim'],
                                                     conf['input_encoder']['n_layers'],
                                                     conf["dropout"])
            encoded_input_dim = self.input_encoder.encoding_dim()
        elif input_encoder_name == 'lstm':
            self.input_encoder = LSTMInputEncoder(input_dim,
                                                  conf['input_encoder']['hidden_dim'],
                                                  conf['input_encoder']['n_layers'],
                                                  conf["dropout"])
            encoded_input_dim = self.input_encoder.encoding_dim()
        elif input_encoder_name == 'dummy':
            self.input_encoder = DummyInputEncoder()
            encoded_input_dim = input_dim
        else:
            raise ValueError('Unknown input encoder: {}'.format(input_encoder_name))

        if conf["classifier"]["type"].lower() == 'crf':
            self.classify_layer = CRFLayer(encoded_input_dim, n_class, use_cuda)
        else:
            self.classify_layer = ClassifyLayer(encoded_input_dim, n_class, use_cuda)

        self.train_time = 0
        self.eval_time = 0
        self.emb_time = 0
        self.classify_time = 0

    def forward(self, input_: List[torch.Tensor],
                output_: torch.Tensor):
        # input_: (batch_size, seq_len)
        embeddings_ = []
        for input_layer in self.input_layers:
            input_field_name = input_layer.input_field_name
            embeddings_.append(input_layer(input_[input_field_name]))

        input_ = torch.cat(embeddings_, dim=-1)
        # input_: (batch_size, seq_len, input_dim)

        encoded_input_ = self.input_encoder(input_)
        # encoded_input_: (batch_size, seq_len, dim)

        encoded_input_ = self.dropout(encoded_input_)

        start_time = time.time()

        output, loss = self.classify_layer.forward(encoded_input_, output_)

        if not self.training:
            self.classify_time += time.time() - start_time
        return output, loss


def eval_model(model: torch.nn.Module,
               batcher: Batcher,
               ix2label: Dict[int, str],
               args,
               gold_path: str):
    if args.output is not None:
        path = args.output
        fpo = codecs.open(path, 'w', encoding='utf-8')
    else:
        descriptor, path = tempfile.mkstemp(suffix='.tmp')
        fpo = codecs.getwriter('utf-8')(os.fdopen(descriptor, 'w'))

    is_segment = isinstance(model, SemiCRFModel)
    model.eval()
    tagset = []
    orders = []
    for input_, output_, order in batcher.get():
        output, _ = model.forward(input_, output_)
        for bid in range(len(input_['text'])):
            tags = []
            output_data_ = output[bid]
            if is_segment:
                for start, length, label in output_data_:
                    tag = ix2label[int(label)]
                    tags.append((start, length, tag))
            else:
                lens = input_['length'][bid]
                for k in range(lens):
                    tag = ix2label[output[bid][k].item()]
                    tags.append(tag)
            tagset.append(tags)
        orders.extend(order)

    for order in orders:
        if is_segment:
            print(' '.join(['{0}:{1}:{2}'.format(start, length, tag) for start, length, tag in tagset[order]]),
                  file=fpo)
        else:
            print(' '.join(['{0}'.format(tag) for tag in tagset[order]]), file=fpo)
    fpo.close()

    model.train()
    p = subprocess.Popen([args.script, gold_path, path], stdout=subprocess.PIPE)
    p.wait()
    f = 0
    for line in p.stdout.readlines():
        f = line.strip().split()[-1]
    # os.remove(path)
    return float(f)


def train_model(epoch: int,
                opt: argparse.Namespace,
                model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                train_batch: Batcher,
                valid_batch: Batcher,
                test_batch: Batcher,
                ix2label: Dict,
                best_valid: float,
                test_result: float):
    model.train()

    total_loss, total_tag = 0.0, 0
    cnt = 0
    start_time = time.time()

    for input_, segment_, _ in train_batch.get():
        cnt += 1
        model.zero_grad()
        _, loss = model.forward(input_, segment_)

        total_loss += loss.item()
        n_tags = sum(input_['length'])
        total_tag += n_tags
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip_grad)
        optimizer.step()

        if cnt % opt.report_steps == 0:
            logger.info("Epoch={} iter={} lr={:.6f} train_ave_loss={:.6f} time={:.2f}s".format(
                epoch, cnt, optimizer.param_groups[0]['lr'],
                1.0 * loss.item() / n_tags.float(), time.time() - start_time
            ))
            start_time = time.time()

        if cnt % opt.eval_steps == 0:
            valid_result = eval_model(model, valid_batch, ix2label, opt, opt.gold_valid_path)
            logger.info("Epoch={} iter={} lr={:.6f} train_loss={:.6f} valid_acc={:.6f}".format(
                epoch, cnt, optimizer.param_groups[0]['lr'], total_loss, valid_result))

            if valid_result > best_valid:
                torch.save(model.state_dict(), os.path.join(opt.model, 'model.pkl'))
                logger.info("New record achieved!")
                best_valid = valid_result
                if test is not None:
                    test_result = eval_model(model, test_batch, ix2label, opt, opt.gold_test_path)
                    logger.info("Epoch={} iter={} lr={:.6f} test_acc={:.6f}".format(
                        epoch, cnt, optimizer.param_groups[0]['lr'], test_result))

    return best_valid, test_result


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
    cmd.add_argument("--report_steps", type=int, default=1024, help='eval every x batches')
    cmd.add_argument("--eval_steps", type=int, help='eval every x batches')
    cmd.add_argument("--lr", type=float, default=0.01, help='the learning rate.')
    cmd.add_argument("--lr_decay", type=float, default=0, help='the learning rate decay.')
    cmd.add_argument("--clip_grad", type=float, default=1, help='the tense of clipped grad.')
    cmd.add_argument('--output', help='The path to the output file.')
    cmd.add_argument("--script", required=True, help="The path to the evaluation script")

    opt = cmd.parse_args(sys.argv[2:])
    print(opt)

    # setup random
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
    use_semi_crf = conf["classifier"]["type"].lower() == 'semicrf'
    if use_semi_crf:
        read_fn = read_segment_corpus
        read_train_fn_args = [opt.train_path, opt.max_seg_len]
        read_valid_fn_args = [opt.valid_path, opt.max_seg_len]
        read_test_fn_args = [opt.test_path, opt.max_seg_len]
    else:
        read_fn = read_tag_corpus
        read_train_fn_args = [opt.train_path]
        read_valid_fn_args = [opt.valid_path]
        read_test_fn_args = [opt.test_path]

    # load raw data
    raw_training_input_data_, raw_training_output_data_ = read_fn(*read_train_fn_args)
    raw_valid_input_data_, raw_valid_output_data_ = read_fn(*read_valid_fn_args)
    if opt.test_path is not None:
        raw_test_input_data_, raw_test_output_data_ = read_fn(*read_test_fn_args)
    else:
        raw_test_input_data_, raw_test_output_data_ = [], []

    logger.info('we have {0} fields'.format(len(raw_training_input_data_)))
    logger.info('training instance: {}, validation instance: {}, test instance: {}.'.format(
        len(raw_training_input_data_[0]), len(raw_valid_input_data_[0]), len(raw_test_input_data_[0])))
    logger.info('training tokens: {}, validation tokens: {}, test tokens: {}.'.format(
        sum([len(seq) for seq in raw_training_input_data_[0]]),
        sum([len(seq) for seq in raw_valid_input_data_[0]]),
        sum([len(seq) for seq in raw_test_input_data_[0]])))

    # create batcher
    input_batchers = {}
    for c in conf['input']:
        if c['type'] == 'embeddings':
            batcher = InputBatch(c['name'], c['field'], c['min_cut'],
                                 c.get('oov', '<oov>'), c.get('pad', '<pad>'), not c.get('cased', True), use_cuda)
            if c['fixed']:
                if 'pretrained' in c:
                    batcher.create_dict_from_file(c['pretrained'])
                else:
                    logger.warning('it is un-reasonable to use fix embedding without pretraining.')
            else:
                batcher.create_dict_from_dataset(raw_training_input_data_[c['field']])
            input_batchers[c['name']] = batcher

    # till now, lexicon is fixed, but embeddings was not
    # keep the order of [textbatcher, lengthbatcher]
    input_batchers['text'] = TextBatch(use_cuda)
    input_batchers['length'] = LengthBatch(use_cuda)

    if conf['classifier']['type'].lower() == 'semicrf':
        output_batcher = SegmentBatch(use_cuda)
        output_batcher.create_dict_from_dataset(raw_training_output_data_)
        logger.info('tags: {0}'.format(output_batcher.mapping))
    else:
        output_batcher = TagBatch(use_cuda)
        output_batcher.create_dict_from_dataset(raw_training_output_data_)
        logger.info('tags: {0}'.format(output_batcher.mapping))

    input_layers = []
    for i, c in enumerate(conf['input']):
        if c['type'] == 'embeddings':
            if 'pretrained' in c:
                embs = load_embedding_txt(c['pretrained'], c['has_header'])
                logger.info('loaded {0} embedding entries.'.format(len(embs[0])))
            else:
                embs = None
            name = c['name']
            mapping = input_batchers[name].mapping
            layer = EmbeddingLayer(name, c['dim'], mapping, fix_emb=c['fixed'],
                                   embs=embs, normalize=c.get('normalize', False))
            logger.info('embedding layer for field {0} '
                        'created with {1} x {2}.'.format(c['field'], layer.n_V, layer.n_d))
            input_layers.append(layer)
        elif c['type'] == 'elmo':
            name = c['name']
            layer = ContextualizedWordEmbeddings(name, c['path'], use_cuda)
            input_layers.append(layer)
        else:
            raise ValueError('{} unknown input layer'.format(c['type']))

    n_tags = output_batcher.n_tags
    id2label = {ix: label for label, ix in output_batcher.mapping.items()}

    training_batcher = Batcher(n_tags, opt.max_seg_len,
                               raw_training_input_data_,
                               raw_training_output_data_,
                               input_batchers, output_batcher, opt.batch_size, use_cuda=use_cuda)

    if opt.eval_steps is None or opt.eval_steps > len(raw_training_input_data_):
        opt.eval_steps = training_batcher.num_batches()

    valid_batcher = Batcher(n_tags, opt.max_seg_len,
                            raw_valid_input_data_, raw_valid_output_data_,
                            input_batchers, output_batcher, opt.batch_size,
                            shuffle=False, sorting=True, keep_full=True,
                            use_cuda=use_cuda)

    if opt.test_path is not None:
        test_batcher = Batcher(n_tags, opt.max_seg_len,
                               raw_test_input_data_, raw_test_output_data_,
                               input_batchers, output_batcher,
                               opt.batch_size,
                               shuffle=False, sorting=True, keep_full=True,
                               use_cuda=use_cuda)
    else:
        test_batcher = None

    if use_semi_crf:
        model = SemiCRFModel(conf, input_layers, opt.max_seg_len, n_tags, use_cuda)
    else:
        model = SeqLabelModel(conf, input_layers, n_tags, use_cuda)

    logger.info(str(model))
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

    for name_, input_batcher in input_batchers.items():
        if name_ == 'text' or name_ == 'length':
            continue
        with codecs.open(os.path.join(opt.model, '{0}.dic'.format(input_batcher.name)), 'w',
                         encoding='utf-8') as fpo:
            for w, i in input_batcher.mapping.items():
                print('{0}\t{1}'.format(w, i), file=fpo)

    with codecs.open(os.path.join(opt.model, 'label.dic'), 'w', encoding='utf-8') as fpo:
        for label, i in output_batcher.mapping.items():
            print('{0}\t{1}'.format(label, i), file=fpo)

    new_config_path = os.path.join(opt.model, os.path.basename(opt.config))
    shutil.copy(opt.config, new_config_path)
    opt.config = new_config_path
    json.dump(vars(opt), codecs.open(os.path.join(opt.model, 'config.json'), 'w', encoding='utf-8'))

    best_valid, test_result = -1e8, -1e8
    for epoch in range(opt.max_epoch):
        best_valid, test_result = train_model(epoch, opt, model, optimizer,
                                              training_batcher, valid_batcher, test_batcher,
                                              id2label, best_valid, test_result)
        if opt.lr_decay > 0:
            optimizer.param_groups[0]['lr'] *= opt.lr_decay
        logger.info('Total encoder time: {:.2f}s'.format(model.eval_time / (epoch + 1)))
        logger.info('Total embedding time: {:.2f}s'.format(model.emb_time / (epoch + 1)))
        logger.info('Total classify time: {:.2f}s'.format(model.classify_time / (epoch + 1)))

    logger.info("best_valid_acc: {:.6f}".format(best_valid))
    logger.info("test_acc: {:.6f}".format(test_result))


def test():
    cmd = argparse.ArgumentParser('The testing components of')
    cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    cmd.add_argument("--input", help="the path to the test file.")
    cmd.add_argument('--output', help='the path to the output file.')
    cmd.add_argument("--model", required=True, help="path to save model")

    args = cmd.parse_args(sys.argv[2:])
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    model_path = args.model

    model_cmd_opt = dict2namedtuple(json.load(codecs.open(os.path.join(model_path, 'config.json'), 'r', encoding='utf-8')))
    conf = json.load(open(model_cmd_opt.config, 'r'))

    torch.manual_seed(model_cmd_opt.seed)
    random.seed(model_cmd_opt.seed)
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(model_cmd_opt.seed)
        use_cuda = True

    input_batchers = {}
    input_layers = []
    for c in conf['input']:
        if c['type'] == 'embeddings':
            name = c['name']
            batcher = InputBatch(c['name'], c['field'], c['min_cut'],
                                 c.get('oov', '<oov>'), c.get('pad', '<pad>'), not c.get('cased', True), use_cuda)
            with open(os.path.join(model_path, '{0}.dic'.format(c['name'])), 'r') as fpi:
                mapping = batcher.mapping
                for line in fpi:
                    token, i = line.strip().split('\t')
                    mapping[token] = int(i)
            layer = EmbeddingLayer(name, c['dim'], batcher.mapping, fix_emb=c['fixed'],
                                   embs=None, normalize=c.get('normalize', False))
            logger.info('embedding layer for field {0} '
                        'created with {1} x {2}.'.format(c['field'], layer.n_V, layer.n_d))
            input_batchers[name] = batcher
        elif c['type'] == 'elmo':
            name = c['name']
            layer = ContextualizedWordEmbeddings(name, c['path'], use_cuda)
        else:
            raise ValueError('Unsupported input type')
        input_layers.append(layer)

    input_batchers['text'] = TextBatch(use_cuda)
    input_batchers['length'] = LengthBatch(use_cuda)

    use_semi_crf = conf["classifier"]["type"].lower() == 'semicrf'
    if use_semi_crf:
        output_batcher = SegmentBatch(use_cuda)
    else:
        output_batcher = TagBatch(use_cuda)

    id2label = {}
    with codecs.open(os.path.join(model_path, 'label.dic'), 'r', encoding='utf-8') as fpi:
        for line in fpi:
            token, i = line.strip().split('\t')
            output_batcher.mapping[token] = int(i)
            id2label[int(i)] = token
    logger.info('labels: {0}'.format(output_batcher.mapping))

    n_tags = len(id2label)
    if use_semi_crf:
        model = SemiCRFModel(conf, input_layers, model_cmd_opt.max_seg_len, n_tags, use_cuda)
    else:
        model = SeqLabelModel(conf, input_layers, n_tags, use_cuda)

    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pkl'), map_location=lambda storage, loc: storage))
    if use_cuda:
        model = model.cuda()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info('# of params: {0}'.format(params))

    if use_semi_crf:
        raw_test_input_data_, raw_test_segment_data_ = read_segment_corpus(args.input, model_cmd_opt.max_seg_len)
    else:
        raw_test_input_data_, raw_test_segment_data_ = read_tag_corpus(args.input)

    batcher = Batcher(n_tags, model_cmd_opt.max_seg_len,
                      raw_test_input_data_, raw_test_segment_data_,
                      input_batchers, output_batcher,
                      model_cmd_opt.batch_size,
                      shuffle=False, sorting=True, keep_full=True,
                      use_cuda=use_cuda)

    if args.output is not None:
        fpo = codecs.open(args.output, 'w', encoding='utf-8')
    else:
        fpo = codecs.getwriter('utf-8')(sys.stdout)

    model.eval()
    tagset = []
    orders = []
    cnt = 0
    for input_, output_, order in batcher.get():
        cnt += 1
        output, _ = model.forward(input_, output_)
        for bid in range(len(input_['text'])):
            tags = []
            output_data_ = output[bid]
            if use_semi_crf:
                for start, length, label in output_data_:
                    tag = id2label[int(label)]
                    tags.append((start, length, tag))
            else:
                lens = input_['length'][bid]
                for k in range(lens):
                    tag = id2label[output[bid][k].item()]
                    tags.append(tag)
            tagset.append(tags)
        if cnt % model_cmd_opt.report_steps == 0:
            logger.info('finished {0} x {1} batches'.format(cnt, model_cmd_opt.batch_size))
        orders.extend(order)

    for order in orders:
        if use_semi_crf:
            print(' '.join(['{0}:{1}:{2}'.format(start, length, tag) for start, length, tag in tagset[order]]),
                  file=fpo)
        else:
            print(' '.join(['{0}'.format(tag) for tag in tagset[order]]), file=fpo)
    fpo.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train()
    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        test()
    else:
        print('Usage: {0} [train|test] [options]'.format(sys.argv[0]), file=sys.stderr)
