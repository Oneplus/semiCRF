#!/usr/bin/env python
from __future__ import print_function
from __future__ import division 
import sys
gold_dataset = open(sys.argv[1], 'r').read().strip().splitlines()
pred_dataset = open(sys.argv[2], 'r').read().strip().splitlines()
assert len(gold_dataset) == len(pred_dataset)


def get_labels(line):
    tokens = line.strip().split('|||')[-1].strip().split()
    return tokens


def get_segments(lines):
    segs = set()
    start, tag = None, None
    for i, line in enumerate(lines):
        label = line.split()[0].lower()
        if label.startswith('b-') or label.startswith('s-') or label == 'o':
            if start is not None:
                segs.add((start, i - 1, tag))
            if label.startswith('b-') or label.startswith('s-'):
                start, tag = i, label.split('-', 1)[1]
            else:
                start, tag = None, None
    if start is not None:
        segs.add((start, len(lines) - 1, tag))
    return segs


n_pred, n_gold, n_correct = 0, 0, 0
for gold_data, pred_data in zip(gold_dataset, pred_dataset):
    gold_segs = get_segments(get_labels(gold_data))
    pred_segs = get_segments(get_labels(pred_data))
    for gold_seg in gold_segs:
        if gold_seg in pred_segs:
            n_correct += 1
    n_pred += len(pred_segs)
    n_gold += len(gold_segs)
p = n_correct / n_pred
r = n_correct / n_gold
print(2 * p * r / (p + r))
