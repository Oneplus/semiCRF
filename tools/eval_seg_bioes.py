#!/usr/bin/env python
from __future__ import print_function
from __future__ import division 
import sys


gold_dataset = open(sys.argv[1], 'r').read().strip().splitlines()
pred_dataset = open(sys.argv[2], 'r').read().strip().splitlines()
assert len(gold_dataset) == len(pred_dataset)


def get_labels(line):
    tokens = line.strip().split('|||')[-1].strip().split()
    segs = set()
    start = None
    for i, label in enumerate(tokens):
        label = label.lower()
        if label == 'b' or label == 's' or label == 'o':
            if start is not None:
                segs.add((start, i - 1))
            if label == 'b' or label == 's':
                start = i
            else:
                start = None
    if start is not None:
        segs.add((start, len(tokens) - 1))
    return segs


n_corr, n_gold, n_pred = 0, 0, 0
for gold_data, pred_data in zip(gold_dataset, pred_dataset):
    gold_segs = get_labels(gold_data)
    pred_segs = get_labels(pred_data)
    for gold_seg in gold_segs:
        if gold_seg in pred_segs:
            n_corr += 1
    n_gold += len(gold_segs)
    n_pred += len(pred_segs)
p = n_corr / n_pred
r = n_corr / n_gold
print(2 * p * r / (p + r))
