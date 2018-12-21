#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import sys


def get_segments(filename):
    ret = []
    for line in open(filename, 'r').read().strip().splitlines():
        fields = line.split('|||')[-1].strip().split()
        ret.append(set([field for field in fields if field.split(':')[-1].lower() != 'o']))
    return ret


gold_dataset = get_segments(sys.argv[1])
pred_dataset = get_segments(sys.argv[2])
assert len(gold_dataset) == len(pred_dataset)

n_pred, n_gold, n_correct = 0, 0, 0
for gold_data, pred_data in zip(gold_dataset, pred_dataset):
    for gold_seg in gold_data:
        if gold_seg in pred_data:
            n_correct += 1
    n_pred += len(gold_data)
    n_gold += len(gold_data)
p = n_correct / n_pred
r = n_correct / n_gold
print(2 * p * r / (p + r))
