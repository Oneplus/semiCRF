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


n_count = 0
n_correct = 0
for gold_data, pred_data in zip(gold_dataset, pred_dataset):
    for gold_line, pred_line in zip(get_labels(gold_data), get_labels(pred_data)):
        gold_tag = gold_line.split()[0]
        pred_tag = pred_line.split()[0]
        n_count += 1
        if gold_tag == pred_tag:
            n_correct += 1
print(n_correct / n_count)
