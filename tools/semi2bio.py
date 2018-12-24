#!/usr/bin/env python
import sys


for line in open(sys.argv[1]):
    fields = line.strip().split("|||")[-1].strip().split()
    k = 0
    for field in fields:
        start_id, length, label = field.split(':')
        for i in range(int(length)):
            if label.lower() == 'o':
                print(label)
            else:
                if i == 0:
                    print('B-{0}'.format(label))
                else:
                    print('I-{0}'.format(label))

    print()
