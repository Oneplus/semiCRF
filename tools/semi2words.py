#!/usr/bin/env python
import sys


for line in open(sys.argv[1]):
    fields = line.strip().split("|||")[0].strip().split()
    k = 0
    for field in fields:
        print(field)
    print()
