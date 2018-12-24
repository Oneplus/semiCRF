#!/bin/bash
python semi2words.py $1 > w
python semi2bio.py $1 > x
python semi2bio.py $2 > y
paste w x y | sed -E 's/^	*$//g' | python ./conlleval.py
rm w x y
