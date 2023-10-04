#!/bin/bash

for s in shuffle no_shuffle
do
    python print_equiv_class_examples.py $s > $s.log
done
