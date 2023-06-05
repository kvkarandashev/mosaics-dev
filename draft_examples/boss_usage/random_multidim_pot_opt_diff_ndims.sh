#!/bin/bash

for i in $(seq 1 20)
do
    d=ndims_$i
    mkdir -p $d
    cd $d
    python ../random_multidim_pot_opt.py $i > opt.log &
    cd ..
done
