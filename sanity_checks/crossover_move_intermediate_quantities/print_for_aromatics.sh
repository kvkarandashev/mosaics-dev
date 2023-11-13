#!/bin/bash

cgs=( "6#1@1@2:6#1@3:6#1@4:6#1@5:6#1@5:6#1" "6#1@1@2:7@3:6#1@4:6#1@5:6#1@5:6#1")
for cg in ${cgs[@]}
do
    python print_affected_bond_statuses.py $cg
done
