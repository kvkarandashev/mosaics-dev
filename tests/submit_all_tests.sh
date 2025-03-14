#!/bin/bash

test_scripts=(
                "01_toy_minimization/toy_minimization.py"
                "03_distributed_random_walk/parallelized_toy_minimization.py"
                "03_distributed_random_walk/parallelized_toy_minimization_other_seed.py"
                "04_blind_optimization_protocol/toy_minimization_beta_optimization.py"
            )

CPU_nums=("1" "10" "10" "16")

for test_id in $(seq 4)
do
    arr_id=$((test_id-1))
    CPU_num=${CPU_nums[$arr_id]}
    test_script=${test_scripts[$arr_id]}
    jobname=$(echo $test_script | tr '/' '_')
    spython --CPUs=$CPU_num --OMP_NUM_THREADS=1 $test_script $jobname
done
