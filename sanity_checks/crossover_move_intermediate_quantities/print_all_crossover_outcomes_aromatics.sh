#!/bin/bash

echo "FIRST:"
python print_all_crossover_outcomes.py "6#1@1@2:6#1@3:6#1@4:6#1@5:6#1@5:6#1" "6#1@1@2:7@3:6#1@4:6#1@5:6#1@5:6#1"
echo "SECOND:"
python print_all_crossover_outcomes.py "6#1@1@2:6#1@3:6#1@4:6#1@5:7@5:6#1" "6#1@1@2:7@3:6#1@4:6#1@5:6#1@5:6#1"
