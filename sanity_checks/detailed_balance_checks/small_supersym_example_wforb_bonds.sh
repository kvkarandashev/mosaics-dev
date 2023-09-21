#!/bin/bash

init_str1="6#1@1@3@5:7#1@2:6#2@7:7#1@4:6#2@7:6#2@6:6#2@7:6#1"
init_str2="6#1@1@3@5:8@2:6#2@7:6#2@4:6#2@7:6#2@6:6#2@7:6#1"

final_str1="6#1@1@3@5:6#2@2:6#2@7:6#2@4:6#2@7:6#2@6:6#2@7:6#1"
final_str2="6#1@1@3@5:7#1@2:6#2@7:6#2@4:8@7:7#1@6:6#2@7:6#1"

python check_detailed_balance_crossover_cmd_both_sides.py $init_str1 $init_str2 $final_str1 $final_str2 4 #forb_bonds
