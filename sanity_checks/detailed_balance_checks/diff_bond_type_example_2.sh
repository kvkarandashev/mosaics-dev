#!/bin/bash

init_str1="6@1@2@3:6#2@4:7@5:6#1@6:6#2@7:6#1@7:6#1@7:6"
init_str2="14@1@2@3:6#2@4:6#1@5:6#1@6:6#2@7:6#1@7:6#1@7:6"

final_str1="14@1@2@3:6#2@4:6#1@5:6#1@6:6#2@7:6#1@7:7@7:6"
final_str2="6@1@2@3:6#2@4:6#1@5:6#1@6:6#2@7:6#1@7:6#1@7:6"

python check_detailed_balance_crossover_cmd_both_sides.py $init_str1 $init_str2 $final_str1 $final_str2 4
