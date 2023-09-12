#!/bin/bash

init_str1="9@1:15@2@3:6#1@4:6#1@4:6#2"
init_str2="8@1@2:6@2@3:6@4:6#1@4:6#1"

final_str1="8@1@2:6@2@3:6@3:6#2"
final_str2="9@1:15@2@3:6#1@4:6#1@5:6@5@6:6@6:8"

python check_detailed_balance_crossover_cmd_both_sides.py $init_str1 $init_str2 $final_str1 $final_str2 4
