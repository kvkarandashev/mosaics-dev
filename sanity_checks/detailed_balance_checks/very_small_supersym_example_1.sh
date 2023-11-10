#!/bin/bash

#init_str1="6#3@1:16@2:15#1@3:16@4:6#3"
#init_str2="6#3@1:7#1@2:15#1@3:15#1@4:14#3"

#final_str1="6#3@1:16@2:15#1@3:16@4:14#3"
#final_str2="6#3@1:7#1@2:15#1@3:15#1@4:6#3"

init_str1="6#3@1:7#1@2:15#1@3:7#1@4:6#3"
init_str2="6#3@1:7#1@2:15#1@3:15#1@4:14#3"

final_str1="6#3@1:7#1@2:15#1@3:7#1@4:14#3"
final_str2="6#3@1:7#1@2:15#1@3:15#1@4:6#3"


python check_detailed_balance_crossover_cmd_both_sides.py $init_str1 $init_str2 $final_str1 $final_str2 4
