#!/bin/bash

echo "FIRST:"
python print_all_reconnectors.py "6#1@1@2:6#1@3:6#1@4:6#1@5:6#1@5:6#1" "6#1@1@2:7@3:6#1@4:6#1@5:6#1@5:6#1"
echo "SECOND:"
python print_all_reconnectors.py "6#1@1@2:6#1@3:6#1@4:6#1@5:7@5:6#1" "6#1@1@2:7@3:6#1@4:6#1@5:6#1@5:6#1"
echo "THIRD"
python print_all_reconnectors.py "6@1@2@3:6#2@4:6#1@5:6#1@6:6#2@7:6#1@7:6#1@7:6" "6#1@1@2@3:6#2@4:6#2@5:6#1@6:6#2@7:6#2@7:6#1@7:6#1"