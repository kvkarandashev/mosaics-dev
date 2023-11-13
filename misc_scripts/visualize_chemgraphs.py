import sys, os
from subprocess import run

script = os.path.dirname(__file__) + "/visualize_chemgraph.py"

for cg_str in sys.argv[1:]:
    run(["python", script, cg_str])
