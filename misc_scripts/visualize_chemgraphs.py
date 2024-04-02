import os
import subprocess
import sys
from subprocess import run

script = os.path.dirname(__file__) + "/visualize_chemgraph.py"

for cg_str in sys.argv[1:]:
    try:
        run(["python", script, cg_str])
    except subprocess.CalledProcessError:
        print("run encountered problems")
