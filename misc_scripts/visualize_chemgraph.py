import os
import sys

from mosaics.rdkit_draw_utils import draw_chemgraph_to_file
from mosaics.valence_treatment import InvalidAdjMat, str2ChemGraph

chemgraph_str = sys.argv[1]

if len(sys.argv) > 2:
    filename_prefix = sys.argv[2]
else:
    filename_prefix = "chemgraph_visualization_"

try:
    cg = str2ChemGraph(chemgraph_str)
except InvalidAdjMat:
    print("Invalid string")
    quit()

i = 0

while True:
    filename = filename_prefix + str(i) + ".png"
    if os.path.isfile(filename):
        i += 1
    else:
        draw_chemgraph_to_file(
            cg,
            filename,
            size=(300, 200),
        )
        break
