import sys

from mosaics.rdkit_utils import SMILES_to_chemgraph
from mosaics.utils import comparison_list_to_chemgraph, padded_comparison_list

SMILES = sys.argv[1]

if len(sys.argv) > 2:
    max_nhatoms = int(sys.argv[2])
else:
    max_nhatoms = 16

cg = SMILES_to_chemgraph(SMILES)

comp_list = padded_comparison_list(cg, max_nhatoms=max_nhatoms)

print(comp_list)

cg_recovered = comparison_list_to_chemgraph(comp_list)

print(cg_recovered)

assert cg == cg_recovered
