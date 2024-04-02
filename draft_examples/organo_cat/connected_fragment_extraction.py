from mosaics.random_walk import TrajectoryPoint
from mosaics.rdkit_utils import (
    canonical_connected_rdkit_list_from_tp,
    canonical_connected_SMILES_list_from_tp,
)
from mosaics.valence_treatment import str2ChemGraph

cg_str = "6#4:8#1@2:7#2:15#3"

cg = str2ChemGraph(cg_str)

tp = TrajectoryPoint(cg=cg)

print("SMILES of components:", canonical_connected_SMILES_list_from_tp(tp))
print("rdkit and SMILES of components:", canonical_connected_rdkit_list_from_tp(tp))
