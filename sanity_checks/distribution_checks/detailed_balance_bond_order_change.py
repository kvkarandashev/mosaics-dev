# Demonstrates that setting add_heavy_atom_chain and remove_heavy_atom probabilities differently does not affect detailed balance.
import random

import numpy as np

from mosaics.ext_graph_compound import ExtGraphCompound
from mosaics.modify import (
    TrajectoryPoint,
    change_bond_order,
    change_bond_order_valence,
    change_valence,
)
from mosaics.random_walk import RandomWalk
from mosaics.valence_treatment import str2ChemGraph

random.seed(1)
np.random.seed(1)

possible_elements = ["C", "S"]

forbidden_bonds = None

num_MC_steps = 100000

bias_coeff = None

randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": [1, 5],
    "final_nhatoms_range": [1, 5],
    "linear_scaling_elementary_mutations": True,
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "forbidden_bonds": forbidden_bonds,
    "change_prob_dict": {
        change_bond_order: 1.0,
        change_bond_order_valence: 1.0,
        change_valence: 1.0,
    },
}

init_str = "16@1@4:6#2@2:6#2@3:6#2@4:6#2"  # "16@1@2:6#2@2:6#2" "16@1@4:6#2@2:6#2@3:6#2@4:6#2"

available_strs = [
    init_str,
    "16@1@2@4:6#2@2:6@3:6#2@4:6#2",
    "16#1@1@2@4:6#2@2:6#1@3:6#2@4:6#2",
]

num_replicas = 1
init_egcs = [ExtGraphCompound(chemgraph=str2ChemGraph(init_str)) for _ in range(num_replicas)]

restricted_tps = [
    TrajectoryPoint(cg=str2ChemGraph(chemgraph_str)) for chemgraph_str in available_strs
]

rw = RandomWalk(
    init_egcs=init_egcs,
    bias_coeff=bias_coeff,
    randomized_change_params=randomized_change_params,
    visit_num_count_acceptance=True,
    num_replicas=num_replicas,
    keep_histogram=True,
    no_exploration=True,
    restricted_tps=restricted_tps,
)
for MC_step in range(num_MC_steps):
    # If changing randomized_change_params is required mid-simulation they can be updated via *.change_rdkit arguments
    rw.MC_step_all()
    print(MC_step, rw.cur_tps)

for tp in rw.histogram:
    if tp.num_visits is not None:
        print(tp, tp.num_visits)
