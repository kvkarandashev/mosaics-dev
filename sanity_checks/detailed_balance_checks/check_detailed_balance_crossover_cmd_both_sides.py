# Check that probability balance is well calculated for larger molecules.
import random
import numpy as np
from mosaics.random_walk import TrajectoryPoint
from mosaics.test_utils import check_prop_probability
from mosaics.valence_treatment import str2ChemGraph
from mosaics.minimized_functions.toy_problems import ChargeSum
import sys


def get_tps(chemgraph_strings):
    cgs = [str2ChemGraph(cg_str) for cg_str in chemgraph_strings]
    tps = tuple([TrajectoryPoint(cg=cg) for cg in cgs])
    return tps


init_chemgraph_strings = [
    sys.argv[1],
    sys.argv[2],
]

new_chemgraph_strings = [sys.argv[3], sys.argv[4]]

if len(sys.argv) > 5:
    seed = int(sys.argv[5])
else:
    seed = 1

forbidden_bonds = len(sys.argv) > 6

random.seed(seed)
np.random.seed(seed)

init_tps = get_tps(init_chemgraph_strings)

new_tps = get_tps(new_chemgraph_strings)

nhatoms_range = [1, 9]

randomized_change_params = {
    "nhatoms_range": nhatoms_range,
    "crossover_smallest_exchange_size": 1,
    "linear_scaling_crossover_moves": True,
}
if forbidden_bonds:
    randomized_change_params["forbidden_bonds"] = [
        (7, 7),
        (7, 8),
        (8, 8),
        (9, 9),
        (8, 9),
        (7, 9),
    ]

minimized_function = ChargeSum()

ln2 = np.log(2.0)

betas = [ln2, ln2 / 2.0]
# betas = [ln2, ln2]

num_attempts = 200000  # 4000 40000


print("BETAS:", betas)
check_prop_probability(
    init_tps,
    new_tps,
    randomized_change_params=randomized_change_params,
    num_attempts=num_attempts,
    min_function=minimized_function,
    betas=betas,
    nprocs=20,
    bin_size=0.01,
)
