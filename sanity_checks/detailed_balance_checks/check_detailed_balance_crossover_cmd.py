# Check that probability balance is well calculated for larger molecules.
import random
import sys

import numpy as np

from mosaics.minimized_functions.toy_problems import ChargeSum
from mosaics.modify import randomized_cross_coupling
from mosaics.random_walk import TrajectoryPoint
from mosaics.test_utils import check_prop_probability
from mosaics.valence_treatment import str2ChemGraph

chemgraph_strings = [
    sys.argv[1],
    sys.argv[2],
]

if len(sys.argv) > 3:
    seed = int(sys.argv[3])
else:
    seed = 1

random.seed(seed)
np.random.seed(seed)

cgs = [str2ChemGraph(cg_str) for cg_str in chemgraph_strings]

tps = tuple([TrajectoryPoint(cg=cg) for cg in cgs])

nhatoms_range = [1, 9]

num_new_pairs = 4

new_pairs = []

attempts_to_generate = 40000

randomized_change_params = {
    "nhatoms_range": nhatoms_range,
    "cross_coupling_smallest_exchange_size": 2,
    "linear_scaling_crossover_moves": True,
}


for _ in range(attempts_to_generate):
    new_cg_pair, _ = randomized_cross_coupling(cgs, **randomized_change_params)
    if new_cg_pair is None:
        continue
    tnew_pair = tuple([TrajectoryPoint(cg=cg) for cg in new_cg_pair])
    if tnew_pair not in new_pairs:
        new_pairs.append(tnew_pair)
        if len(new_pairs) == num_new_pairs:
            break


minimized_function = ChargeSum()

ln2 = np.log(2.0)

betas = [ln2, ln2 / 2.0]
# betas = [ln2, ln2]

num_attempts = 400  # 40000


print("BETAS:", betas)
check_prop_probability(
    tps,
    new_pairs,
    randomized_change_params=randomized_change_params,
    num_attempts=num_attempts,
    min_function=minimized_function,
    betas=betas,
    bin_size=0.01,
)
