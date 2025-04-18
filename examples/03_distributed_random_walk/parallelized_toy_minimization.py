# The same example script as in 01_toy_minimization, but capitalizing on distributed parallelism.
import random
import sys

import numpy as np

from mosaics.beta_choice import gen_exp_beta_array
from mosaics.distributed_random_walk import DistributedRandomWalk
from mosaics.ext_graph_compound import str2ExtGraphCompound
from mosaics.minimized_functions import OrderSlide

random.seed(1)
np.random.seed(1)

possible_elements = ["C", "N", "O", "F", "P", "S"]

forbidden_bonds = [
    (7, 7),
    (8, 8),
    (9, 9),
    (7, 8),
    (7, 9),
    (8, 9),
    (15, 15),
    (16, 16),
    (15, 16),
]

NCPUs = 16  # 20
num_subpopulations = 20

# Whether we are using cloned betas.
if len(sys.argv) == 1:
    cloned_betas = True
else:
    cloned_betas = sys.argv[1] == "True"

if cloned_betas:
    num_exploration_betas = 16
    num_greedy_betas = 1
else:
    num_exploration_betas = 256
    num_greedy_betas = 16
betas = gen_exp_beta_array(num_greedy_betas, 8.0, num_exploration_betas, max_real_beta=0.125)

nbetas = len(betas)
num_beta_subpopulation_clones = 2
if cloned_betas:
    num_replicas = nbetas * num_subpopulations * num_beta_subpopulation_clones
else:
    num_replicas = nbetas

num_propagations = 10


randomized_change_params = {
    "max_fragment_num": 1,
    "nhatoms_range": [1, 16],
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "forbidden_bonds": forbidden_bonds,
    "not_protonated": [16],  # S not protonated
}
global_step_params = {
    "num_parallel_tempering_attempts": 64,
    "num_crossover_attempts": 16,
    "prob_dict": {"simple": 0.5, "crossover": 0.25, "tempering": 0.25},
}

# All replicas are initialized in methane.
init_egcs = [str2ExtGraphCompound("6#4") for _ in range(num_replicas)]

min_func = OrderSlide(possible_elements=possible_elements)
num_saved_candidates = 40
drw = DistributedRandomWalk(
    betas=betas,
    init_egcs=init_egcs,
    min_function=min_func,
    num_processes=NCPUs,
    num_subpopulations=num_subpopulations,
    num_internal_global_steps=200,
    global_step_params=global_step_params,
    num_saved_candidates=num_saved_candidates,
    greedy_delete_checked_paths=True,
    debug=True,
    randomized_change_params=randomized_change_params,
    subpopulation_propagation_seed=1,
    cloned_betas=cloned_betas,
    num_beta_subpopulation_clones=num_beta_subpopulation_clones,
)


for propagation_step in range(num_propagations):
    drw.propagate()
    print(propagation_step, drw.saved_candidates[0])


for i, cur_cand in enumerate(drw.saved_candidates):
    print("Best molecule", i, ":", cur_cand.tp)
    print("Value of minimized function:", cur_cand.func_val)
