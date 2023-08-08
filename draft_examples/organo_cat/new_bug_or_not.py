# This script makes an example optimization run similar to the ones whose results are presented in the original MOSAiCS paper.
from mosaics.random_walk import RandomWalk
from mosaics.beta_choice import gen_exp_beta_array
from mosaics.rdkit_utils import SMILES_to_egc, canonical_SMILES_from_tp
from mosaics.rdkit_draw_utils import draw_chemgraph_to_file
from mosaics.minimized_functions.chemspace_potentials import (
    potential_ECFP,
    initialize_from_smiles,
)
import rdkit
import os
from rdkit import Chem
import pdb
import random, numpy

from rdkit.Chem import RDConfig
import rdkit.Chem.Crippen as Crippen
from rdkit.Contrib.SA_Score import sascorer
import sys

random.seed(2)
numpy.random.seed(2)

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
# SMILES of the molecule from which we will start optimization.
init_SMILES = "CC(=O)OC1=C2OC3C(OC(C)=O)C=CC4C5CC(=C2C43CCN5C)C=C1"
# "CC(=O)OC1=C2OC3C(OC(C)=O)C=CC4C5CC(=C2C43CCN5C)C=C1"
# Parameter of the QM9* chemical space over which the property is optimized.
possible_elements = ["C", "N", "O", "F", "Cl", "Br", "S"]
forbidden_bonds = [(7, 7), (7, 8), (8, 8), (7, 9), (8, 9), (9, 9)]
not_protonated = [8, 9]
nhatoms_range = [1, 30]
# Define minimized function using parameters discussed in the MOSAiCS paper.


# None corresponds to greedy optimization, other betas are used in a Metropolis scheme.
betas = gen_exp_beta_array(4, 1.0, 40, max_real_beta=1e-2)
print("Chosen betas:", betas)
# Each replica starts at methane.
init_egcs = [SMILES_to_egc(init_SMILES) for _ in betas]

# On average, 300-400 global MC steps is enough to find the minimum over QM9*.
num_MC_steps = 1000
make_restart_frequency = 100
# Soft exit is triggered by creating a file called "EXIT".
soft_exit_check_frequency = 100
# We do not use the history-dependent vias here.
bias_coeff = None

randomized_change_params = {
    "max_fragment_num": 2,
    "nhatoms_range": nhatoms_range,
    "final_nhatoms_range": nhatoms_range,
    "possible_elements": possible_elements,
    "bond_order_changes": [-1, 1],
    "forbidden_bonds": forbidden_bonds,
    "not_protonated": not_protonated,
    "added_bond_orders": [1, 2, 3],
}
global_change_params = {
    "num_parallel_tempering_tries": 128,
    "num_genetic_tries": 32,
    "prob_dict": {"simple": 0.3, "genetic": 0.6, "tempering": 0.1},
}

params = {
    "min_d": 0.0,
    "max_d": 6.0,
    "NPAR": 1,
    "Nsteps": 100,
    "bias_strength": "none",
    "possible_elements": possible_elements,
    "not_protonated": None,
    "forbidden_bonds": [(8, 9), (8, 8), (9, 9), (7, 7)],
    "nhatoms_range": nhatoms_range,
    "betas": gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
    "make_restart_frequency": None,
    "rep_type": "2d",
    "nBits": 2048,
    "mmff_check": True,
    "synth_cut_soft": 3,
    "synth_cut_hard": 7,
    "V_0_pot": 0.05,
    "V_0_synth": 0.05,
    "verbose": False,
}

X, rdkit_init, egc = initialize_from_smiles(init_SMILES)
minimized_function = potential_ECFP(X, params=params)


rw = RandomWalk(
    init_egcs=init_egcs,
    bias_coeff=bias_coeff,
    randomized_change_params=randomized_change_params,
    betas=betas,
    min_function=minimized_function,
    min_function_name="organo_cat",
    make_restart_frequency=make_restart_frequency,
    soft_exit_check_frequency=make_restart_frequency,
    restart_file="restart_file.pkl",
    num_saved_candidates=10,
    keep_histogram=True,
    greedy_delete_checked_paths=True,
)

print("Started candidate search.")
print("# MC_step # Minimum value found # Minimum SMILES")
for MC_step in range(num_MC_steps):
    rw.global_random_change(**global_change_params)
    # Print information about the current known molecule with best minimized function value.
    cur_best_candidate = rw.saved_candidates[0]
    cur_min_func_val = cur_best_candidate.func_val
    cur_min_SMILES = canonical_SMILES_from_tp(cur_best_candidate.tp)
    print(MC_step, cur_min_func_val, cur_min_SMILES)

print("Finished candidate search.")
rw.make_restart()

print("Number of minimized function calls:", minimized_function.call_counter)
