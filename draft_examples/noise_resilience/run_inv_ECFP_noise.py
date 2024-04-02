import random

import numpy as np

from mosaics.beta_choice import gen_exp_beta_array
from mosaics.minimized_functions import chemspace_potentials

random.seed(1337)


def flip_bits(X, p):
    # Flattening the array to make the bit flipping easier
    flat_X = X.flatten()

    # Calculate the number of bits to flip
    n_flip = int(len(flat_X) * p)

    # Choose n_flip random indices to flip
    flip_indices = np.random.choice(len(flat_X), size=n_flip, replace=False)

    # Flip the bits at the chosen indices
    flat_X[flip_indices] = 1 - flat_X[flip_indices]

    # Reshape the array back to its original shape
    new_X = flat_X.reshape(X.shape)

    return new_X


def main():
    params = {
        "V_0_pot": 0.5,
        "NPAR": 4,
        "max_d": 0.1,
        "strictly_in": False,
        "Nsteps": 2000,
        "bias_strength": "none",
        "pot_type": "parabola",
        "possible_elements": ["C", "O", "N", "F"],
        "not_protonated": None,
        "forbidden_bonds": [(8, 9), (8, 8), (9, 9), (7, 7)],
        "nhatoms_range": [1, 20],
        "betas": gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
        "make_restart_frequency": None,
        "rep_type": "2d",
        "nBits": 2048,
        "rep_name": "inv_ECFP",
        "strategy": "modify_pot",
        "d_threshold": 0.05,
        "Nparts": 20,
        "growth_factor": 1.5,
        "verbose": False,
    }

    smiles_init, smiles_target = "C", "OC1=CC=CC=C1"
    X_target, _, _ = chemspace_potentials.initialize_from_smiles(
        smiles_target, nBits=params["nBits"]
    )
    # pdb.set_trace()

    D_orig, D_best = [], []
    LEVELS = np.arange(0, 10, 2) / 100

    for p in LEVELS:
        MOLS, D = chemspace_potentials.chemspacesampler_inv_ECFP(
            smiles_init, flip_bits(X_target, p), params=params
        )
        mol_best, d_best = MOLS[0], D[0]
        X_best, _, _ = chemspace_potentials.initialize_from_smiles(mol_best, nBits=params["nBits"])
        d_orig = chemspace_potentials.tanimoto_distance(X_target, X_best)
        D_orig.append(d_orig)
        D_best.append(d_best)
        print("p = ", p, "d_best = ", d_best, d_orig, mol_best)

    np.savez_compressed(
        "noise_resilience_inv_ECFP.npz", LEVELS=LEVELS, D_orig=D_orig, D_best=D_best
    )


if __name__ == "__main__":
    main()
