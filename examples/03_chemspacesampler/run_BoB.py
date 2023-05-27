from mosaics.minimized_functions import chemspace_potentials
from mosaics.beta_choice import gen_exp_beta_array

def main():
    params = {
        'min_d': 0.0,
        'max_d': 80.0,
        'NPAR': 2,
        'Nsteps': 20,
        'bias_strength': "none",
        'possible_elements': ["C", "O", "N", "F"],
        'not_protonated': None, 
        'forbidden_bonds': [(8, 9), (8, 8), (9, 9), (7, 7)],
        'nhatoms_range': [13,16],
        'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
        'make_restart_frequency': None,
        'rep_type': '3d',
        'rep_name': 'BoB',
        'synth_cut': 2,
        'ensemble': True,
        "verbose": True,
    }

    MOLS, D= chemspace_potentials.chemspacesampler_BoB(smiles="CC(=O)OC1=CC=CC=C1C(=O)O", params=params)
    print(MOLS)
    print(D)

if __name__ == "__main__":
    main()