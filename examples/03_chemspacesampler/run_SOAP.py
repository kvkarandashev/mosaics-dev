from mosaics.minimized_functions import chemspace_potentials
from mosaics.beta_choice import gen_exp_beta_array

def main():
    params = {
        'min_d': 0.0,
        'max_d': 130.0,
        'NPAR': 18,
        'Nsteps': 100,
        'bias_strength': "none",
        'possible_elements': ["C", "O", "N", "F"],
        'not_protonated': None, 
        'forbidden_bonds': [(8, 9), (8, 8), (9, 9), (7, 7)],
        'nhatoms_range': [6, 6],
        'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
        'make_restart_frequency': None,
        'rep_type': '3d',
        "verbose": True,
    }

    MOLS, D= chemspace_potentials.chemspacesampler_SOAP(smiles="CCCCCC", params=params)
    print(MOLS)
    print(D)

if __name__ == "__main__":
    main()