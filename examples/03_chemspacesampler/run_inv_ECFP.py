from mosaics.minimized_functions import chemspace_potentials
from mosaics.beta_choice import gen_exp_beta_array

def main():
    params = {
        'V_0_pot': 0.05,
        'NPAR': 8,
        'max_d': 3.0,
        'Nsteps': 1000,
        'bias_strength': "none",
        'possible_elements': ["C", "O", "N", "F"],
        'not_protonated': None, 
        'forbidden_bonds': None,
        'nhatoms_range': [6, 6],
        'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
        'make_restart_frequency': None,
        "rep_type": "2d",
        "nBits": 2048,
        'rep_name': 'inv_ECFP',
        "verbose": True
    }

    smiles_init, smiles_target = "CCCCCC", "C1=CC=CC=C1"
    X_target, _, _ = chemspace_potentials.initialize_from_smiles(smiles_target)
    MOLS, D = chemspace_potentials.chemspacesampler_inv_ECFP(smiles_init,X_target, params=params)    
    print(MOLS)
    print(D)

if __name__ == "__main__":
    main()