from mosaics.minimized_functions import chemspace_potentials
from mosaics.beta_choice import gen_exp_beta_array
import pdb  
def main():
    params = {
        'V_0_pot': 0.01,
        'NPAR': 1,
        'max_d': 2.0,
        'Nsteps': 1000,
        'bias_strength': "none",
        'possible_elements': ["C", "O", "N", "F"],
        'not_protonated': None, 
        'forbidden_bonds': None,
        'nhatoms_range': [12, 14],
        'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
        'make_restart_frequency': None,
        "rep_type": "2d",
        "nBits": 512,
        'rep_name': 'inv_ECFP',
        'strategy': 'contract',
        'Nparts': 5,
        'growth_factor': 1.5,
        "verbose": False
    }

    #"CC(O)OC1CC=C(C(=O)O)CC1"
    smiles_init, smiles_target = "CCCCCCCCCCCC", "CC(=O)OC1=CC=CC=C1C(=O)O"
    X_target, _, _ = chemspace_potentials.initialize_from_smiles(smiles_target,nBits=params['nBits'])
    MOLS, D = chemspace_potentials.chemspacesampler_inv_ECFP(smiles_init,X_target, params=params)    

if __name__ == "__main__":
    main()
