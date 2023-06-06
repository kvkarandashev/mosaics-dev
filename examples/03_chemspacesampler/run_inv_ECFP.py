from mosaics.minimized_functions import chemspace_potentials
from mosaics.beta_choice import gen_exp_beta_array
import pdb  
def main():
    params = {
        'V_0_pot': 0.05,
        'NPAR': 48,
        'max_d': 0.1,
        'strictly_in': True,
        'Nsteps': 500,
        'bias_strength': "stronger",
        'possible_elements': ["C", "O", "N", "F"],
        'not_protonated': None, 
        'forbidden_bonds': None,
        'nhatoms_range': [6, 20],
        'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
        'make_restart_frequency': None,
        "rep_type": "2d",
        "nBits": 2048,
        'rep_name': 'inv_ECFP',
        'strategy': 'contract',
        'd_threshold': 0.1,
        'Nparts': 9,
        'growth_factor': 1.5,
        "verbose": False
    }

    smiles_init, smiles_target ="O=C(O)c1cc1=O", "CC(C)CC(N(C)C)C1(C2=CC=C(Cl)C=C2)CCC1"
    X_target, _, _ = chemspace_potentials.initialize_from_smiles(smiles_target,nBits=params['nBits'])
    MOLS, D = chemspace_potentials.chemspacesampler_inv_ECFP(smiles_init,X_target, params=params)    
    print("MOLS", MOLS)
    print("D", D)
if __name__ == "__main__":
    main()
