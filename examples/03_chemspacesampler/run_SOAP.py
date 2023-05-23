

from mosaics.minimized_functions import chemspace_potentials
from mosaics.beta_choice import gen_exp_beta_array
import pdb

if __name__ == "__main__":
    min_d, max_d = 0.0, 150.0
    params = {
        'min_d': min_d,
        'max_d': max_d,
        'NPAR': 4,
        'Nsteps': 120,
        'bias_strength': "none",
        'possible_elements': ["C", "O", "N", "F"],
        'not_protonated': None, 
        'forbidden_bonds': [(8, 9), (8,8), (9,9), (7,7)],
        'nhatoms_range': [6, 6],
        'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
        'make_restart_frequency': None,
        "verbose": True,
    }
    
    N, MOLS = chemspace_potentials.chemspacesampler_SOAP(smiles = "CCCCCC",params=params)
    print(N)
    print(MOLS)
    pdb.set_trace()