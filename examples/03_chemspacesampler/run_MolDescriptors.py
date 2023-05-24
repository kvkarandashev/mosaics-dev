from mosaics.minimized_functions import chemspace_potentials
from mosaics.beta_choice import gen_exp_beta_array


params = {
    'min_d': 0,
    'max_d': 2.5,
    'NPAR': 1,
    'Nsteps': 100,
    'bias_strength': "none",
    'possible_elements': ["C", "O", "N", "F"],
    'not_protonated': None, 
    'forbidden_bonds': [(8, 9), (8, 8), (9, 9), (7, 7)],
    'nhatoms_range':[13, 13],
    'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
    'make_restart_frequency': None,
    'rep_type': 'MolDescriptors',
    'synth_cut': 2,
    "verbose": True
}
if __name__ == "__main__":
    MOLS, D = chemspace_potentials.chemspacesampler_MolDescriptors(smiles="CC(=O)OC1=CC=CC=C1C(=O)O", params=params)
    print(MOLS)
    print(D)