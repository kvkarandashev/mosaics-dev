from mosaics.minimized_functions import chemspace_potentials
from mosaics.beta_choice import gen_exp_beta_array

def main():
    params = {
        'min_d': 0.0,
        'max_d': 120.0,
        'strictly_in': True,
        'V_0_pot': 0.05,
        'V_0_synth': 0.05,
        'NPAR':2,
        'Nsteps': 5,
        'bias_strength': "none",
        'possible_elements': ["C", "O", "N", "F", "Si"],
        'not_protonated': None, 
        'forbidden_bonds': [(8, 9), (8, 8), (9, 9), (7, 7)],
        'nhatoms_range': [21,25],
        'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
        'make_restart_frequency': None,
        'rep_type': '3d',
        'rep_name': 'BoB',
        'synth_cut_soft':7,
        'synth_cut_hard':9,
        'ensemble': False,
        "verbose": True,
    }
    print(params['betas'])
    exit()
    MOLS, D= chemspace_potentials.chemspacesampler_BoB(smiles="CCCCCC1=CC2=C(C(O)=C1)C1C=C(C)CCC1C(C)(C)O2", params=params)
    print(MOLS)
    print(D)

if __name__ == "__main__":
    main()