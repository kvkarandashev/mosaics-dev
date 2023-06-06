from mosaics.beta_choice import gen_exp_beta_array


morfeus_args_confs = {
                            "morfeus": {
                                "num_attempts": 100,
                                "ff_type": "MMFF94",
                                "return_rdkit_obj": False,
                                "all_confs": True
                            }
                        }
morfeus_args_single = {
                            "morfeus": {
                                "num_attempts": 2,
                                "ff_type": "MMFF94",
                                "return_rdkit_obj": False,
                                "all_confs": False
                            }
                        }



def make_params_dict(selected_descriptor, min_d, max_d,strictly_in, Nsteps, possible_elements, forbidden_bonds, nhatoms_range, synth_cut_soft,synth_cut_hard, ensemble, mmff_check):
    if selected_descriptor == 'RDKit':

        params = {
        'min_d': min_d,
        'max_d': max_d,
        'strictly_in': strictly_in,
        'V_0_pot': 0.05,
        'V_0_synth': 0.05,
        'NPAR': 1,
        'Nsteps': Nsteps,
        'bias_strength': "none",
        'possible_elements': possible_elements,
        'not_protonated': None, 
        'forbidden_bonds': forbidden_bonds,
        'nhatoms_range': [int(n) for n in nhatoms_range],
        'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
        'make_restart_frequency': None,
        'rep_type': 'MolDescriptors',
        'synth_cut_soft': synth_cut_soft,
        'synth_cut_hard': synth_cut_hard,
        'rep_name': 'MolDescriptors',
        'mmff_check': mmff_check,
        "verbose": False
        }
    elif selected_descriptor == 'ECFP4':
        params = {
        'min_d': min_d,
        'max_d': max_d,
        'strictly_in': strictly_in,
        'V_0_pot': 0.05,
        'V_0_synth': 0.05,
        'NPAR': 1,
        'Nsteps': Nsteps,
        'bias_strength': "none",
        'possible_elements': possible_elements,
        'not_protonated': None, 
        'forbidden_bonds': forbidden_bonds,
        'nhatoms_range': [int(n) for n in nhatoms_range],
        'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
        'make_restart_frequency': None,
        "rep_type": "2d",
        "nBits": 2048,
        "mmff_check": True,
        "synth_cut_soft": synth_cut_soft,
        "synth_cut_hard": synth_cut_hard,
        "rep_name": "ECFP",
        "verbose": False
        }        
    
    elif selected_descriptor == 'BoB':
        params = {
            'min_d': min_d,
            'max_d': max_d,
            'strictly_in': strictly_in,
            'V_0_pot': 0.05,
            'V_0_synth': 0.05,
            'NPAR':1,
            'Nsteps': Nsteps,
            'bias_strength': "none",
            'possible_elements': possible_elements,
            'not_protonated': None, 
            'forbidden_bonds': forbidden_bonds,
            'nhatoms_range': [int(n) for n in nhatoms_range],
            'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
            'make_restart_frequency': None,
            'rep_type': '3d',
            'rep_name': 'BoB',
            'synth_cut_soft':synth_cut_soft,
            'synth_cut_hard':synth_cut_hard,
            'ensemble': ensemble,
            "verbose": False}        

    elif selected_descriptor == 'SOAP':
        params = {
        'min_d': min_d,
        'max_d': max_d,
        'strictly_in': strictly_in,
        'V_0_pot': 0.05,
        'V_0_synth': 0.05,
        'NPAR': 1,
        'Nsteps': Nsteps,
        'bias_strength': "none",
        'possible_elements': possible_elements,
        'not_protonated': None, 
        'forbidden_bonds': forbidden_bonds,
        'nhatoms_range': [int(n) for n in nhatoms_range],
        'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
        'make_restart_frequency': None,
        'rep_type': '3d',
        'synth_cut_soft': synth_cut_soft,
        'synth_cut_hard': synth_cut_hard,
        'rep_name': 'SOAP',
        'ensemble': ensemble,
        "verbose": False}
    else:
        raise ValueError('Invalid descriptor')
    
    return params