from mosaics.beta_choice import gen_exp_beta_array

def make_params_dict(selected_descriptor, min_d, max_d, Nsteps, possible_elements, forbidden_bonds, nhatoms_range, synth_cut_soft,synth_cut_hard, ensemble, mmff_check):
    if selected_descriptor == 'RDKit':

        params = {
        'min_d': min_d,
        'max_d': max_d,
        'NPAR': 2,
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
        'mmff_check': mmff_check,
        "verbose": True
        }
    elif selected_descriptor == 'ECFP4':
        params = {
        'min_d': min_d,
        'max_d': max_d,
        'NPAR': 2,
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
        "verbose": True
        }        
    
    elif selected_descriptor == 'BoB':
        params = {
            'min_d': min_d,
            'max_d': max_d,
            'NPAR':2,
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
            "verbose": True}        

    elif selected_descriptor == 'SOAP':
        params = {
        'min_d': min_d,
        'max_d': max_d,
        'NPAR': 2,
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
        "verbose": True}
    else:
        raise ValueError('Invalid descriptor')
    
    return params