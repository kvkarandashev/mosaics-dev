from mosaics.minimized_functions import chemspace_potentials
import inversion_potentials
from mosaics.beta_choice import gen_exp_beta_array
from mosaics.utils import dump2pkl, loadpkl
import pdb  
from rdkit import Chem
import rdkit.Chem.Crippen as Crippen
import matplotlib.pyplot as plt
import numpy as np

def count_heavy_atoms(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    num_heavy_atoms = molecule.GetNumHeavyAtoms()
    return num_heavy_atoms

def resample():

    
    #ALL_MODELS,ALL_MAEs ,X_train, X_test, y_train, y_test, SMILES_train, SMILES_test,misc = loadpkl(f"{SAVEPATH}/pcovr_max.pkl")
    #scalar_features,scalar_values = misc[1], misc[2]
    #selected_model = ALL_MODELS[-1]
    ALL_MODELS,ALL_MAEs,misc = loadpkl(f"{SAVEPATH}/pcovr.pkl")
    #ALL_MODELS,ALL_MAEs ,X_train, X_test, y_train, y_test, SMILES_train, SMILES_test,misc  = loadpkl(f"{SAVEPATH}/pcovr_max.pkl")
    #pdb.set_trace()
    scalar_features,scalar_values = misc[1], misc[2]
    selected_model = ALL_MODELS[0]

    latent_space_results = loadpkl(f"{SAVEPATH}/optimization_results.pkl")

    params = {
        'V_0_pot': 0.05,
        'NPAR': 24,
        'max_d': 0.1,
        'strictly_in': False,
        'Nsteps': 20,
        'model' : selected_model,
        'scalar_features': scalar_features,
        'scalar_values': scalar_values,
        'bias_strength': "stronger",
        'pot_type': 'parabola',
        'possible_elements': ["C", "O", "N", "F"],
        'not_protonated': None, 
        'forbidden_bonds': None,
        'nhatoms_range': [1, 20],
        'V_0_synth': 0.05,
        'synth_cut_soft':3,
        'synth_cut_hard':9,
        'mmff_check': True,
        'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
        'make_restart_frequency': None,
        "rep_type": "2d",
        "nBits": 2048,
        'rep_name': 'inv_latent',
        "verbose": False
    }



    RESULTS = []
    for res in latent_space_results:
        #try:
        CURR_RESULTS = []
        SMILES_init,X_init, y_ref,T_init, y_pred, y_goal, y_opt, T_opt, X_opt  = res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8]
        T_init = np.zeros_like(T_init)
        #heavy_at = count_heavy_atoms(SMILES_init)
        #params["nhatoms_range"] = [heavy_at, heavy_at]

        MOLS, D = inversion_potentials.chemspacesampler_inv_latent(SMILES_init,T_init,T_opt,params=params)  
        y_sampled = []
        for mol in MOLS:
            X_sampled, _, _ = chemspace_potentials.initialize_from_smiles(mol, nBits=params['nBits'])
            y_pred_ML  = scalar_values.inverse_transform( selected_model.predict(scalar_features.transform(X_sampled)))[0][0]
            y_pred_ref = Crippen.MolLogP(Chem.MolFromSmiles(mol) , True)
            y_sampled.append([y_pred_ML, y_pred_ref])

        y_sampled = np.array(y_sampled)
        CURR_RESULTS.append([SMILES_init,X_init, y_ref,T_init, y_pred, y_goal, y_opt, T_opt, X_opt, MOLS, D, y_sampled])
        print("MOLS", MOLS)
        print("D", D)
        RESULTS.append(CURR_RESULTS)
        #except Exception as e:
        #    print("Error")
        #    print(e)
    
    RESULTS = np.array(RESULTS)
    dump2pkl(RESULTS, f"{SAVEPATH}/optimization_results_inv_latent.pkl")

def analyze():
    RESULTS = loadpkl(f"{SAVEPATH}/optimization_results_inv_latent.pkl")
    
    ALL_PRED = []
    ALL_SAMPLED = []
    for res in RESULTS:
        res = res[0]
        SMILES_init = res[0]
        y_ref = res[2]
        y_pred = res[4][0][0]
        MOLS = res[9]
        D = res[10]
        y_sampled = res[11][0]
        print(SMILES_init, MOLS[0])
        #print(y_pred, y_ref, y_sampled,  np.mean(res[11][:,0][:10]),  np.mean(res[11][:,1][:10]) )


        print(y_pred,y_ref, y_sampled[0], y_sampled[1])
        ALL_PRED.append(y_pred)
        ALL_SAMPLED.append(y_sampled[0])

    #make box plots of all_differences
    plt.plot(np.arange(len(ALL_SAMPLED)),ALL_PRED, "o")
    plt.plot(np.arange(len(ALL_SAMPLED)),ALL_SAMPLED, "x")
    plt.show()


    pdb.set_trace()



if __name__ == "__main__":
    SAVEPATH = "/data/jan/calculations/BOSS"
    #resample()
    analyze()
