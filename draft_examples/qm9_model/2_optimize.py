import numpy as np
from skmatter.decomposition import KernelPCovR
from skmatter.preprocessing import StandardFlexibleScaler as SFS
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as MAE
import matplotlib.pyplot as plt
from mosaics.minimized_functions.chemspace_potentials import QM9Dataset
from sklearn.model_selection import KFold
from tqdm import tqdm
from typing import Tuple, List
import matplotlib.pyplot as plt
from mosaics.utils import dump2pkl, loadpkl
# import train test spliot from sklearn
from mosaics.minimized_functions import chemspace_potentials
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['JOBLIB_START_METHOD'] = 'fork'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import rdkit.Chem.Crippen as Crippen
import rdkit
from rdkit import Chem
import pdb
import numpy as np
import random
from rdkit.Chem import RDConfig
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
import pdb

def latent_pred(model, T_n):
    return (T_n@model.pty_)[0]


def full_model_pred(model, T_n):
    X_n_scaled  = model.inverse_transform(T_n)
    y_n = scalar_values.inverse_transform(latent_pred(model, T_n).reshape(1,-1))
    X_n = scalar_features.inverse_transform(X_n_scaled)
    X_n_rounded = np.round(X_n).astype(int)
    return y_n[0][0], X_n, X_n_rounded

#compute error on inverted vector!

def gradient_descent(model, T_init, gamma, N_steps,  delta=1e-8, minimize=False):
    T_n = T_init
    DIM = T_init.shape[0]

    for n in range(N_steps):
        # Initialize gradients
        gradients = []

        for dim in range(DIM):
            # Create a displacement vector for the current dimension
            displacement = np.zeros(DIM)
            displacement[dim] = delta

            # Compute gradient for the current dimension
            grad = ( latent_pred(model, T_n + displacement) - latent_pred(model, T_n)) / delta
            gradients.append(grad)

        # Convert list of gradients to numpy array
        gradient = np.array(gradients).reshape(1,-1)


        if minimize:
            T_n = T_n + gamma * gradient
        else:
            T_n = T_n - gamma * gradient

        y_n, X_n = full_model_pred(model, T_n)
        pdb.set_trace()
        print("y_n", y_n)

    return T_n




def greedy_optimizer(model, T_init, delta, N_steps, y_goal):
    T_n = T_init.reshape(1, 1000)
    y_n = full_model_pred(model, T_n)[0]

    DIM = T_init.shape[0]

    for n in range(N_steps):
        # Randomly select a dimension to perturb
        dim = np.random.randint(DIM)

        # Create a displacement vector for the chosen dimension
        displacement = np.zeros(DIM)
        displacement[dim] = delta * np.random.uniform(-1, 1)

        # Perturb the current solution
        T_perturbed = T_n + displacement
        y_perturbed, X_pertubed,X_n_rounded = full_model_pred(model, T_perturbed)

        # If the perturbation improves the output, keep it
        if y_perturbed > y_n:
            T_n = T_perturbed
            y_n = y_perturbed

        # print progress
        if n % 100 == 0:
            print(f"Step: {n}, Best output: {y_n}")
        
        if y_n >= y_goal:
            print("target reached")
            print(y_n, y_goal)
            return T_n, y_n, X_n_rounded

    return T_n, y_n,X_n_rounded




def acceptance_probability(old_cost, new_cost, temperature):
    if new_cost > old_cost:
        return 1.0
    else:
        return np.exp((new_cost - old_cost) / temperature)

def swap_replicas(replicas, costs, temperatures):
    for i in range(len(replicas) - 1):
        swap_prob = (costs[i+1] - costs[i]) * (1.0/temperatures[i+1] - 1.0/temperatures[i])
        if swap_prob > np.random.rand():
            replicas[i], replicas[i+1] = replicas[i+1], replicas[i]
            costs[i], costs[i+1] = costs[i+1], costs[i]
    return replicas, costs

def parallel_tempering(model, T_init, delta, N_steps, temperatures):
    num_replicas = len(temperatures)
    replicas = [T_init.reshape(1, 1000) for _ in range(num_replicas)]
    costs = [full_model_pred(model, T)[0] for T in replicas]
    best_T = replicas[np.argmax(costs)]
    DIM = T_init.shape[0]
    y_best  = full_model_pred(model, best_T)[0]

    for n in range(N_steps):
        for i in range(num_replicas):
            # Randomly select a dimension to perturb
            dim = np.random.randint(DIM)

            # Create a displacement vector for the chosen dimension
            displacement = np.zeros(DIM)
            displacement[dim] = delta * np.random.uniform(-1, 1)

            # Perturb the current solution
            T_perturbed = replicas[i] + displacement
            y_perturbed, X_pertubed,X_n_rounded = full_model_pred(model, T_perturbed)

            # If the perturbation improves the output or is accepted by Metropolis criterion, keep it
            if np.random.rand() < acceptance_probability(costs[i], y_perturbed, temperatures[i]):
                replicas[i] = T_perturbed
                costs[i] = y_perturbed

            # Update best solution found so far
            if y_perturbed > y_best:
                best_T = T_perturbed
                y_best = y_perturbed

        # Attempt to swap configurations between replicas
        replicas, costs = swap_replicas(replicas, costs, temperatures)

        # print progress
        if n % 100 == 0:
            print(f"Step: {n}, Best output: {full_model_pred(model, best_T)[0]}")
        
        if y_best >= y_goal:
            print("target reached")
            print(y_best, y_goal)
            return best_T, y_best, X_n_rounded

    return  best_T, y_best, X_n_rounded

def load_file_into_list(filename):
    with open(filename, 'r') as file:
        lines = file.read().splitlines()
    return lines


if __name__ == "__main__":
        


        SAVEPATH = "/data/jan/calculations/BOSS"
        ALL_MODELS,ALL_MAEs ,X_train, X_test, y_train, y_test, SMILES_train, SMILES_test,misc = loadpkl(f"{SAVEPATH}/pcovr_max.pkl")
        
        scalar_features,scalar_values = misc[1], misc[2]
        selected_model = ALL_MODELS[-1]
        filename = "/home/jan/projects/MOLOPT/chemspacesampler/paper/3_biased_sampling/QM9_targets.txt"
        SMILES = load_file_into_list(filename)
        #["CC(=O)OC1=CC=CC=C1C(=O)O","CCCO" "CCO", "CCF", "C1COCCO1", "CCCCCCCC", "C1=CC=CC=C1"]
        results = []
        for smi in SMILES:
            curr_results = []
            #run through list of smiles and randomly increase initial value by 20 percent or decrease by 20 percent
            #check if design goal was achieved
            y_ref = Crippen.MolLogP(Chem.MolFromSmiles(smi) , True)
            print(y_ref)
            X = chemspace_potentials.initialize_from_smiles(smi)[0][0].reshape(1,-1)
            X = scalar_features.transform(X)
            T_init = selected_model.transform(X)[0]
            y_pred = scalar_values.inverse_transform(selected_model.predict(X))
            error = abs(y_pred - y_ref)
            y_goal =  (100*y_ref+error)
            print(f"y_pred: {y_pred}")
            print(f"y_goal: {y_goal}")
            T_opt, y_opt, X_opt  = parallel_tempering(selected_model, T_init, 1, 300000, [1e4,1e3,1e2,1, 1e-1, 1e-2, 1e-3, 1e-4])
            #T_opt, y_opt, X_opt = greedy_optimizer(selected_model, T_init, 1e-1, 200000, y_goal)
            print(f"y_opt: {y_opt}")
            #print(f"T_opt: {T_opt}")


            results.append([smi,X, y_ref,T_init, y_pred, y_goal, y_opt, T_opt, X_opt])



       # print(results)

        dump2pkl(results, f"{SAVEPATH}/optimization_results.pkl")