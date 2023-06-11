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
from sklearn.model_selection import train_test_split
from mosaics.minimized_functions import chemspace_potentials
import rdkit.Chem.Crippen as Crippen
import rdkit
from rdkit import Chem
import pdb
import numpy as np


def latent_pred(model, T_n):
    return (T_n@model.pty_)[0]

def gradient_descent(model, T_init, y_target, gamma, N_steps, delta=1e-3):

    T_n = T_init

    for n in range(N_steps):
        # compute gradients
        grad_x = (latent_pred(model, T_n + np.array([[delta, 0]])) - latent_pred(model, T_n)) / delta
        grad_y = (latent_pred(model, T_n + np.array([[0, delta]])) - latent_pred(model, T_n)) / delta

        gradient = np.array([grad_x, grad_y]).reshape(1,-1)
        
        # compute the prediction for current T_n
        y_pred = latent_pred(model, T_n)
        print(f"{n} {y_pred}")
        # compute the error
        error = y_pred - y_target

        # adjust the gradient by the error
        gradient *= error
        
        # update rule
        T_n = T_n - gamma * gradient

    return T_n




if __name__ == "__main__":
        SAVEPATH = "/data/jan/calculations/BOSS"

        ALL_MODELS, ALL_MAEs, misc = loadpkl(f"{SAVEPATH}/pcovr.pkl")
        selected_model = ALL_MODELS[0]

        SMILES = "OCCCO"
        X = chemspace_potentials.initialize_from_smiles(SMILES)[0][0].reshape(1,-1)
        X_transformed = selected_model.transform(X)[0]
        y_pred = selected_model.predict(X)
        print(f"y_pred: {y_pred}")
        #TODO only do a few steps with gradient descent then find the closest graph
        T_opt = gradient_descent(selected_model, X_transformed,y_target=-0.15, gamma=1e-4, N_steps=10000)
        print(f"T_opt: {T_opt}")
        X_opt = selected_model.inverse_transform(T_opt)
        print(f"X_opt: {X_opt}")
        pdb.set_trace()
        exit()
        #TODO find a graph closest to T_opt