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

def gradient_descent(model, T_init, y_target, gamma, N_steps, delta=1e-6):
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

        # Compute the prediction for current T_n
        y_pred = latent_pred(model, T_n)
        #print(f"{n} {y_pred}")

        # Compute the error
        error = y_pred - y_target
        if n % 1000 == 0:
            print(f"error: {error}")

        # Adjust the gradient by the error
        gradient *= error

        # Update rule
        T_n = T_n - gamma * gradient

    return T_n




if __name__ == "__main__":
        SAVEPATH = "/data/jan/calculations/BOSS"

        ALL_MODELS, ALL_MAEs, misc = loadpkl(f"{SAVEPATH}/pcovr.pkl")
        scalar_features,scalar_values = misc[1], misc[2]
        
        selected_model = ALL_MODELS[-1]

        SMILES = "OCCCO"
        X = chemspace_potentials.initialize_from_smiles(SMILES)[0][0].reshape(1,-1)
        X = scalar_features.transform(X)
        X_transformed = selected_model.transform(X)[0]
        y_pred = scalar_values.inverse_transform(selected_model.predict(X))
        print(f"y_pred: {y_pred}")
        #TODO only do a few steps with gradient descent then find the closest graph
        y_target = scalar_values.transform(np.array([-4.9]).reshape(1,-1))
        T_opt = gradient_descent(selected_model, X_transformed,y_target=y_target, gamma=1e-4, N_steps=40000)
        print(f"T_opt: {T_opt}")
        X_opt = scalar_features.inverse_transform( selected_model.inverse_transform(T_opt))
        y_opt = scalar_values.inverse_transform(selected_model.predict( selected_model.inverse_transform(T_opt)))
        print(f"X_opt_suggested: {X_opt}")
        print(f"y_opt: {y_opt}")
        X_suggested = np.round(X_opt[0]).astype(int)
        pdb.set_trace()
        exit()
        #TODO find a graph closest to T_opt