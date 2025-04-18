import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error

# import train test spliot from sklearn
from sklearn.model_selection import KFold, train_test_split
from skmatter.decomposition import KernelPCovR
from skmatter.preprocessing import StandardFlexibleScaler as SFS
from tqdm import tqdm

from mosaics.utils import dump2pkl, loadpkl

random.seed(42)


def GridSearchCV_KernelPCovR(
    X: np.ndarray,
    y: np.ndarray,
    mixing_values: List[float],
    gamma_values: List[float],
    DIMENSIONS: int,
    cv: int = 5,
) -> Tuple[KernelPCovR, float, dict]:
    # Create the parameter grid
    param_grid = [(mixing, gamma) for mixing in mixing_values for gamma in gamma_values]

    # Initialize KFold cross-validation
    kf = KFold(n_splits=cv)

    # Placeholder for best parameters and score
    best_params = None
    best_score = float("inf")

    # Grid search with cross-validation
    for params in tqdm(param_grid):
        mixing, gamma = params
        scores = []

        # Cross-validation
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Fit the model
            model = KernelPCovR(
                mixing=mixing,
                n_components=DIMENSIONS,
                regressor=KernelRidge(kernel="laplacian", gamma=gamma),
                kernel="laplacian",
                gamma=gamma,
                fit_inverse_transform=True,
            )
            model.fit(X_train, y_train)

            # Predict and calculate error
            y_pred = model.predict(X_test)
            score = mean_squared_error(y_test, y_pred)
            scores.append(score)

        # Average cross-validation score for this parameter set
        avg_score = np.mean(scores)

        # If this score is better than the previous best, update best score and parameters
        if avg_score < best_score:
            best_score = avg_score
            best_params = params

    best_model = KernelPCovR(
        mixing=best_params[0],
        n_components=DIMENSIONS,
        regressor=KernelRidge(kernel="laplacian", gamma=best_params[1]),
        kernel="laplacian",
        gamma=best_params[1],
        fit_inverse_transform=True,
    )

    # Fit the best model to the full data
    best_model.fit(X, y)

    return best_model, best_score, {"mixing": best_params[0], "gamma": best_params[1]}


if __name__ == "__main__":
    NEW_FIT, PLOT, NEW_FIT_ALL = True, False, False
    PROPERTY = 4

    SAVEPATH = "/data/jan/calculations/BOSS"
    data = np.load(f"{SAVEPATH}/qm9_processed.npz", allow_pickle=True)
    X, y, SMILES = data["X"], data["y"], data["SMILES"]

    X = np.stack(data["X"][:, 0])
    X_train, X_test, y_train, y_test, SMILES_train, SMILES_test = train_test_split(
        X, y, SMILES, test_size=0.2, random_state=42
    )
    y_train = y_train[:, PROPERTY].reshape(-1, 1)
    y_test = y_test[:, PROPERTY].reshape(-1, 1)
    N_train = [2**i for i in range(8, 17)][:7]
    ALL_DIMENSIONS = [2]  # [2, 5, 10, 20, 100]
    mixing_values = [0.05, 0.1, 0.3]
    # gamma_values  = np.logspace(-2, 0, 3)  # Adjust these as needed
    gamma_values = np.logspace(-2, 2, 10)  # Property 0

    if NEW_FIT:
        scalar_features = SFS()
        scalar_values = SFS()

        X_train = scalar_features.fit_transform(X_train)
        y_train = scalar_values.fit_transform(y_train)
        X_test = scalar_features.transform(X_test)

        ALL_MODELS, ALL_MAEs = [], []
        for DIMENSIONS in ALL_DIMENSIONS:
            MAEs = []
            for n in N_train:
                pcovr, best_score, best_params = GridSearchCV_KernelPCovR(
                    X_train[:n],
                    y_train[:n],
                    mixing_values,
                    gamma_values,
                    DIMENSIONS=DIMENSIONS,
                )
                y_hat_cov = scalar_values.inverse_transform(pcovr.predict(X_test))
                error_cov = MAE(y_test, y_hat_cov)
                MAEs.append(error_cov)
                ALL_MODELS.append(pcovr)
                print(f"Best parameters: {best_params}")
                print(f"{n, error_cov}")

            ALL_MAEs.append(MAEs)

        dump2pkl(
            [ALL_MODELS, ALL_MAEs, [N_train, scalar_features, scalar_values]],
            f"{SAVEPATH}/pcovr_{PROPERTY}.pkl",
        )

    if NEW_FIT_ALL:
        n = 40000  # 32768
        scalar_features = SFS()
        scalar_values = SFS()

        X_train = scalar_features.fit_transform(X_train)
        y_train = scalar_values.fit_transform(y_train)
        X_test = scalar_features.transform(X_test)

        ALL_MODELS, ALL_MAEs = [], []
        MAEs = []

        N_train = len(X_train)
        pcovr, best_score, best_params = GridSearchCV_KernelPCovR(
            X_train[:n], y_train[:n], mixing_values, gamma_values, DIMENSIONS=1000
        )
        y_hat_cov = scalar_values.inverse_transform(pcovr.predict(X_test))
        error_cov = MAE(y_test, y_hat_cov)
        print(f"Best parameters: {best_params}")
        print(f"{n, error_cov}")
        ALL_MAEs.append(error_cov)
        MAEs.append(error_cov)
        ALL_MODELS.append(pcovr)
        print(f"Best parameters: {best_params}")
        print(f"{n, error_cov}")

        ALL_MAEs.append(MAEs)

        dump2pkl(
            [
                ALL_MODELS,
                ALL_MAEs,
                X_train,
                X_test,
                y_train,
                y_test,
                SMILES_train,
                SMILES_test,
                [N_train, scalar_features, scalar_values],
            ],
            f"{SAVEPATH}/pcovr_max_{PROPERTY}.pkl",
        )

    if PLOT:
        ALL_MODELS, ALL_MAEs, misc = loadpkl(f"{SAVEPATH}/pcovr_{PROPERTY}.pkl")
        N_train = misc[0]
        # Plot learning curves
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, DIMENSIONS in enumerate(ALL_DIMENSIONS):
            plt.plot(N_train, ALL_MAEs[i], label=f"Dimensions = {DIMENSIONS}")

        ax.set_xlabel("Number of training samples")
        ax.set_ylabel("Mean Absolute Error")
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.legend()
        plt.title("Learning curves for different dimensions")
        plt.grid(True)
        plt.savefig("test2.png")
        plt.close()
        # only plot the final MAEs for each dimension
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.plot(ALL_DIMENSIONS, [ALL_MAEs[i][-1] for i in range(len(ALL_MAEs))])
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.savefig("test3.png")
