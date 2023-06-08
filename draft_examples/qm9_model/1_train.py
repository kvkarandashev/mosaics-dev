import numpy as np
from skmatter.decomposition import KernelPCovR
from skmatter.preprocessing import StandardFlexibleScaler as SFS
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as MAE
from skmatter.decomposition import PCovR
import matplotlib.pyplot as plt
from mosaics.minimized_functions.chemspace_potentials import QM9Dataset
from sklearn.model_selection import KFold
from tqdm import tqdm
from typing import Tuple, List
import matplotlib.pyplot as plt
from mosaics.utils import dump2pkl, loadpkl
# import train test spliot from sklearn
from sklearn.model_selection import train_test_split
import pdb



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
    best_score = float('inf')

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

    data = np.load("/data/jan/calculations/BOSS/qm9_processed.npz", allow_pickle=True)
    X_train, X_test, y_train, y_test = train_test_split(data["X"], data["y"], test_size=0.2, random_state=42)
    
    DIMENSIONS  = 2
    scalar_features = SFS()
    scalar_values   = SFS()
    
    X_train = scalar_features.fit_transform(X_train)
    y_train = scalar_values.fit_transform(y_train)
    X_test = scalar_features.transform(X_test)
    


    mixing_values = np.linspace(0.05, 0.9, 9)  # Adjust these as needed
    gamma_values = np.logspace(-3, 1, 7)  # Adjust these as needed
    pcovr, best_score, best_params = GridSearchCV_KernelPCovR(X_train, y_train, mixing_values, gamma_values, DIMENSIONS = DIMENSIONS)
    print(f"Best parameters: {best_params}")



    y_hat_cov = scalar_values.inverse_transform(pcovr.predict(X_test))

    error_cov = MAE(y_test, y_hat_cov)
    print(f"MAE COV: {error_cov}")

    

    # Transform X_test using the fitted model
    X_transformed = pcovr.transform(X_test)

    # Create a scatter plot of the first two components
    # Color by y_test (assuming y_test is categorical; if it's continuous, this will be a color gradient)
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_test, cmap='viridis')


    num_points = 100

    # Get the min and max of X_transformed along both axes
    x_min, y_min = X_transformed.min(axis=0)
    x_max, y_max = X_transformed.max(axis=0)

    # Generate a sequence of points along each axis
    x_values = np.linspace(x_min, x_max, num_points)
    y_values = np.linspace(y_min, y_max, num_points)

    # Create a grid of points
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    # If you need the grid points as a list of (x, y) pairs, you can reshape:
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    

    X_inverted_new = pcovr.inverse_transform( grid_points)
    y_new = scalar_values.inverse_transform(pcovr.predict(X_inverted_new))
    
    sc2 = plt.scatter(grid_points[:, 0], grid_points[:, 1], marker="*", c=y_new, cmap='viridis', alpha=0.1)
    plt.colorbar(sc)
    plt.xlabel('First component')
    plt.ylabel('Second component')
    plt.title('First two components after KernelPCovR')
    plt.savefig('test.png')
    plt.show()
    plt.close()

    pdb.set_trace()
    exit()


    #where x smaller -1  and y largetr 0.225
    interesting_points = np.argwhere( (grid_points[:,0] < -1.0)& (grid_points[:,1] > 0.225)) 
    X_interesting_inverted = pcovr.inverse_transform( grid_points[interesting_points])
    pdb.set_trace()