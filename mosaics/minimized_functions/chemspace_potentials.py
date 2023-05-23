# Standard library imports
import glob
import random
import tempfile
import shutil
import copy

# Third party imports
import numpy as np
from numpy.linalg import norm
from rdkit import Chem
from rdkit.Chem import DataStructs, rdMolDescriptors
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm

try:
    from ase import Atoms
    from dscribe.descriptors import SOAP
except:
    print("local_space_sampling: ase or dscribe not installed")

# Local application/library specific imports
from mosaics import RandomWalk
from mosaics.random_walk import TrajectoryPoint, ordered_trajectory_from_restart
from mosaics.beta_choice import gen_exp_beta_array
from mosaics.utils import loadpkl
from mosaics.data import *
from mosaics.rdkit_utils import (
    chemgraph_to_canonical_rdkit, 
    RdKitFailure,
    SMILES_to_egc
)
from mosaics.minimized_functions.morfeus_quantity_estimates import (
    morfeus_coord_info_from_tp,
    morfeus_FF_xTB_code_quants,
)



def trajectory_point_to_canonical_rdkit(tp_in, SMILES_only=False):
    """
    Converts a trajectory point to a canonical RDKit molecule.

    Args:
        tp_in: A trajectory point (typically an instance of the TrajectoryPoint class)
        SMILES_only: If True, only the SMILES string of the molecule is returned. Otherwise, 
                     a RDKit molecule object is returned. Default is False.

    Returns:
        RDKit molecule object or SMILES string, based on the value of SMILES_only.
    """
    return chemgraph_to_canonical_rdkit(tp_in.egc.chemgraph, SMILES_only=SMILES_only)


def gen_soap(crds, chgs, species):
    """
    Generate the Smooth Overlap of Atomic Positions (SOAP) descriptor for a molecule.

    Args:
        crds: A list or array of atomic coordinates.
        chgs: A list or array of atomic charges.
        species: A list of species for atoms present in the molecule.
    
    Returns:
        A SOAP descriptor for the molecule.

    Note:
        The average output is a global of the molecule. See https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html for more details.
    """
    average_soap = SOAP(
        r_cut=6.0,
        n_max=8,
        l_max=6,
        average="inner",
        species=species,
        sparse=False,
    )

    molecule = Atoms(numbers=chgs, positions=crds)
    
    return average_soap.create(molecule)


def get_boltzmann_weights(energies, T=300):
    """
    Calculate the Boltzmann weights for a set of energies at a given temperature.

    Parameters
    ----------
    energies : np.array of shape (n,) 
        1-D array containing the energies of 'n' states.
    T : float, optional
        Temperature in Kelvin. Default is 300 K.

    Returns
    -------
    boltzmann_weights : np.array of shape (n,)
        1-D array containing the Boltzmann weights corresponding to the input energies.
    """
    beta = 1/(K_B_KCAL_PER_MOL_PER_K * T)
    boltzmann_weights = np.exp(-energies * beta)
    boltzmann_weights /= np.sum(boltzmann_weights)
    return boltzmann_weights


def fml_rep(COORDINATES, NUC_CHARGES, WEIGHTS, possible_elements=["C", "O", "N", "F"]):
    """
    Calculate the FML (Free Energy Machine Learning) representation, which is the Boltzmann-weighted SOAP representation.

    Parameters
    ----------
    COORDINATES : np.array of shape (n, m, 3)
        3-D array containing the coordinates of 'm' atoms for 'n' states.
    NUC_CHARGES : np.array of shape (m,)
        1-D array containing the nuclear charges of 'm' atoms.
    WEIGHTS : np.array of shape (n,)
        1-D array containing the weights for 'n' states.
    possible_elements : list of strings, optional
        List of possible elements present in the molecules. Default is ["C", "O", "N", "F"].

    Returns
    ------- 
    fml_rep : np.array of shape (N,) where 'N' is the dimension of the SOAP vector
        FML representation of the input molecular system.
    """
    X = []
    
    for i in range(len(COORDINATES)):
        X.append(gen_soap(COORDINATES[i], NUC_CHARGES, possible_elements))
    X = np.array(X)
    X = np.average(X, axis=0, weights=WEIGHTS)
    return X


def ExplicitBitVect_to_NumpyArray(fp_vec):
    """
    Convert the RDKit fingerprint to a numpy array.

    Parameters
    ----------
    fp_vec : rdkit.DataStructs.cDataStructs.ExplicitBitVect
        The RDKit fingerprint vector.

    Returns
    -------
    fp2 : numpy.ndarray
        The fingerprint vector converted to a numpy array.
    """
    fp2 = np.zeros((0,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp_vec, fp2)
    return fp2


def extended_get_single_FP(smi, nBits=2048, useFeatures=True):
    """
    Get the fingerprint of a molecule in numpy array form.

    Parameters
    ----------
    smi : str
        SMILES string of the molecule.
    nBits : int, optional
        Length of the fingerprint. Default is 2048.
    useFeatures : bool, optional
        Whether to use feature information when generating the fingerprint. Default is True.

    Returns
    -------
    x : numpy.ndarray
        The fingerprint of the molecule as a numpy array.
    """
    x = ExplicitBitVect_to_NumpyArray(
        get_single_FP(smi, nBits=nBits, useFeatures=useFeatures)
    )
    return x


def get_single_FP(mol, nBits=2048, useFeatures=True):
    """
    Compute the fingerprint of a molecule.

    Parameters
    ----------
    mol : str or rdkit.Chem.rdchem.Mol
        SMILES string or RDKit molecule object.
    nBits : int, optional
        Length of the fingerprint. Default is 2048.
    useFeatures : bool, optional
        Whether to use feature information when generating the fingerprint. Default is True.

    Returns
    -------
    fp_mol : rdkit.DataStructs.cDataStructs.ExplicitBitVect
        The fingerprint of the molecule.
    """
    fp_mol = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol,
        radius=4,
        nBits=nBits,
        useFeatures=useFeatures
    )
    return fp_mol


def get_all_FP(SMILES, **kwargs):
    """
    Return a list of fingerprints for all the molecules in the list of SMILES.

    Parameters
    ----------
    SMILES : list of str
        List of SMILES strings.

    Returns
    -------
    X : numpy.ndarray
        An array of fingerprints for all the molecules.
    """
    X = []
    for smi in SMILES:
        X.append(extended_get_single_FP(smi, **kwargs))
    return np.array(X)


class potential_SOAP:
    """
    Class to represent a potential using Smooth Overlap of Atomic Positions (SOAP).
    """
    def __init__(
        self,
        X_init,
        Q_init,
        gamma=70.0,
        sigma=80.0,
        possible_elements=["C", "O", "N", "F", "H"],
        verbose=False,
    ):
        """
        Initializes the potential_SOAP class.
        
        Parameters:
        X_init (np.array): Initial positions of particles.
        Q_init (np.array): Initial charges.
        gamma (float): A parameter for flat_parabola_potential function.
        sigma (float): A parameter for flat_parabola_potential function.
        possible_elements (list): List of possible atomic elements.
        verbose (bool): Verbosity flag.
        """

        self.X_init = X_init
        self.Q_init = Q_init
        self.gamma = gamma
        self.sigma = sigma
        self.possible_elements = possible_elements
        self.verbose = verbose
        
        
        self.morfeus_output = {"morfeus": morfeus_coord_info_from_tp}
        self.potential = self.flat_parabola_potential



    def fml_distance(self,coords,charges,energies):
        """
        Calculates the FML distance.
        
        Parameters:
        coords (np.array): Positions of particles.
        charges (np.array): Charges.
        energies (np.array): Energies.
        
        Returns:
        float: FML distance.
        """
        
        X_test = self.repfct(coords, charges,energies)
        return norm(X_test - self.X_init)


    def flat_parabola_potential(self, d):
        """
        A function to describe the potential as a flat parabola.
        
        Parameters:
        d (float): Distance.
        
        Returns:
        float: Potential value.
        """

        if d < self.gamma:
            return 0.05 * (d - self.gamma) ** 2
        if self.gamma <= d <= self.sigma:
            return 0
        if d > self.sigma:
            return 0.05 * (d - self.sigma) ** 2

    def __call__(self, trajectory_point_in):
        """
        Calculates the potential energy of a trajectory point.
        
        Parameters:
        trajectory_point_in (TrajectoryPoint): Trajectory point.
        
        Returns:
        float: Potential energy of the trajectory point.
        """

        try:
            output = trajectory_point_in.calc_or_lookup(
                self.morfeus_output,
                kwargs_dict={
                    "morfeus": {
                        "num_attempts": 100,
                        "ff_type": "MMFF94",
                        "return_rdkit_obj": False,
                        "all_confs": True
                    }
                },
            )["morfeus"]
            
            coords = output["coordinates"]

            charges = output["nuclear_charges"]
            SMILES = output["canon_rdkit_SMILES"]
            X_test = fml_rep(coords, charges, output["rdkit_Boltzmann"],possible_elements= self.possible_elements)
        except Exception as e:
            print(e)
            print("Error in 3d conformer sampling")
            return None
            
        
        distance = np.linalg.norm(X_test - self.X_init)
        V = self.potential(distance)

        if self.verbose:
            print(SMILES, distance, V)
        return V
    


class potential_ECFP:

    """
    Sample local chemical space of the inital compound
    with input representation X_init.
    """

    def __init__(
        self,
        X_init,
        sigma=1.0,
        gamma=1,
        nbits=4096,
        verbose=False
    ):
        self.X_init = X_init
        self.sigma = sigma
        self.gamma = gamma
        self.nbits = nbits
        self.verbose = verbose

        self.canonical_rdkit_output = {
            "canonical_rdkit": trajectory_point_to_canonical_rdkit
        }

        self.potential = self.flat_parabola_potential
        


    def flat_parabola_potential(self, d):
        """
        Flat parabola potential. Allows sampling within a distance basin
        interval of I in [gamma, sigma]. The potential is given by:
        """

        if d < self.gamma:
            return 0.05 * (d - self.gamma) ** 2
        if self.gamma <= d <= self.sigma:
            return 0
        if d > self.sigma:
            return 0.05 * (d - self.sigma) ** 2

    def __call__(self, trajectory_point_in):

        rdkit_mol, canon_SMILES = trajectory_point_in.calc_or_lookup(
            self.canonical_rdkit_output
        )["canonical_rdkit"]

        if rdkit_mol is None:
            raise RdKitFailure
            
        X_test = extended_get_single_FP(rdkit_mol, nBits=self.nbits) 
        d = norm(X_test - self.X_init)
        V = self.potential(d)

        if self.verbose:
            print(canon_SMILES, d, V)

        return V

    def evaluate_point(self, trajectory_point_in):
        """
        Evaluate the function on a list of trajectory points
        """

        rdkit_mol, canon_SMILES = trajectory_point_in.calc_or_lookup(
            self.canonical_rdkit_output
        )["canonical_rdkit"]

        X_test = extended_get_single_FP(
            rdkit_mol, nBits=self.nbits
        )
        d = norm(X_test - self.X_init)
        V = self.potential(d)

        return V, d

    def evaluate_trajectory(self, trajectory_points):
        """
        Evaluate the function on a list of trajectory points
        """

        values = []
        for trajectory_point in trajectory_points:
            values.append(self.evaluate_point(trajectory_point))

        return np.array(values)
    

def mc_run(init_egc,min_func,min_func_name, respath,label, params):
    seed = int(str(hash(label))[1:8])
    np.random.seed(1337+seed)
    random.seed(1337+seed)

    num_replicas = len(params['betas'])
    init_egcs = [copy.deepcopy(init_egc) for _ in range(num_replicas)]

    bias_coeffs = {"none": None, "weak": 0.2, "stronger": 0.4}
    bias_coeff = bias_coeffs[params['bias_strength']]
    vbeta_bias_coeff = bias_coeffs[params['bias_strength']]

    randomized_change_params = {
        "max_fragment_num": 1,
        "nhatoms_range": params['nhatoms_range'],
        "possible_elements": params['possible_elements'],
        "bond_order_changes": [-1, 1],
        "forbidden_bonds": params['forbidden_bonds'],
        "not_protonated": params['not_protonated'],
        "added_bond_orders": [1, 2, 3],
    }
    global_change_params = {
        "num_parallel_tempering_tries": 128,
        "num_genetic_tries": 32,
        "prob_dict": {"simple": 0.6, "genetic": 0.2, "tempering": 0.2},
    }

    rw = RandomWalk(
                    init_egcs=init_egcs,
                    bias_coeff=bias_coeff,
                    vbeta_bias_coeff=vbeta_bias_coeff,
                    randomized_change_params=randomized_change_params,
                    betas=params['betas'],
                    min_function=min_func,
                    min_function_name=min_func_name,
                    keep_histogram=True,
                    keep_full_trajectory=False,
                    make_restart_frequency=params['make_restart_frequency'],
                    soft_exit_check_frequency=params['make_restart_frequency'],
                    restart_file=respath+f"/{label}.pkl",
                    max_histogram_size=None,
                    linear_storage=True,
                    greedy_delete_checked_paths=True,
                    debug=True,
    )
    for MC_step in range(params["Nsteps"]):
        rw.global_random_change(**global_change_params)

    rw.ordered_trajectory()
    rw.make_restart(tarball=True)


def mc_run_QM9(init_egc,Nsteps, min_func,min_func_name, respath,label):
    bias_strength = "none"
    possible_elements = ["C", "O", "N", "F"]
    not_protonated = None

    forbidden_bonds= [(8, 9), (8,8), (9,9), (7,7)]

    nhatoms_range = [1, 9]
    betas = gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0)
    num_replicas = len(betas)
    init_egcs = [copy.deepcopy(init_egc) for _ in range(num_replicas)]

    make_restart_frequency = None
    num_MC_steps = Nsteps

    bias_coeffs = {"none": None, "weak": 0.2, "stronger": 0.4}
    bias_coeff = bias_coeffs[bias_strength]
    vbeta_bias_coeff = bias_coeffs[bias_strength]


    randomized_change_params = {
        "max_fragment_num": 1,
        "nhatoms_range": nhatoms_range,
        "possible_elements": possible_elements,
        "bond_order_changes": [-1, 1],
        "forbidden_bonds": forbidden_bonds,
        "not_protonated": not_protonated,
        "added_bond_orders": [1, 2, 3],
    }
    global_change_params = {
        "num_parallel_tempering_tries": 128,
        "num_genetic_tries": 32,
        "prob_dict": {"simple": 0.6, "genetic": 0.2, "tempering": 0.2},
    }

    

    rw = RandomWalk(
                    init_egcs=init_egcs,
                    bias_coeff=bias_coeff,
                    vbeta_bias_coeff=vbeta_bias_coeff,
                    randomized_change_params=randomized_change_params,
                    betas=betas,
                    min_function=min_func,
                    min_function_name=min_func_name,
                    keep_histogram=True,
                    keep_full_trajectory=False,
                    make_restart_frequency=make_restart_frequency,
                    soft_exit_check_frequency=make_restart_frequency,
                    restart_file=respath+f"/{label}.pkl",
                    max_histogram_size=None,
                    linear_storage=True,
                    greedy_delete_checked_paths=True,
                    debug=True,
    )
    for MC_step in range(num_MC_steps):
        rw.global_random_change(**global_change_params)

    rw.ordered_trajectory()
    rw.make_restart(tarball=True)


def initialize_fml_from_smiles(smiles, mode=True):
    rdkit_H =Chem.AddHs(Chem.MolFromSmiles(smiles))
    smiles_with_H = Chem.MolToSmiles(rdkit_H)
    init_egc  = SMILES_to_egc(smiles_with_H)
    trajectory_point = TrajectoryPoint(init_egc)
    morfeus_output = {"morfeus": morfeus_coord_info_from_tp}
    
    if mode:
        output = trajectory_point.calc_or_lookup(
                    morfeus_output,
                    kwargs_dict={
                        "morfeus": {
                            "num_attempts": 100,
                            "ff_type": "MMFF94s",
                            "return_rdkit_obj": False,
                            "all_confs": True
                        }
                    },
                )["morfeus"]
    else:
        output = None

    return init_egc, output,rdkit_H

def compute_values(smi,**kwargs):

    quantities = [
        "solvation_energy",
        "HOMO_LUMO_gap",
    ]


    kwargs = {
        "ff_type": "MMFF94",
        "remaining_rho": 0.9,
        "num_conformers": 32,
        "num_attempts": 16,
        "solvent": "water",
        "quantities": quantities,
    }

    egc = SMILES_to_egc(smi)
    tp = TrajectoryPoint(egc=egc)
    results = morfeus_FF_xTB_code_quants(tp, **kwargs)

    val = results["mean"]["solvation_energy"]
    std = results["std"]["solvation_energy"]

    val_gap = results["mean"]["HOMO_LUMO_gap"]
    std_gap = results["std"]["HOMO_LUMO_gap"]

    try:
        properties = [float(val), float(std), float(val_gap), float(std_gap)]
    except:
        if not os.path.exists("errorfile.txt"):
            errorfile = open("errorfile.txt", "w")
            errorfile.close()
        errorfile = open("errorfile.txt", "a")
        errorfile.write(f"{smi} {results}/n")
        errorfile.close()
        properties = [np.nan, np.nan, np.nan, np.nan]
    return properties

def compute_values_parallel(SMILES, njobs=10):
    PROPERTIES = [Parallel(n_jobs=njobs)(delayed(compute_values)(smi) for smi in SMILES)]
    return PROPERTIES[0]

def initialize_from_smiles(SMILES, bits=2048):
    smiles_with_H = Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(SMILES)))
    output =  SMILES_to_egc(smiles_with_H)
    rdkit_init = Chem.AddHs(Chem.MolFromSmiles(smiles_with_H))
    X_init = get_all_FP([rdkit_init],nBits=bits) 
    return X_init,rdkit_init, output





class Analyze_Chemspace:
    def __init__(self, path,rep_type="2d" ,full_traj=False, verbose=False):

        """
        mode : either optimization of dipole and gap = "optimization" or
               sampling locally in chemical space = "sampling"
        """

        self.path = path
        self.rep_type = rep_type
        self.results = glob.glob(path)
        self.verbose = verbose
        self.full_traj = full_traj
        print(self.results)

    def parse_results(self):
        if self.verbose:
            print("Parsing results...")
            Nsim = len(self.results)
            print("Number of simulations: {}".format(Nsim))

        ALL_HISTOGRAMS = []
        ALL_TRAJECTORIES = []


        if self.full_traj:
            for run in tqdm(self.results, disable=not self.verbose):
                
                restart_data = loadpkl(run, compress=True)

                HISTOGRAM = self.to_dataframe(restart_data["histogram"])
                HISTOGRAM = HISTOGRAM.sample(frac=1).reset_index(drop=True)
                ALL_HISTOGRAMS.append(HISTOGRAM)
                if self.full_traj:
                    traj = np.array(ordered_trajectory_from_restart(restart_data))
                    CURR_TRAJECTORIES = []
                    for T in range(traj.shape[1]):
                        sel_temp = traj[:, T]
                        TRAJECTORY = self.to_dataframe(sel_temp)
                        CURR_TRAJECTORIES.append(TRAJECTORY)
                    ALL_TRAJECTORIES.append(CURR_TRAJECTORIES)
        else:
            ALL_HISTOGRAMS = Parallel(n_jobs=8)(delayed(self.process_run)(run) for run in self.results)


        self.ALL_HISTOGRAMS, self.ALL_TRAJECTORIES = ALL_HISTOGRAMS, ALL_TRAJECTORIES
        self.GLOBAL_HISTOGRAM = pd.concat(ALL_HISTOGRAMS)
        self.GLOBAL_HISTOGRAM = self.GLOBAL_HISTOGRAM.drop_duplicates(subset=["SMILES"])

        return self.ALL_HISTOGRAMS, self.GLOBAL_HISTOGRAM, self.ALL_TRAJECTORIES


    def process_run(self, run):
        restart_data = loadpkl(run, compress=True)

        HISTOGRAM = self.to_dataframe(restart_data["histogram"])
        HISTOGRAM = HISTOGRAM.sample(frac=1).reset_index(drop=True)
        return HISTOGRAM




    def convert_from_tps(self, mols):
        """
        Convert the list of trajectory points molecules to SMILES strings.
        tp_list: list of molecudfles as trajectory points
        smiles_mol: list of rdkit molecules
        """


        if self.rep_type == "2d":
            SMILES = []
            VALUES = []

            
            for tp in mols:
                curr_data = tp.calculated_data
                SMILES.append(curr_data["canonical_rdkit"][-1])
                VALUES.append(curr_data["chemspacesampler"])


            VALUES = np.array(VALUES)

            return SMILES, VALUES
        if self.rep_type == "3d":
            """
            (Pdb) tp.calculated_data["morfeus"].keys()
            dict_keys(['coordinates', 'nuclear_charges', 'canon_rdkit_SMILES', 'rdkit_energy', 'rdkit_degeneracy', 'rdkit_Boltzmann'])
            """



            SMILES = []
            VALUES = []
            
            for tp in mols:
                curr_data = tp.calculated_data
                SMILES.append(curr_data["morfeus"]["canon_rdkit_SMILES"])
                VALUES.append(curr_data["chemspacesampler"])

            VALUES = np.array(VALUES)

            return SMILES, VALUES


    def compute_representations(self, MOLS, nBits):
        """
        Compute the representations of all unique smiles in the random walk.
        """

        X = get_all_FP(MOLS, nBits=nBits)
        return X

    def compute_projection(self, MOLS, nBits=2048, clustering=False, projector="PCA"):
        


        X = self.compute_representations(MOLS, nBits=nBits)
        if projector == "UMAP":
            import umap
            reducer = umap.UMAP(random_state=42)
        if projector == "PCA":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)

        reducer.fit(X)
        X_2d = reducer.transform(X)

        if clustering == False:
            return X_2d


        else:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            sil_scores = []
            for n_clusters in range(2, 6):
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(X_2d)
                labels = kmeans.labels_
                sil_scores.append(silhouette_score(X_2d, labels))

            optimal_n_clusters = np.argmax(sil_scores) + 2 #5
            print(optimal_n_clusters)
            kmeans = KMeans(n_clusters=optimal_n_clusters,tol=1e-8,max_iter=500,random_state=0)
            kmeans.fit(X_2d)
            labels = kmeans.labels_
            
            # Assign each molecule to the closest cluster
            clusters = [[] for _ in range(optimal_n_clusters)]
            cluster_X_2d = [[] for _ in range(optimal_n_clusters)]
            for i, label in enumerate(labels):
                clusters[label].append(MOLS[i])
                cluster_X_2d[label].append(X_2d[i])
            
            # Find most representative molecule for each cluster
            representatives = []
            indices = []

            X_rep_2d = []
            for label, cluster in enumerate(clusters):
                
                distances = [np.linalg.norm(x - kmeans.cluster_centers_[label]) for x in cluster_X_2d[label]]
                #
                representative_index = np.argmin(distances)
                representatives.append(cluster[representative_index])
                indices.append(representative_index)
                X_rep_2d.append(cluster_X_2d[label][representative_index])

            print(kmeans.cluster_centers_)
            SMILES_rep = [Chem.MolToSmiles(mol) for mol in representatives]
            X_rep_2d = np.array(X_rep_2d)
            return X_2d ,clusters,cluster_X_2d, X_rep_2d,SMILES_rep,reducer


    def to_dataframe(self, obj):
        """
        Convert the trajectory point object to a dataframe
        and extract xTB values if available.
        """

        df = pd.DataFrame()

        SMILES, VALUES = self.convert_from_tps(obj)
        df["SMILES"] = SMILES
        df["VALUES"] = VALUES
        df = df.dropna()
        df = df.reset_index(drop=True)
        return df

    def count_shell_value(self, curr_h, return_mols=False):
        in_interval = curr_h["VALUES"] == 0.0

        if return_mols:
            return in_interval.sum(), curr_h["SMILES"][in_interval].values
        else:
            return in_interval.sum()

    def count_shell(
        self, X_init, SMILES_sampled, dl, dh, nBits=2048, return_mols=False
    ):
        """
        Count the number of molecules in
        the shell of radius dl and dh.
        """
        darr = np.zeros(len(SMILES_sampled))
        for i, s in enumerate(SMILES_sampled):
            darr[i] = np.linalg.norm(
                X_init - self.compute_representations([s], nBits=nBits)
            )

        in_interval = (darr >= dl) & (darr <= dh)
        N = len(darr[in_interval])

        if return_mols == False:
            return N
        else:
            return N, SMILES_sampled[in_interval][:1000]

    def make_canon(self, SMILES):
        """
        Convert to canonical smiles form.
        """

        CANON_SMILES = []
        for smi in SMILES:

            can = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
            CANON_SMILES.append(can)

        return CANON_SMILES





def chemspacesampler_ECFP(smiles, params=None):
    """
    Run the chemspacesampler with ECFP fingerprints.
    """

    X, rdkit_init, egc = initialize_from_smiles(smiles)
    

    if params is None:
        num_heavy_atoms = rdkit_init.GetNumHeavyAtoms()
        elements = list({atom.GetSymbol() for atom in rdkit_init.GetAtoms()})
        params = {
            'min_d': 0.0,
            'max_d': 6.0,
            'NPAR': 1,
            'Nsteps': 100,
            'bias_strength': "none",
            'possible_elements': elements,
            'not_protonated': None, 
            'forbidden_bonds': [(8, 9), (8,8), (9,9), (7,7)],
            'nhatoms_range': [num_heavy_atoms, num_heavy_atoms],
            'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
            'make_restart_frequency': None,
            "verbose": False,
        }

    
    min_func = potential_ECFP(X ,gamma=params['min_d'], sigma=params['max_d'], nbits=2048,verbose=params["verbose"])

    respath = tempfile.mkdtemp()
    Parallel(n_jobs=params['NPAR'])(delayed(mc_run)(egc,min_func,"chemspacesampler", respath, f"results_{i}", params) for i in range(params['NPAR']) )
    ana = Analyze_Chemspace(respath+f"/*.pkl",rep_type="2d" , full_traj=False, verbose=False)
    _, GLOBAL_HISTOGRAM, _ = ana.parse_results()
    N, MOLS  = ana.count_shell_value(GLOBAL_HISTOGRAM, return_mols=True )
    shutil.rmtree(respath)

    return N, MOLS

def chemspacesampler_SOAP(smiles, params=None):
    """
    Runs the chemspacesampler with SOAP Ensemble representations.

    Parameters:
    smiles (str): SMILES string of the molecule.
    params (dict, optional): Parameters for SOAP representation. If None, default parameters will be used.

    Returns:
    tuple: Number of structures (N) and molecular SMILES (MOLS).
    """

    init_egc, tp,rdkit_init =  initialize_fml_from_smiles(smiles)



    if params is None:
        
        num_heavy_atoms = rdkit_init.GetNumHeavyAtoms()
        elements = list({atom.GetSymbol() for atom in rdkit_init.GetAtoms()})

        params = {
            'min_d': 0.0,
            'max_d': 150.0,
            'NPAR': 1,
            'Nsteps': 100,
            'bias_strength': "none",
            'possible_elements': elements,
            'not_protonated': None, 
            'forbidden_bonds': [(8, 9), (8,8), (9,9), (7,7)],
            'nhatoms_range': [num_heavy_atoms, num_heavy_atoms],
            'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
            'make_restart_frequency': None,
            "verbose": False,
        }
    
    X         = fml_rep(tp["coordinates"], tp["nuclear_charges"], tp["rdkit_Boltzmann"], params['possible_elements']+["H"])
    min_func  = potential_SOAP(X,tp["nuclear_charges"],gamma=params["min_d"], sigma=params["max_d"],possible_elements= params["possible_elements"]+["H"], verbose=params["verbose"] )
    respath   = tempfile.mkdtemp()
    Parallel(n_jobs=params["NPAR"])(delayed(mc_run)(init_egc,min_func,"chemspacesampler", respath, f"results_{i}", params) for i in range(params["NPAR"]) )
    ana = Analyze_Chemspace(respath+f"/*.pkl",rep_type="3d" , full_traj=False, verbose=False)
    _, GLOBAL_HISTOGRAM, _ = ana.parse_results()
    N, MOLS  = ana.count_shell_value(GLOBAL_HISTOGRAM, return_mols=True )
    shutil.rmtree(respath)

    return N, MOLS