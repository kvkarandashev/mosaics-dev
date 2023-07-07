
import numpy as np
import rdkit
from rdkit import Chem
import pandas as pd
import os
import collections
from mosaics.minimized_functions import chemspace_potentials
import rdkit.Chem.Crippen as Crippen
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb
import random
from mosaics.minimized_functions.representations import *
from mosaics.misc_procedures import str_atom_corr
import math
import pdb
random.seed(42)

def is_float_not_nan(variable):
    return isinstance(variable, float) and not math.isnan(variable)

def read_xyz(path):
    """
    Reads the xyz files in the directory on 'path'
    Input
    path: the path to the folder to be read

    Output
    atoms: list with the characters representing the atoms of a molecule
    coordinates: list with the cartesian coordinates of each atom
    smile: list with the SMILE representation of a molecule
    prop: list with the scalar properties
    """
    atoms = []
    coordinates = []

    with open(path, "r") as file:
        lines = file.readlines()
        n_atoms = int(lines[0])  # the number of atoms
        smile = lines[n_atoms + 3].split()[0]  # smiles string
        prop = lines[1].split()[2:]  # scalar properties
        mol_id = lines[1].split()[1]

        # to retrieve each atmos and its cartesian coordenates
        for atom in lines[2 : n_atoms + 2]:
            line = atom.split()
            # atomic charge
            atoms.append(line[0])
            # cartesian coordinates
            # Some properties have '*^' indicading exponentiation
            try:
                coordinates.append([float(line[1]), float(line[2]), float(line[3])])
            except:
                coordinates.append(
                    [
                        float(line[1].replace("*^", "e")),
                        float(line[2].replace("*^", "e")),
                        float(line[3].replace("*^", "e")),
                    ]
                )

    # atoms  = np.array([NUCLEAR_CHARGE[ele] for ele in atoms])
    return mol_id, atoms, coordinates, smile, prop


def canonize(mol):
    return Chem.MolToSmiles(
        Chem.MolFromSmiles(mol), isomericSmiles=True, canonical=True
    )

def atomization_en(EN, ATOMS, normalize=False):

    """
    Compute the atomization energy, if normalize is True,
    the output is normalized by the number of atoms. This allows
    predictions to be consistent when comparing molecules of different size
    with respect to their bond energies i.e. set to True if the number of atoms
    changes in during the optimization process

    #ATOMIZATION = EN - (COMP['H']*en_H + COMP['C']*en_C + COMP['N']*en_N +  COMP['O']*en_O +  COMP['F']*en_F)
    #N^tot = Number of H-atoms x 1 + Number of C-atoms x 4 + Number of N-atoms x 3 + Number of O-atoms x 2 + Number of F-atoms x1
    #you divide atomization energy by N^tot and you're good

    =========================================================================================================
    Ele-    ZPVE         U (0 K)      U (298.15 K)    H (298.15 K)    G (298.15 K)     CV
    ment   Hartree       Hartree        Hartree         Hartree         Hartree        Cal/(Mol Kelvin)
    =========================================================================================================
    H     0.000000     -0.500273      -0.498857       -0.497912       -0.510927       2.981
    C     0.000000    -37.846772     -37.845355      -37.844411      -37.861317       2.981
    N     0.000000    -54.583861     -54.582445      -54.581501      -54.598897       2.981
    O     0.000000    -75.064579     -75.063163      -75.062219      -75.079532       2.981
    F     0.000000    -99.718730     -99.717314      -99.716370      -99.733544       2.981
    =========================================================================================================
    """

    en_H = -0.500273
    en_C = -37.846772
    en_N = -54.583861
    en_O = -75.064579
    en_F = -99.718730
    COMP = collections.Counter(ATOMS)

    if normalize:
        Ntot = (
            COMP["C"] * 4
            + COMP["N"] * 3
            + COMP["O"] * 2
            + COMP["F"] * 1
            + COMP["H"] * 1
        )
        ATOMIZATION = EN - (
            COMP["H"] * en_H
            + COMP["C"] * en_C
            + COMP["N"] * en_N
            + COMP["O"] * en_O
            + COMP["F"] * en_F
        )
        return ATOMIZATION / Ntot

    else:
        ATOMIZATION = EN - (
            COMP["H"] * en_H
            + COMP["C"] * en_C
            + COMP["N"] * en_N
            + COMP["O"] * en_O
            + COMP["F"] * en_F
        )
        return ATOMIZATION

def process_qm9(directory, all=True):

    """
    Reads the xyz files in the directory on 'path' as well as the properties of
    the molecules in the same directory.
    """

    file = os.listdir(directory)[0]
    data = []
    smiles = []
    properties = []

    if all:
        nr_molecules = len(os.listdir(directory))
    else:
        nr_molecules = 10000

    for file in os.listdir(directory)[:nr_molecules]:
        path = os.path.join(directory, file)
        mol_id, atoms, coordinates, smile, prop = read_xyz(path)
        # A tuple with the atoms and its coordinates
        data.append((atoms, coordinates))
        smiles.append(smile)  # The SMILES representation

        ATOMIZATION = atomization_en(float(prop[10]), atoms, normalize=False)
        prop += [ATOMIZATION]
        prop += [mol_id]
        properties.append(prop)  # The molecules properties

    properties_names = [
        "A",
        "B",
        "C",
        "mu",
        "alfa",
        "homo",
        "lumo",
        "gap",
        "R_squared",
        "zpve",
        "U0",
        "U",
        "H",
        "G",
        "Cv",
        "atomization",
        "GDB17_ID",
    ]
    df = pd.DataFrame(properties, columns=properties_names)  # .astype('float32')
    df["smiles"] = smiles
    df.head()

    df["mol"] = df["smiles"].apply(lambda x: Chem.MolFromSmiles(x))
    df["mol"].isnull().sum()

    canon_smile = []
    for molecule in smiles:
        canon_smile.append(canonize(molecule))

    df["canon_smiles"] = canon_smile
    df["canon_smiles"][df["canon_smiles"].duplicated()]

    ind = df.index[df["canon_smiles"].duplicated()]
    df = df.drop(ind)
    df["mol"] = df["canon_smiles"].apply(lambda x: Chem.MolFromSmiles(x))
    df.to_csv("qm9.csv", index=False)
    return df


def average_distance(X, distance_function):
    """
    Calculate the average distance between all non-identical points in X.
    
    Parameters:
    X (ndarray): Input numpy array of shape (n_samples, n_features).
    distance_function: Distance function that takes two vectors and returns the distance.
    
    Returns:
    float: Average distance.
    """
    n_samples, n_features = X.shape
    total_distance = 0
    count = 0

    for i in range(n_samples):
        for j in range(i+1, n_samples):
            if not np.array_equal(X[i], X[j]):
                dist = distance_function(X[i], X[j])
                total_distance += dist
                count += 1

    if count > 0:
        average_dist = total_distance / count
        return average_dist
    else:
        return 0

def find_V_0_pot(lowest_beta, dbar):
    return (2/lowest_beta)* (1/dbar)**2






ha2kcalmol = 630

if __name__ == "__main__":

    COMPUTE_VALUE_PLOT = False
    process= False

    if process:
        qm9_df = process_qm9('/store/common/jan/qm9/')
    else:
        qm9_df = pd.read_csv('qm9.csv')
    print(qm9_df)
    #random suffle the data
    #qm9_df = qm9_df.sample(frac=1, random_state=42).reset_index(drop=True)
    #SMILES = qm9_df['canon_smiles'].values

    GAP = qm9_df['gap'].values * ha2kcalmol #qm9_df['h298_atom'].values #atomization energy in kcal/mol
    ATOMIZATION_ENERGY = qm9_df['h298_atom'].values #atomization energy in kcal/mol
    #random suffle the data
    qm9_df = qm9_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    SMILES = qm9_df['smiles'].values
    #add hydrogens because Crippen descriptors need them and also the representation vectors from rdkit in our convention
    SMILES_H = [Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(smi))) for smi in SMILES]
    X, y, SMILES_SET       =  [], [], []
    for smi, y1, y2 in tqdm(zip(SMILES_H , GAP, ATOMIZATION_ENERGY)):
        try:
            y3 = Crippen.MolLogP(Chem.MolFromSmiles(smi) , True)
            y4 = Crippen.MolMR(Chem.MolFromSmiles(smi), True)
            y5 = chemspace_potentials.compute_values(smi)[0]*ha2kcalmol
            #just get free energies with bmapqml!
            if is_float_not_nan(y5):
                x1 = chemspace_potentials.initialize_from_smiles(smi)[0][0]
                output = chemspace_potentials.initialize_fml_from_smiles(smi, ensemble=False)[1]
                R, Q = output["coordinates"], output["nuclear_charges"]
                CHG = [str_atom_corr(q) for q in Q]
                x2 = generate_bob(CHG, R)
                X.append([x1, x2])
                y.append([y1, y2, y3, y4, y5])
                SMILES_SET.append(smi)
        except Exception as e:
            #some molecules from qm9 are not valid and fail to be processed by rdkit
            print(e)

    X, y, SMILES_SET = np.array(X), np.array(y), np.array(SMILES_SET)

    np.savez_compressed('/data/jan/calculations/BOSS/qm9_processed.npz', X=X, y=y, SMILES=SMILES_SET)