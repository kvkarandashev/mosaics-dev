import random
import sys

import numpy as np

from mosaics.valence_treatment import str2ChemGraph


def shuffled_atom_ids(chemgraph):
    atom_ids = list(range(chemgraph.nhatoms()))
    random.shuffle(atom_ids)
    return atom_ids


def print_for(chemgraph_str, shuffle=False):
    print("Shuffle:", shuffle)
    chemgraph = str2ChemGraph(chemgraph_str, shuffle=shuffle)
    print("Canonical representation:", chemgraph)
    print("Equivalence class examples:")

    atom_ids = shuffled_atom_ids(chemgraph)
    for i in atom_ids:
        chemgraph.check_equivalence_class((i,))

    print(chemgraph.unrepeated_atom_list())

    atom_ids2 = shuffled_atom_ids(chemgraph)
    for i1 in atom_ids:
        for i2 in atom_ids2:
            if i2 <= i1:
                continue
            chemgraph.check_equivalence_class((i1, i2))

    print("Bond equivalence class examples:")
    for atom_pair in chemgraph.equiv_class_examples(2):
        print(atom_pair)
    print("Ended bond_equivalence class examples.")


def main():
    if len(sys.argv) > 1:
        shuffle = sys.argv[1] == "shuffle"
    else:
        shuffle = False
    random.seed(1)
    np.random.seed(1)
    considered_chemgraphs = [
        "6#1@1@2:6#1@3:6#1@4:6#1@5:6#1@5:6#1",
        "6#1@1@2:6#1@3:6#1@4:6#1@5:6#1@5:6@6:7#2",
        "6#3@1:6#2@2:6#2@3:6#3",
        "6#3@1:6#2@2:7#1@3:6#3",
    ]
    for chemgraph in considered_chemgraphs:
        print("Considered chemgraph:", chemgraph)
        print_for(chemgraph, shuffle=shuffle)


if __name__ == "__main__":
    main()
