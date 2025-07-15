import numpy as np

from mosaics.xyz2graph import egc_from_ncharges_coords

coords = np.array(
    [
        [-1.4090010435957512, -0.3957869685518845, 0.5080594785889679],
        [-1.077128124961168, 0.9757340629507426, 0.6521411212909896],
        [0.2550250893819784, 1.2394962026302263, 0.21932615048515983],
        [1.1099714315071263, 0.06670378377579302, 0.23295401422717676],
        [2.4048070493861124, 0.37803262561694145, -0.1252097521752924],
        [0.5369927542871366, -0.9195236335208328, -0.6711893383066329],
        [-0.888656894022456, -0.9369477673981939, -0.6916761718251835],
        [-2.499712257028062, -0.47453945200195613, 0.4753906497613145],
        [-1.067540892839858, -0.9676332057053766, 1.3806974753640802],
        [0.21167003969332757, 1.6880267279994503, -0.7807958162604135],
        [0.6659447207921094, 2.0012938423140922, 0.8894715780167466],
        [0.8694833628274249, -0.7370654886136061, -1.699121318451711],
        [0.8881447645720959, -1.9177907294954168, -0.390048070715205],
    ]
)

nuclear_charges = np.array([6, 8, 6, 7, 9, 6, 8, 1, 1, 1, 1, 1, 1])

natoms = len(nuclear_charges)

atom_indices = np.array(list(range(natoms)))

np.random.shuffle(atom_indices)

print("old nuclear charges")
print(nuclear_charges)

print("atom indices")
print(atom_indices)

print("new nuclear_charges")
new_nuclear_charges = nuclear_charges[atom_indices]
new_coords = coords[atom_indices]
print(new_nuclear_charges)

labels = ["C0", "O", "C1", "N", "F", "C1", "O", "C0H", "C0H", "CH", "CH", "CH", "CH"]

print("old labels")
print(labels)

shuffled_labels = []
for i in atom_indices:
    shuffled_labels.append(labels[i])

print("new labels")
print(shuffled_labels)

egc = egc_from_ncharges_coords(nuclear_charges, coords)

egc_shuffled = egc_from_ncharges_coords(new_nuclear_charges, new_coords)

print("Label preservation comparison:")
for i, i_shuffled in zip(
    egc.get_inv_original_canonical_permutation(),
    egc_shuffled.get_inv_original_canonical_permutation(),
):
    lbl = labels[i]
    lbl_shuffled = shuffled_labels[i_shuffled]
    assert lbl == lbl_shuffled
    print(lbl, lbl_shuffled)

# Equivalence vector.
print("Original equivalence vector:")
equiv_vector = egc.get_original_equivalence_classes()
print(equiv_vector)
print("Shuffled equivalence vector:")
shuffled_equiv_vector = egc_shuffled.get_original_equivalence_classes()
print(shuffled_equiv_vector)
print("Checking equivalence preservation:")
for i1, i1_shuffled in zip(
    egc.get_inv_original_canonical_permutation(),
    egc_shuffled.get_inv_original_canonical_permutation(),
):
    for i2, i2_shuffled in zip(
        egc.get_inv_original_canonical_permutation(),
        egc_shuffled.get_inv_original_canonical_permutation(),
    ):
        is_eq = equiv_vector[i1] == equiv_vector[i2]
        is_eq_shuffled = shuffled_equiv_vector[i1_shuffled] == shuffled_equiv_vector[i2_shuffled]
        assert is_eq == is_eq_shuffled
