from mosaics.valence_treatment import str2ChemGraph
import sys
from mosaics.crossover import FragmentPair, matching_status_reconnectors

cg1 = str2ChemGraph(sys.argv[1])
cg2 = str2ChemGraph(sys.argv[2])

if len(sys.argv) > 3:
    neighborhood_size1 = int(sys.argv[3])
    neighborhood_size2 = int(sys.argv[4])
else:
    neighborhood_size1 = 3
    neighborhood_size2 = 3

origin_point1 = 0
origin_point2 = 0


print(cg1.hatoms)
print(cg1.neighbors(origin_point1))

print(cg2.hatoms)
print(cg2.neighbors(origin_point2))

frag_pair1 = FragmentPair(cg1, origin_point1, neighborhood_size=neighborhood_size1)

frag_pair1.init_affected_status_info()

print(frag_pair1.membership_vector)

frag_pair2 = FragmentPair(cg2, origin_point2, neighborhood_size=neighborhood_size2)

frag_pair2.init_affected_status_info()

print(frag_pair2.membership_vector)

reconnectors = matching_status_reconnectors(frag_pair1, frag_pair2)

print("Reconnectors and statuses:")
for recon, status_ids in zip(*reconnectors):
    print(recon, status_ids)
