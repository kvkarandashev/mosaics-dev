import sys
from mosaics.valence_treatment import str2ChemGraph
from mosaics.crossover import FragmentPair

cg = str2ChemGraph(sys.argv[1])

if len(sys.argv) > 2:
    frag_size = int(sys.argv[2])
else:
    frag_size = 3

if len(sys.argv) > 3:
    origin_point = int(sys.argv[3])
else:
    origin_point = 0

frag_pair = FragmentPair(cg, origin_point, neighborhood_size=frag_size)

frag_pair.init_affected_status_info()

print("Considered ChemGraph:", cg)

print("Affected bond statuses:")

print(frag_pair.affected_status)
