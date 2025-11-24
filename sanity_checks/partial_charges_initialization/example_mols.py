from mosaics.chem_graph import str2ChemGraph

cg_strings = ["6#3@1:7#2", "6#3@1:7#3_1", "6#2@1:7#2_1", "6#3@1:7@2@3:8:8"]

for cg_str in cg_strings:
    print("string:", cg_str)
    cg = str2ChemGraph(cg_str)
    for hatom_id, hatom in enumerate(cg.hatoms):
        print(hatom_id, hatom, hatom.charge)
