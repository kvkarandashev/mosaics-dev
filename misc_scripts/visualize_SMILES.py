from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdAbbreviations
import sys, os


#TODO: move to rdkit_draw_utils and use there?
def draw_rdkit(mol, filename, kekulize=True, size=(300, 200), rotate=None, bw_palette=True, abbrevs=True, abbreviate_max_coverage=1.0, centreMoleculesBeforeDrawing=True):
    drawing=rdMolDraw2D.MolDraw2DCairo(*size)

    do = drawing.drawOptions()
    if bw_palette:
        do.useBWAtomPalette()
    if rotate is None:
        do.rotate=0
    else:
        do.rotate=rotate

    if abbrevs:
        used_abbrevs = rdAbbreviations.GetDefaultAbbreviations()
        full_mol = mol
        mol = rdAbbreviations.CondenseMolAbbreviations(
                full_mol, used_abbrevs, maxCoverage=abbreviate_max_coverage
            )
    if centreMoleculesBeforeDrawing:
        do.centreMoleculesBeforeDrawing = centreMoleculesBeforeDrawing

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawing,
        mol,
        kekulize=kekulize,
    )

    text=drawing.GetDrawingText()
    with open(filename, "wb") as f:
        f.write(text)


SMILES = sys.argv[1]

if len(sys.argv) > 2:
    filename_prefix = sys.argv[2]
else:
    filename_prefix = "chemgraph_visualization_"

rdkit_mol = Chem.MolFromSmiles(SMILES)

i = 0

while True:
    filename = filename_prefix + str(i) + ".png"
    if os.path.isfile(filename):
        i += 1
    else:
        draw_rdkit(
            rdkit_mol,
            filename,
        )
        break
