from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdAbbreviations
import sys, os

# Tried using "*" : "R", didn't work.
# Ended up with python visualize_SMILES.py '[Li+].C(#N)C1=C(N=C([N-]1)[*])C#N |$;;;;;;;;R;;$|'

default_abbreviations = {
    "C(F)(F)C(F)(F)C(F)(F)F": "CF2CF2CF3",
    "C(F)(F)C(F)(F)F": "CF2CF3",
    "C(F)(F)F": "CF3",
    "C#N": "CN",
}


def abbrev_defns(abbr_dict):
    return "\n".join(
        sorted(
            [displ + " " + SMILES for SMILES, displ in default_abbreviations.items()],
            reverse=True,
        )
    )


# TODO: move to rdkit_draw_utils and use there?
def draw_rdkit(
    mol,
    filename,
    kekulize=True,
    size=(300, 200),
    rotate=None,
    bw_palette=True,
    abbrevs=True,
    custom_abbreviations=default_abbreviations,
    abbreviate_max_coverage=1.0,
    centreMoleculesBeforeDrawing=False,
):
    drawing = rdMolDraw2D.MolDraw2DCairo(*size)

    do = drawing.drawOptions()
    if bw_palette:
        do.useBWAtomPalette()
    if rotate is None:
        do.rotate = 0
    else:
        do.rotate = rotate

    if abbrevs:
        if custom_abbreviations is None:
            used_abbrevs = rdAbbreviations.GetDefaultAbbreviations()
        else:
            abbrevs_txt = abbrev_defns(custom_abbreviations)
            used_abbrevs = rdAbbreviations.ParseAbbreviations(abbrevs_txt)
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

    text = drawing.GetDrawingText()
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
