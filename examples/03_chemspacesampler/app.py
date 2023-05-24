import streamlit as st
from mosaics.minimized_functions import chemspace_potentials
from mosaics.beta_choice import gen_exp_beta_array
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG, display
import cairosvg
from rdkit.Chem import AllChem

def mol_to_img(mol):
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return cairosvg.svg2png(bytestring=svg.encode('utf-8'))


def mol_to_3d(mol):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    return mol


st.title('ChemSpace Sampler App')

# Parameters input
min_d = st.number_input('min_d', value=0.0)
max_d = st.number_input('max_d', value=2.0)
Nsteps = st.number_input('Nsteps', value=10)
possible_elements = st.text_input('possible_elements', value="C, O, N, F").split(', ')
nhatoms_range = st.text_input('nhatoms_range', value="6, 9").split(', ')
verbose = st.checkbox('Verbose', value=True)
smiles = st.text_input('SMILES', value="CCCCCC")

params = {
    'min_d': min_d,
    'max_d': max_d,
    'NPAR': 1,
    'Nsteps': Nsteps,
    'bias_strength': "none",
    'possible_elements': possible_elements,
    'not_protonated': None, 
    'forbidden_bonds': [(8, 9), (8, 8), (9, 9), (7, 7)],
    'nhatoms_range': [int(n) for n in nhatoms_range],
    'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
    'make_restart_frequency': None,
    'rep_type': 'MolDescriptors',
    "verbose": verbose
}

if st.button('Run ChemSpace Sampler'):
    MOLS, D = chemspace_potentials.chemspacesampler_MolDescriptors(smiles=smiles, params=params)
    
    # Assuming D contains distances and has the same length as MOLS
    D = D[:10]

    # Convert MOLS to dataframe
    mol_df = pd.DataFrame(MOLS[:10], columns=['SMILES'])  # creating DataFrame from MOLS
    mol_df['Distance'] = D

    mol_df['img'] = mol_df['SMILES'].apply(lambda x: mol_to_img(mol_to_3d(Chem.MolFromSmiles(x))))
    mol_df['img'] = mol_df['img'].apply(lambda x: base64.b64encode(x).decode())
    st.image([BytesIO(base64.b64decode(img_str)) for img_str in mol_df['img']])

    # Create a Streamlit table with SMILES strings and respective images.
    table_data = pd.DataFrame(columns=["SMILES", "Distance"])
    for idx, row in mol_df.iterrows():
        table_data = table_data.append(
            {"SMILES": row["SMILES"], "Distance": row["Distance"]}, ignore_index=True
        )


    st.table(table_data)