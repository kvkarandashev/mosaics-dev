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

import random

random.seed(42)


def mol_to_img(mol):
    mol = AllChem.RemoveHs(mol)
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return cairosvg.svg2png(bytestring=svg.encode('utf-8'))

def mol_to_3d(mol):
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    return mol

def str_to_tuple_list(string):
    # Remove surrounding brackets
    string = string[1:-1]
    # Split into tuples
    tuples = string.split('),')
    # Split tuples into numbers and convert to integers
    tuples = [tuple(map(int, t.replace('(', '').replace(')', '').split(','))) for t in tuples]
    return tuples

default_value_bonds = "[(8, 9), (8, 8), (9, 9), (7, 7)]"
descriptor_options = ['RDKit', 'ECFP4','BoB', 'SOAP']

st.title('ChemSpace Sampler App')
st.write('README: \
          This application generates new chemical structures starting from a given molecule. \
          Just enter the parameters below and click "Run ChemSpace Sampler"! \
          Ensemble representation will make distances less noisy. Without it you may get distances vastly outside of the defined target interval (min_d, max_d).\
          But it will take longer to run...')


# Parameters input
st.subheader('Input Parameters')
smiles = st.text_input('Start molecule', value="CC(=O)OC1=CC=CC=C1C(=O)O")
selected_descriptor = st.selectbox('Select Descriptor', descriptor_options)




min_d = st.number_input('Minimal distance', value=5.0)
max_d = st.number_input('Maximal distance', value=12.0)
Nsteps = st.number_input('#MC iterations', value=20)
possible_elements = st.text_input('possible_elements', value="C, O, N, F").split(', ')
nhatoms_range = st.text_input('Number heavy atoms (non-hydrogen)', value="13, 16").split(', ')
synth_cut = st.number_input('Synthesizability (1 easy to 10 impossible to make) ', value=2)
mmff_check = st.checkbox('MMFF94 parameters exist? (another sanity check)', value=True)
ensemble   = st.checkbox('Ensemble representation (affects only geometry-based representations, BoB & SOAP)', value=False)
user_input = st.text_input("Enter forbidden bonds", default_value_bonds)



forbidden_bonds = str_to_tuple_list(user_input)




# Determine the chemspacesampler function to use based on selected_descriptor
if selected_descriptor == 'RDKit':
    chemspace_function = chemspace_potentials.chemspacesampler_MolDescriptors
    params = {
    'min_d': min_d,
    'max_d': max_d,
    'NPAR': 1,
    'Nsteps': Nsteps,
    'bias_strength': "none",
    'possible_elements': possible_elements,
    'not_protonated': None, 
    'forbidden_bonds': forbidden_bonds,
    'nhatoms_range': [int(n) for n in nhatoms_range],
    'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
    'make_restart_frequency': None,
    'rep_type': 'MolDescriptors',
    'synth_cut': synth_cut,
    'mmff_check': mmff_check,
    "verbose": True
    }

elif selected_descriptor == 'ECFP4':
    chemspace_function = chemspace_potentials.chemspacesampler_ECFP
    params = {
    'min_d': min_d,
    'max_d': max_d,
    'NPAR': 1,
    'Nsteps': Nsteps,
    'bias_strength': "none",
    'possible_elements': possible_elements,
    'not_protonated': None, 
    'forbidden_bonds': forbidden_bonds,
    'nhatoms_range': [int(n) for n in nhatoms_range],
    'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
    'make_restart_frequency': None,
    "rep_type": "2d",
    "nBits": 2048,
    "verbose": False
    }

elif selected_descriptor == 'BoB':
    chemspace_function = chemspace_potentials.chemspacesampler_BoB
    params = {
        'min_d': min_d,
        'max_d': max_d,
        'NPAR':1,
        'Nsteps': Nsteps,
        'bias_strength': "none",
        'possible_elements': possible_elements,
        'not_protonated': None, 
        'forbidden_bonds': forbidden_bonds,
        'nhatoms_range': [int(n) for n in nhatoms_range],
        'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
        'make_restart_frequency': None,
        'rep_type': '3d',
        'rep_name': 'BoB',
        'synth_cut': synth_cut,
        'ensemble': ensemble,
        "verbose": True}

elif selected_descriptor == 'SOAP':
    chemspace_function = chemspace_potentials.chemspacesampler_SOAP
    params = {
        'min_d': min_d,
        'max_d': max_d,
        'NPAR': 1,
        'Nsteps': Nsteps,
        'bias_strength': "none",
        'possible_elements': possible_elements,
        'not_protonated': None, 
        'forbidden_bonds': forbidden_bonds,
        'nhatoms_range': [int(n) for n in nhatoms_range],
        'betas': gen_exp_beta_array(4, 1.0, 32, max_real_beta=8.0),
        'make_restart_frequency': None,
        'rep_type': '3d',
        'synth_cut': 5,
        'rep_name': 'SOAP',
        'ensemble': ensemble,
        "verbose": True,
    }
else:
    st.error('Unknown Descriptor selected')

if st.button('Run ChemSpace Sampler'):


    MOLS, D = chemspace_function(smiles=smiles, params=params)
    
    # Assuming D contains distances and has the same length as MOLS
    D = D[:10]
    print(MOLS)
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