import streamlit as st
from mosaics.minimized_functions import chemspace_potentials
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
import csv
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
random.seed(42)
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from default_params import make_params_dict
import pdb
sns.set_style("whitegrid")  # Set style to whitegrid for better readability
sns.set_context("notebook")  # Set context to "notebook"

st.set_page_config(
   page_title="Chemspace",
   page_icon=":shark:",
   layout="wide",
)

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
    string = string[1:-1]
    tuples = string.split('),')
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
st.sidebar.subheader('Input Parameters')



smiles = st.sidebar.text_input('Start molecule', value="CC(=O)OC1=CC=CC=C1C(=O)O", help='Enter the SMILES string of your starting molecule.')
selected_descriptor = st.sidebar.selectbox('Select Descriptor', descriptor_options, help='Choose the descriptor used to calculate the distance between molecules.')
min_d = st.sidebar.number_input('Minimal distance', value=5.0, help='Enter the minimal desired distance from the start molecule.')
max_d = st.sidebar.number_input('Maximal distance', value=12.0, help='Enter the maximal desired distance from the start molecule.')
Nsteps = st.sidebar.number_input('#MC iterations', value=20, help='Enter the number of Monte Carlo iterations to be performed.')
possible_elements = st.sidebar.text_input('possible_elements', value="C, O, N, F", help='Enter the elements that are allowed in the generated molecules.').split(', ')
nhatoms_range = st.sidebar.text_input('Number of heavy atoms (non-hydrogen)', value="13, 16", help='Enter the range of the number of heavy atoms that should be in the generated molecules.').split(', ')
synth_cut = st.sidebar.number_input('Synthesizability (1 easy to 10 impossible to make) ', value=2, help='Enter the synthesizability cut-off. A lower value means easier to synthesize.')
mmff_check = st.sidebar.checkbox('MMFF94 parameters exist? (another sanity check)', value=True, help='Check if the generated molecules should have MMFF94 parameters.')
ensemble   = st.sidebar.checkbox('Ensemble representation (affects only geometry-based representations, BoB & SOAP)', value=False, help='Check if the ensemble representation should be used. It affects only geometry-based representations (BoB & SOAP).')
user_input = st.sidebar.text_input("Enter forbidden bonds", default_value_bonds)
forbidden_bonds = str_to_tuple_list(user_input)


params = make_params_dict(selected_descriptor, min_d, max_d, Nsteps, possible_elements, forbidden_bonds, nhatoms_range, synth_cut,ensemble, mmff_check)
if selected_descriptor == 'RDKit':
    chemspace_function = chemspace_potentials.chemspacesampler_MolDescriptors
elif selected_descriptor == 'ECFP4':
    chemspace_function = chemspace_potentials.chemspacesampler_ECFP
elif selected_descriptor == 'BoB':
    chemspace_function = chemspace_potentials.chemspacesampler_BoB
elif selected_descriptor == 'SOAP':
    chemspace_function = chemspace_potentials.chemspacesampler_SOAP

else:
    st.error('Unknown Descriptor selected')

if st.button('Run ChemSpace Sampler'):
    try:


        MOLS, D = chemspace_function(smiles=smiles, params=params)

        ALL_RESULTS =  pd.DataFrame(MOLS, columns=['SMILES']) 
        ALL_RESULTS['Distance'] = D


        if len(ALL_RESULTS) > 4:
        # Calculate fingerprints for all molecules
            FP_array = chemspace_potentials.get_all_FP( [Chem.MolFromSmiles(smi) for smi in ALL_RESULTS['SMILES'].values ]  , nBits=2048)
                                            
            # Perform PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(FP_array)
            pca_start = pca.transform(chemspace_potentials.get_all_FP([Chem.MolFromSmiles(smiles)], nBits=2048))

            # Add PCA results to DataFrame
            ALL_RESULTS['PCA1'] = pca_result[:,0]
            ALL_RESULTS['PCA2'] = pca_result[:,1]

        csv = ALL_RESULTS.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}" download="chemspace_sampler_results.csv">Download Results CSV File</a>'

        # Add download link to Streamlit
        st.markdown(href, unsafe_allow_html=True)

        
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
        plt.figure(figsize=(10, 6))
        plt.hist(ALL_RESULTS['Distance'].values, bins=20, color='skyblue', edgecolor='black')
        plt.title('Histogram of Distances')
        plt.xlabel('D')
        plt.ylabel('#')
        st.pyplot(plt)
        st.write('Table of the 10 closest molecules, see above to download all results.')
        st.table(table_data)


        if len(ALL_RESULTS) > 4:
            # Use a diverging color palette, increase point transparency and change marker style
            st.write('PCA plot of all molecules (alwatys using ECFP4 fingerprints for speed)')
            plt.figure(figsize=(10, 8))
            other_mols = ALL_RESULTS[ALL_RESULTS['SMILES'] != smiles]
            scatter_plot = sns.scatterplot(data=other_mols, x='PCA1', y='PCA2', s=100, palette='coolwarm', hue='Distance', alpha=0.7, legend=False, marker='o')

            # Increase size of start molecule marker and its edge color for emphasis
            plt.scatter(pca_start[:,0], pca_start[:,1], color='red', edgecolor='black', marker='*', s=500, label='Start Molecule')
            # Create a custom legend for the start molecule


            plt.title('PCA of Molecular Fingerprints', fontsize=21, weight='bold', pad=20)
            plt.xlabel('PCA1', fontsize=18, labelpad=15)
            plt.ylabel('PCA2', fontsize=18, labelpad=15)

            # Create a custom legend for the start molecule
            legend_marker = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, markeredgecolor='black')
            plt.legend(handles=[legend_marker], labels=['Start Molecule'], loc='upper right')
            # Create colorbar
            norm = Normalize(other_mols['Distance'].min(), other_mols['Distance'].max())
            sm = ScalarMappable(norm=norm, cmap='coolwarm')
            plt.colorbar(sm)
            # Remove top and right spines
            sns.despine()

            # Show the plot in Streamlit
            st.pyplot(plt.gcf())


    except:
        st.error('An error occurred. Please check your input parameters and try again. \
                 Is the starting molecule consistent with the conditions i.e. number of heavy atoms, elements, etc.? \
                 sometimes things fail for no apparent reason, just try again.')
