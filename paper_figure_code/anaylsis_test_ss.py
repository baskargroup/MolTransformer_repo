import sys
import os
import pandas as pd
from rdkit.Chem import MolFromSmiles, AddHs
from rdkit.Chem.rdMolDescriptors import CalcNumAtoms
import matplotlib.pyplot as plt
import selfies as sf # type: ignore
import numpy as np

# Add the path to the MolTransformer directory to sys.path
moltransformer_path = os.path.abspath('/work/mech-ai/bella/MolTransformer_repo/')
sys.path.append(moltransformer_path)


# Example usage:
# Assume you've already loaded and processed your CSV files into a DataFrame
csv_file_path = '/work/mech-ai/bella/ChemTransformer/data/SS/test/SS_test_1.csv'
save_path = '/work/mech-ai/bella/MolTransformer_repo/output/hpc/ss_test_analysis/test_1/'
original_dataframe = pd.read_csv(csv_file_path)
#original_dataframe = original_dataframe.sample(n=10000, random_state=42)
print("finish reading")

# Now you can import the MolTransformer module
import MolTransformer
from MolTransformer.generative import GenerateMethods
from MolTransformer.generative.generative_utils import *

def selfies_2_smiles(selfies_list):
    smiles = [sf.decoder(selfies) for selfies in selfies_list]
    return smiles

def count_selfies_symbols(selfies_string):
    return len(list(sf.split_selfies(selfies_string)))

def plot_atom_number_distribution(dataframe, save_file=''):
    if 'num_atoms' not in dataframe.columns:
        dataframe['num_atoms'] = dataframe['rdk_mol'].apply(CalcNumAtoms)
    
    ax = dataframe['num_atoms'].hist(bins=20)
    ax.set_xlabel('Number of Atoms')
    ax.set_ylabel('Frequency')
    plt.title('Distribution of Atom Counts')
    plt.savefig(save_file + 'num_atoms_distribution.png')
    plt.close()

    return dataframe['num_atoms'].value_counts(normalize=True).sort_index()

def plot_selfies_length_distribution(dataframe, save_file=''):
    dataframe['selfies_length'] = dataframe['SELFIES'].apply(count_selfies_symbols)
    
    ax = dataframe['selfies_length'].hist(bins=20)
    ax.set_xlabel('Length of SELFIES String (Symbols)')
    ax.set_ylabel('Frequency')
    plt.title('Distribution of SELFIES String Lengths')
    plt.savefig(save_file + 'selfies_length_distribution.png')
    plt.close()

    return dataframe['selfies_length'].value_counts(normalize=True).sort_index()

def ratio_selfies_length_above_400(dataframe):
    if 'selfies_length' not in dataframe.columns:
        dataframe['selfies_length'] = dataframe['SELFIES'].apply(count_selfies_symbols)
    
    total_molecules = len(dataframe)
    molecules_above_400 = len(dataframe[dataframe['selfies_length'] > 400])

    return molecules_above_400, total_molecules, molecules_above_400 / total_molecules

def min_atoms_with_selfies_above_400(dataframe):
    if 'selfies_length' not in dataframe.columns:
        dataframe['selfies_length'] = dataframe['SELFIES'].apply(count_selfies_symbols)
    if 'num_atoms' not in dataframe.columns:
        dataframe['num_atoms'] = dataframe['rdk_mol'].apply(CalcNumAtoms)
    
    filtered_df = dataframe[dataframe['selfies_length'] > 400]
    
    if not filtered_df.empty:
        return filtered_df['num_atoms'].min()
    else:
        return None  # Return None if no molecules have SELFIES strings longer than 400

def plot_atom_distribution_selfies_above_400(dataframe, save_file=''):
    if 'selfies_length' not in dataframe.columns:
        dataframe['selfies_length'] = dataframe['SELFIES'].apply(count_selfies_symbols)
    if 'num_atoms' not in dataframe.columns:
        dataframe['num_atoms'] = dataframe['rdk_mol'].apply(CalcNumAtoms)
    
    filtered_df = dataframe[dataframe['selfies_length'] > 400]
    
    ax = filtered_df['num_atoms'].hist(bins=20)
    ax.set_xlabel('Number of Atoms')
    ax.set_ylabel('Frequency')
    plt.title('Atom Count Distribution for SELFIES Length > 400 Symbols')
    plt.savefig(save_file + 'atom_distribution_selfies_above_400.png')
    plt.close()

    return filtered_df['num_atoms'].value_counts(normalize=True).sort_index()

def plot_atom_distribution_selfies_below_400(dataframe, save_file=''):
    if 'selfies_length' not in dataframe.columns:
        dataframe['selfies_length'] = dataframe['SELFIES'].apply(count_selfies_symbols)
    if 'num_atoms' not in dataframe.columns:
        dataframe['num_atoms'] = dataframe['rdk_mol'].apply(CalcNumAtoms)
    
    filtered_df = dataframe[dataframe['selfies_length'] < 400]
    
    ax = filtered_df['num_atoms'].hist(bins=20)
    ax.set_xlabel('Number of Atoms')
    ax.set_ylabel('Frequency')
    plt.title('Atom Count Distribution for SELFIES Length < 400 Symbols')
    plt.savefig(save_file + 'atom_distribution_selfies_below_400.png')
    plt.close()

    return filtered_df['num_atoms'].value_counts(normalize=True).sort_index()

def save_statistics_to_file(save_path, ratio_above_400, min_atoms, molecules_above_400, total_molecules):
    with open(save_path + 'statistics.txt', 'w') as f:
        f.write(f"Ratio of SELFIES length > 400: {ratio_above_400}\n")
        f.write(f"Minimum number of atoms for SELFIES length > 400: {min_atoms}\n")
        f.write(f"Molecules with SELFIES length > 400: {molecules_above_400}\n")
        f.write(f"Total molecules: {total_molecules}\n")

def plot_generative_molecules_analysis(dataframe, save_file=''):
    # Correcting the check for empty "rdk_mol" and generating RDKit molecule objects
    if 'rdk_mol' not in dataframe.columns or dataframe['rdk_mol'].isnull().all():
        dataframe['rdk_mol'] = dataframe['SMILES'].apply(MolFromSmiles)
    
    dataframe['num_rings'] = dataframe['rdk_mol'].apply(CalcNumRings)
    dataframe['num_atoms'] = dataframe['rdk_mol'].apply(CalcNumAtoms)
    dataframe['atom_types'] = dataframe['rdk_mol'].apply(lambda x: list(set([a.GetSymbol() for a in AddHs(x).GetAtoms()])))

    # Plotting distributions and returning normalized distributions
    num_atoms_distribution = dataframe['num_atoms'].value_counts(normalize=True).sort_index()
    num_rings_distribution = dataframe['num_rings'].value_counts(normalize=True).sort_index()

    ax = dataframe['num_atoms'].hist(bins=20)
    ax.set_xlabel('Number of Atoms')
    ax.set_ylabel('Frequency')
    plt.title('Distribution of Atom Counts')
    plt.savefig(save_file + 'num_atoms_distribution.png')  # Saving the plot
    plt.close()

    ax = dataframe['num_rings'].hist(bins=20)
    ax.set_xlabel('Number of Rings')
    ax.set_ylabel('Frequency')
    plt.title('Distribution of Ring Counts')
    plt.savefig(save_file + 'num_rings_distribution.png')  # Saving the plot
    plt.close()

    # Handling atom types
    atoms = list(set([a for ats in dataframe['atom_types'] for a in ats]))
    for a in atoms:
        dataframe[a] = dataframe['atom_types'].apply(lambda x: a in x)
    
    atom_types_df = dataframe[atoms].sum()
    atom_types_df.sort_values(ascending=False, inplace=True)
    atom_types_distribution = atom_types_df / atom_types_df.sum()  # Normalize the atom type counts
    ax = atom_types_df.plot.bar()
    ax.set_xlabel('Atom Type')
    ax.set_ylabel('Count')
    plt.title('Distribution of Atom Types')
    plt.savefig(save_file + 'atom_types_distribution.png')  # Saving the plot
    plt.close()

    # Return the normalized distributions
    return num_atoms_distribution, num_rings_distribution, atom_types_distribution

# Convert SELFIES to SMILES
if 'SELFIES' in original_dataframe.columns:
    original_dataframe['SMILES'] = selfies_2_smiles(original_dataframe['SELFIES'])

# Generate RDKit molecules from SMILES
original_dataframe['rdk_mol'] = original_dataframe['SMILES'].apply(MolFromSmiles)

# 1. Plot and get atom number distribution
atom_dist = plot_atom_number_distribution(original_dataframe, save_file=save_path)

# 2. Plot and get SELFIES length distribution
selfies_len_dist = plot_selfies_length_distribution(original_dataframe, save_file=save_path)

# 3. Calculate the ratio of molecules with SELFIES length > 400
molecules_above_400, total_molecules, ratio_above_400 = ratio_selfies_length_above_400(original_dataframe)

# 4. Find the minimum number of atoms for molecules with SELFIES length > 400
min_atoms = min_atoms_with_selfies_above_400(original_dataframe)

# 5. Plot the atom distribution for molecules with SELFIES length > 400
atom_dist_above_400 = plot_atom_distribution_selfies_above_400(original_dataframe, save_file=save_path)

# 6. Plot the atom distribution for molecules with SELFIES length < 400
atom_dist_below_400 = plot_atom_distribution_selfies_below_400(original_dataframe, save_file=save_path)

# Save the statistics to a text file
save_statistics_to_file(save_path, ratio_above_400, min_atoms, molecules_above_400, total_molecules)

# Output the results if needed
print("Ratio of SELFIES length > 400:", ratio_above_400)
print("Minimum number of atoms for SELFIES length > 400:", min_atoms)
print("Molecules with SELFIES length > 400:", molecules_above_400)
print("Total molecules:", total_molecules)


original_distributions = plot_generative_molecules_analysis(original_dataframe, save_file= save_path )

