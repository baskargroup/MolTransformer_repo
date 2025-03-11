import sys
import os
import pandas as pd
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdMolDescriptors import CalcNumAtoms
import matplotlib.pyplot as plt
import selfies as sf # type: ignore
import numpy as np

# Add the path to the MolTransformer directory to sys.path
moltransformer_path = os.path.abspath('/work/mech-ai/bella/MolTransformer_repo/')
sys.path.append(moltransformer_path)

def selfies_2_smiles(selfies_list):
    smiles = [sf.decoder(selfies) for selfies in selfies_list]
    return smiles

def count_selfies_symbols(selfies_string):
    return len(list(sf.split_selfies(selfies_string)))

def min_atoms_with_selfies_above_400(dataframe):
    if 'selfies_length' not in dataframe.columns:
        dataframe['selfies_length'] = dataframe['SELFIES'].apply(count_selfies_symbols)
    if 'num_atoms' not in dataframe.columns:
        dataframe['num_atoms'] = dataframe['rdk_mol'].apply(CalcNumAtoms)
    
    filtered_df = dataframe[dataframe['selfies_length'] > 400]
    
    if not filtered_df.empty:
        return filtered_df['num_atoms'].min(), len(filtered_df), filtered_df
    else:
        return None, 0, pd.DataFrame()  # Return None, 0, and an empty DataFrame if no molecules have SELFIES strings longer than 400

def plot_atom_distribution_selfies_above_400(dataframe, save_file='', global_min_atoms=None):
    ax = dataframe['num_atoms'].hist(bins=20)
    ax.set_xlabel('Number of Atoms')
    ax.set_ylabel('Frequency')
    title = 'Atom Count Distribution for SELFIES Length > 400 Symbols'
    if global_min_atoms is not None:
        title += f"\nGlobal Minimum Atom Count: {global_min_atoms}"
    plt.title(title)
    plt.savefig(save_file + 'atom_distribution_selfies_above_400.png')
    plt.close()

def save_statistics_to_file(save_path, filename, min_atoms, molecules_above_400, total_molecules, global_total_molecules,global_molecular_above_400,global_min_atoms=None):
    with open(os.path.join(save_path, filename), 'a') as f:
        f.write(f"{filename.replace('.txt', '')}:\n")
        f.write(f"Minimum number of atoms for SELFIES length > 400: {min_atoms}\n")
        f.write(f"Molecules with SELFIES length > 400: {molecules_above_400}\n")
        f.write(f"Total molecules: {total_molecules}\n")
        f.write(f"Global total number of molecules:   {global_total_molecules}\n")
        f.write(f"Global total moleculars above 400:   {global_molecular_above_400}\n")
        if global_min_atoms is not None:
            f.write(f"Global minimum number of atoms for SELFIES length > 400: {global_min_atoms}\n")
        f.write("\n")

# Main processing loop
save_path = '/work/mech-ai/bella/MolTransformer_repo/output/hpc/min_num_atom/test_all/'
os.makedirs(save_path, exist_ok=True)

output_file = "statistics.txt"
all_filtered_dfs = []  # List to collect all filtered DataFrames
global_min_atoms = None  # To track the global minimum atom count
global_total_molecules = 0
global_molecular_above_400 = 0
for i in range(20):
    # Load the CSV file
    csv_file_path = f'/work/mech-ai/bella/ChemTransformer/data/SS/test/SS_test_{i}.csv'
    dataframe = pd.read_csv(csv_file_path)
    #dataframe = dataframe.sample(n=1000, random_state=42)
    print("finish reading")
    
    # Convert SELFIES to SMILES
    if 'SELFIES' in dataframe.columns:
        dataframe['SMILES'] = selfies_2_smiles(dataframe['SELFIES'])

    # Generate RDKit molecules from SMILES
    dataframe['rdk_mol'] = dataframe['SMILES'].apply(MolFromSmiles)

    # Calculate minimum number of atoms and the number of molecules with SELFIES length > 400
    min_atoms, molecules_above_400, filtered_df = min_atoms_with_selfies_above_400(dataframe)
    total_molecules = len(dataframe)
    global_total_molecules += total_molecules
    global_molecular_above_400 += molecules_above_400

    # Update global minimum atom count
    if min_atoms is not None:
        if global_min_atoms is None or min_atoms < global_min_atoms:
            global_min_atoms = min_atoms

    # Save the statistics to a text file
    save_statistics_to_file(save_path, output_file, min_atoms, molecules_above_400, total_molecules,global_total_molecules,global_molecular_above_400,global_min_atoms)

    # Plot and save the atom distribution for molecules with SELFIES length > 400
    if not filtered_df.empty:
        plot_atom_distribution_selfies_above_400(filtered_df, save_file=save_path + f'SS_test_{i}_')

    # Collect filtered DataFrame for later combined plotting
    all_filtered_dfs.append(filtered_df)

    # Free the memory by deleting the dataframe
    del dataframe, filtered_df, min_atoms, molecules_above_400, total_molecules

    # Explicitly run garbage collection to free memory
    import gc
    gc.collect()

# Combine all filtered DataFrames
combined_filtered_df = pd.concat(all_filtered_dfs, ignore_index=True)

# Plot atom distribution for all molecules with SELFIES length > 400 across all files
if not combined_filtered_df.empty:
    plot_atom_distribution_selfies_above_400(combined_filtered_df, save_file=save_path + 'combined_', global_min_atoms=global_min_atoms)

print(f"Results saved to {save_path}")
