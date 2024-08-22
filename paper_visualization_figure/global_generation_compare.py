import sys
import os

# Add the path to the MolTransformer directory to sys.path
moltransformer_path = os.path.abspath('/Users/tcpba/MolTransformer_repo/')
sys.path.append(moltransformer_path)
# Now you can import the MolTransformer module
import MolTransformer
# Your code that uses MolTransformer
from MolTransformer.generative import GenerateMethods
from MolTransformer.generative.generative_utils import *
import pandas as pd
from rdkit.Chem import MolFromSmiles, AddHs
from rdkit.Chem.rdMolDescriptors import CalcNumRings, CalcNumAtoms
import matplotlib.pyplot as plt
import selfies as sf # type: ignore
from scipy.stats import entropy

def selfies_2_smile(selfies_list):
    smiles = [sf.decoder(selfies) for selfies in selfies_list]
    return smiles

# Path to the CSV file
# user setting:

csv_file_path = '/Users/tcpba/2024Spring/ss_test_data/SS_test_1.csv'
save_path = '/Users/tcpba/MolTransformer_repo/output/global_generated/SS_test_1/'
print("save_path: " + save_path)
num_of_mol_from_dataset = 500
num_of_gen_mol = 500

# Read the CSV file into a pandas DataFrame
original_dataframe_ = pd.read_csv(csv_file_path)
original_dataframe = original_dataframe_.sample(n=num_of_mol_from_dataset, random_state=42)
print("len of testing", len(original_dataframe))
# Check if the CSV file contains the 'SELFIES' column
if 'SELFIES' in original_dataframe.columns:
    # Convert SELFIES to SMILES
    original_dataframe['SMILES'] = selfies_2_smile(original_dataframe['SELFIES'])

    # Print the updated DataFrame with SMILES column
    print(original_dataframe.head())

    # Optionally, save the updated DataFrame to a new CSV file
    # df.to_csv('/path/to/your/output_file.csv', index=False)
else:
    print("The CSV file does not contain a 'SELFIES' column.")



import pandas as pd
from rdkit.Chem import MolFromSmiles, AddHs
from rdkit.Chem.rdMolDescriptors import CalcNumRings, CalcNumAtoms
import matplotlib.pyplot as plt
import numpy as np
import os

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



def compare_and_plot_distributions(original_distribution, generated_distribution, save_file, plot_title='Comparison of Distributions', epsilon=1e-5):
    # Align both distributions for comparison
    combined_index = original_distribution.index.union(generated_distribution.index)
    original_distribution = original_distribution.reindex(combined_index, fill_value=0)
    generated_distribution = generated_distribution.reindex(combined_index, fill_value=0)
    
    # Apply stronger smoothing to avoid zero probabilities
    original_distribution = original_distribution + epsilon
    generated_distribution = generated_distribution + epsilon
    
    # Normalize distributions after smoothing
    original_distribution /= original_distribution.sum()
    generated_distribution /= generated_distribution.sum()

    # Plot the distributions
    plt.figure(figsize=(10, 6))
    plt.plot(original_distribution, label='Original', marker='o')
    plt.plot(generated_distribution, label='Generated', marker='x')
    plt.xlabel('Value')
    plt.ylabel('Normalized Frequency')
    plt.title(plot_title)
    
    # Compute KL divergence (original as the ground truth)
    kl_div = entropy(original_distribution, generated_distribution)
    plt.legend(title=f'KL Divergence: {kl_div:.4f}')
    
    # Save the plot
    plt.savefig(save_file)
    plt.close()

    return kl_div


# Run the function on the DataFrame
GM = GenerateMethods(save=False)  # Set `save=True` to save results and logs
smiles_list, selfies_list = GM.global_molecular_generation(n_samples=num_of_gen_mol)

# Create a DataFrame from the lists
data = {
    'SMILES': smiles_list,
    'SELFIES': selfies_list
}
generated_dataframe = pd.DataFrame(data)
# Run analysis on the original and generated dataframes
original_distributions = plot_generative_molecules_analysis(original_dataframe, save_file= save_path + 'ss_testing_set_plot/original_')
generated_distributions = plot_generative_molecules_analysis(generated_dataframe, save_file=  save_path +  'generated_plot/generated_')

# Compare and plot distributions
kl_atom_count = compare_and_plot_distributions(original_distributions[0], generated_distributions[0], save_file=  save_path + 'comparison/comparison_num_atoms.png', plot_title='Atom Count Distribution')
kl_ring_count = compare_and_plot_distributions(original_distributions[1], generated_distributions[1], save_file=  save_path + 'comparison/comparison_num_rings.png', plot_title='Ring Count Distribution')
kl_atom_types = compare_and_plot_distributions(original_distributions[2], generated_distributions[2], save_file=  save_path + 'comparison/comparison_atom_types.png', plot_title='Atom Types Distribution')


