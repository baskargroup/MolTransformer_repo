from ..model.utils import *
import os
import matplotlib.pyplot as plt
import certifi  # type: ignore
from rdkit import Chem # type: ignore
from rdkit.Chem import MolFromSmiles, MolToSmiles, AddHs # type: ignore
from rdkit.Chem import inchi # type: ignore
from rdkit.Chem import rdMolDescriptors # type: ignore
from rdkit.Chem.rdMolDescriptors import CalcNumRings, CalcNumAtoms # type: ignore
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity # type: ignore
import pubchempy as pcp # type: ignore
from rdkit.Chem import Draw, AllChem # type: ignore
import pandas as pd
import numpy as np
from itertools import combinations
from PIL import ImageDraw, ImageFont
import math
import pickle
import gzip



# Set SSL_CERT_FILE to use certifi's CA bundle
os.environ['SSL_CERT_FILE'] = certifi.where()

def plot_generative_molecules_analysis(dataframe,save_file = ''):
    # Correcting the check for empty "rdk_mol" and generating RDKit molecule objects
    if 'rdk_mol' not in dataframe.columns or dataframe['rdk_mol'].isnull().all():
        dataframe['rdk_mol'] = dataframe['SMILES'].apply(MolFromSmiles)
    
    dataframe['num_rings'] = dataframe['rdk_mol'].apply(CalcNumRings)
    dataframe['num_atoms'] = dataframe['rdk_mol'].apply(CalcNumAtoms)
    dataframe['atom_types'] = dataframe['rdk_mol'].apply(lambda x: list(set([a.GetSymbol() for a in AddHs(x).GetAtoms()])))

    # Plotting and adding axis titles
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
    ax = atom_types_df.plot.bar()
    ax.set_xlabel('Atom Type')
    ax.set_ylabel('Count')
    plt.title('Distribution of Atom Types')
    plt.savefig(save_file + 'atom_types_distribution.png')  # Saving the plot
    plt.close()

def check_pubchem(smiles):
    """
    Check if a molecule represented by its SMILES string is available in PubChem.

    Parameters:
    - smiles (str): The SMILES string representing the molecule to be checked.

    Returns:
    - bool: True if the molecule is found in PubChem, False otherwise.
    """
    if smiles is None or smiles.strip() == "":
        return True  # Or handle as needed for your application

    clean_smile = MolToSmiles(MolFromSmiles(smiles))
    
    try:
        # Attempt to retrieve information about the molecule from PubChem
        pcpmol = pcp.get_compounds(clean_smile, namespace="smiles")[0]
        return True
    except pcp.BadRequestError:
        # Return False if the molecule is not found in PubChem
        return False

def validate_smiles_in_pubchem(input_data):
    """
    For a given DataFrame or path to a CSV file, check each SMILES in the 'SMILES' column to see if it is
    available in PubChem. Add the results as a new boolean column 'new_molecule' to the DataFrame.

    Parameters:
    - input_data (pd.DataFrame or str): Input DataFrame with a 'SMILES' column or path to a CSV file.

    Returns:
    - pd.DataFrame: The original DataFrame with an added 'new_molecule' column.
    """
    # Check if input_data is a path to a CSV file (str), and read it into a DataFrame
    if isinstance(input_data, str):
        df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data
    else:
        raise ValueError("Input must be a pandas DataFrame or a path to a CSV file")
    if 'SMILES' not in df.columns:
        raise ValueError("DataFrame must contain a 'SMILES' column")
    # Check if each SMILES string is in PubChem and store the result in a new column
    df.loc[:, 'pubchem'] = df['SMILES'].apply(check_pubchem)
    return df

def draw_all_structures(smiles_list, out_dir, mols_per_image=30, molsPerRow=5, name_tag='', file_prefix='molecules',pop_first = True,molecule_prefix = 'k = '):
    """
    Generate grid visualizations for molecular structures based on their SMILES representation with RDKit,
    plotting the first molecule separately with a specific naming convention.
    
    Parameters:
    - smiles_list (list): List of SMILES strings of the molecules.
    - out_dir (str): Directory where the visualizations will be saved.
    - mols_per_image (int, optional): Number of molecules per visualization image. Default is 30.
    - molsPerRow (int, optional): Number of molecules per row in the visualization grid. Default is 5.
    - name_tag (str, optional): Additional tag added to the output visualization file names. Default is an empty string.
    - file_prefix (str, optional): Prefix for the output file names. Default is 'molecules'.

    Returns:
    - None: The function saves visualization images of molecular structures in the specified output directory.
    """

    # Convert SMILES strings to RDKit molecule objects
    mols = []

    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:  # Check if conversion was successful
                mols.append(mol)
        except:
            print(f"Unable to convert SMILES to RDKit molecule: {smiles}")
            continue

    # Process the first molecule separately if the list is not empty
    if pop_first and mols:
        first_mol = mols.pop(0)  # Remove the first molecule for separate processing
        first_mol_img = Draw.MolsToGridImage([first_mol], molsPerRow=1, subImgSize=(300, 300), returnPNG=False)
        first_out_path = os.path.join(out_dir, f'{file_prefix}_initial{name_tag}.png')
        os.makedirs(out_dir, exist_ok=True)
        first_mol_img.save(first_out_path)
        print(f"Saved initial molecule visualization to {first_out_path}")

    # Divide the remaining list of molecules into chunks for visualization
    divided_list = [mols[i:i + mols_per_image] for i in range(0, len(mols), mols_per_image)]
    name_num = 1
    for mols_batch in divided_list:
        img = Draw.MolsToGridImage(mols_batch, molsPerRow=molsPerRow, legends=[ molecule_prefix + str(i) for i in range(1, len(mols_batch) + 1)], subImgSize=(300, 300),returnPNG=False)
        out_path = os.path.join(out_dir, f'{file_prefix}{name_num}{name_tag}.png')
        img.save(out_path)
        print(f"Saved visualization to {out_path}")
        name_num += 1



def plot_tanimoto_histogram(df, out_dir):
    """
    Plots Tanimoto Similarity vs. Distance Ratio from a DataFrame as histograms,
    comparing 'similarity_start' and 'similarity_end' to 'distance_ratio'.

    Parameters:
    - df: A pandas DataFrame that must contain 'TanimotoSimilarity', 'similarity_start',
          'similarity_end', and 'distance_ratio' columns.
    - out_dir: Directory path where the plot image will be saved.

    Returns:
    - A histogram showing both 'similarity_start' and 'similarity_end' against 'distance_ratio',
      each with unique colors and legends.
    """
    # Assuming 'step' is a column representing the order of molecules, if not you can use df.reset_index()
    df['step'] = df.index + 1  # if 'step' is not already a column

    # Extract 'similarity_start' and 'similarity_end'
    similarity_start = df['similarity_start']
    similarity_end = df['similarity_end']

    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Calculate bar width and offset
    bar_width = 0.3  # width of each bar
    offset = 0.2     # distance between the groups of bars

    # Plot 'similarity_start' histogram
    plt.bar(df['step'] - offset , similarity_start, width=bar_width, label='Generated vs. Start Molecule Similarity', color='blue', alpha=0.6)
    
    # Plot 'similarity_end' histogram
    plt.bar(df['step'] + offset, similarity_end, width=bar_width, label='Generated vs. End Molecule Similarity', color='green', alpha=0.6)
    
    # Setting the title and labels
    plt.title('Histogram of Tanimoto Similarity vs. Step')
    plt.xlabel('Step')
    plt.ylabel('Tanimoto Similarity')

    # Adding grid for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Adding a legend to distinguish between the histograms
    plt.legend()

    # Adjust x-axis ticks to center labels under the groups of bars
    plt.xticks(ticks=df['step'], labels=df['step'])

    # Save the plot to the specified directory
    plt.savefig(out_dir + 'Histogram_Tanimoto_Similarity_vs_Step.png')
    
    # Close the plot to free up memory
    plt.close()

def plot_tanimoto(df, out_dir):
    """
    Plots Tanimoto Similarity vs. Distance Ratio from a DataFrame, comparing 'similarity_start'
    and 'similarity_end' to 'distance_ratio'.

    Parameters:
    - df: A pandas DataFrame that must contain 'TanimotoSimilarity', 'similarity_start',
          'similarity_end', and 'distance_ratio' columns.
    - out_dir: Directory path where the plot image will be saved.

    Returns:
    - A plot showing both 'similarity_start' and 'similarity_end' against 'distance_ratio',
      each with unique colors and legends.
    """
    # Normalize 'distance_ratio' by dividing by the last value
    last_value = df['distance_ratio'].iloc[-1]
    normalized_distance_ratio = df['distance_ratio'] / last_value

    # Extract 'similarity_start' and 'similarity_end'
    similarity_start = df['similarity_start']
    similarity_end = df['similarity_end']

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot 'similarity_start' vs 'distance_ratio'
    plt.plot(normalized_distance_ratio, similarity_start, '-o', label='Generated vs. Start Molecule Similarity', color='blue', alpha=0.7, markerfacecolor='blue',markersize=10)
    
    # Plot 'similarity_end' vs 'distance_ratio'
    plt.plot(normalized_distance_ratio, similarity_end, '-o', label='Generated vs. End Molecule Similarity', color='green', alpha=0.7, markerfacecolor='green',markersize=10)
    
    # For 'similarity_start'
    plt.plot(normalized_distance_ratio.iloc[[0, -1]], similarity_start.iloc[[0, -1]], 'o', color='red', markersize=8)
    # For 'similarity_end'
    plt.plot(normalized_distance_ratio.iloc[[0, -1]], similarity_end.iloc[[0, -1]], 'o', color='red', markersize=8)

    # Setting the title and labels
    plt.title('Tanimoto Similarity vs. Normalized Distance Ratio')
    plt.xlabel('Normalized Distance Ratio')
    plt.ylabel('Tanimoto Similarity')
    
    # Adding grid for better readability
    plt.grid(True)
    
    # Adding a legend to distinguish between the lines
    plt.legend()
    
    # Save the plot to the specified directory
    plt.savefig(out_dir + 'Tanimoto_Similarity_vs_Distance_Ratio.png')
    
    # Close the plot to free up memory
    plt.close()

    

def calculate_tanimoto_similarity(dataframe):
    """
    Calculate the Tanimoto similarity of all molecules in the dataframe to the first molecule based on SMILES strings and
    add it as a new column to the dataframe.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame with a 'SMILES' column containing SMILES strings of molecules.

    Returns:
    - pd.DataFrame: Updated DataFrame with a new 'TanimotoSimilarity' column.
    """
    fingerprints = []
    for smiles in dataframe['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol:  # Ensure molecule is valid
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=1024)
            fingerprints.append(fp)
        else:
            fingerprints.append(None)  # Handle invalid SMILES strings

    first_fp = fingerprints[0]  # First molecule's fingerprint
    tanimoto_similarities = [TanimotoSimilarity(first_fp, fp) if fp is not None else 0 for fp in fingerprints]

    dataframe['TanimotoSimilarity'] = tanimoto_similarities
    return dataframe

def calculate_tanimoto_similarity_from_target(dataframe, smiles_target):
    """
    Calculate the Tanimoto similarity of all molecules in the dataframe to a target molecule based on SMILES strings.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame with a 'SMILES' column containing SMILES strings of molecules.
    - smiles_target (str): SMILES string of the target molecule.

    Returns:
    - List of similarity scores (floats) with None for any invalid SMILES strings in the input DataFrame.
    """
    # Convert the target SMILES string to a molecule and then to a fingerprint
    mol_target = Chem.MolFromSmiles(smiles_target)
    if not mol_target:
        raise ValueError("Invalid target SMILES string")
    fp_target = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_target, radius=4, nBits=1024)
    
    tanimoto_similarities = []
    for smiles in dataframe['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol:  # Check if the molecule creation was successful
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=1024)
            similarity = TanimotoSimilarity(fp_target, fp)
            tanimoto_similarities.append(similarity)
        else:
            # Append None for invalid SMILES to maintain the list length
            tanimoto_similarities.append(None)
    
    return tanimoto_similarities


def Tanimoto_diversity(smiles_list):
    # Convert SMILES to RDKit Molecules
    molecules = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    
    # Filter out any None values if SMILES conversion fails
    molecules = [mol for mol in molecules if mol is not None]
    
    # Generate fingerprints for all molecules
    fingerprints = [rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=1024) for mol in molecules]
    
    total_similarity = 0
    num_pairs = 0
    
    # Compute similarity for each unique pair
    for (fp1, fp2) in combinations(fingerprints, 2):
        similarity = TanimotoSimilarity(fp1, fp2)
        total_similarity += similarity
        num_pairs += 1

    # Avoid division by zero
    if num_pairs == 0:
        return None
    
    return 1 - (total_similarity / num_pairs)

def single_pair_TanimotoSimilarity(molecule1, molecule2):
    """
    Calculate the Tanimoto similarity between two molecules represented by their SMILES strings.

    Parameters:
    - molecule1 (str): SMILES string of the starting molecule.
    - molecule2 (str): SMILES string of the ending molecule.

    Returns:
    - float: Tanimoto similarity between the two molecules. Returns None if either SMILES string is invalid.
    """

    try:
        mol1 = Chem.MolFromSmiles(molecule1)
    except:
        print(f"Unable to convert molecule1 to RDKit molecule: {molecule1}")
        return 0
    try:
        mol2 = Chem.MolFromSmiles(molecule2)
    except:
        print(f"Unable to convert molecule2 to RDKit molecule: {molecule2}")
        return 0

    if not mol1 or not mol2:
        print("Invalid molecule SMILES provided.")
        return None  # One or both SMILES strings are invalid

    mol1_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, radius=4, nBits=1024)
    mol2_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, radius=4, nBits=1024)

    similarity = TanimotoSimilarity(mol1_fp, mol2_fp)
    return similarity	

def sort_tanimoto_similarity(generated_results, k):
    """
    Calculate Tanimoto similarity using calculate_tanimoto_similarity, sort by similarity, and select top k entries.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame with a 'SMILES' column.
    - k (int): Number of top similar entries to select.

    Returns:
    - pd.DataFrame: DataFrame of the top k entries sorted by Tanimoto similarity.
    """
    # Calculate Tanimoto similarity
    if isinstance(generated_results, dict):
        df = pd.DataFrame.from_dict(generated_results)
    elif not isinstance(generated_results, pd.DataFrame):
        raise TypeError("generated_results must be a dict or a pandas DataFrame")
    else:
        df = generated_results

    updated_df = calculate_tanimoto_similarity(df)
    
    # Sort by 'TanimotoSimilarity' in descending order
    sorted_df = updated_df.sort_values(by='TanimotoSimilarity', ascending=False).reset_index(drop=True)
    
    # Filter duplicates if needed. Assuming 'filter_duplicates' is defined elsewhere.
    # Uncomment and adjust the next line according to your duplicate filtering function.
    filtered_df = filter_duplicates(sorted_df)

    # Select top k entries. Since the first entry is included, we select top k+1 to get k comparisons.
    top_k_1 = filtered_df[:min(k+1, len(filtered_df))]
    
    return top_k_1

_fscores = None

def readFragmentScores(name='fpscores'):
    global _fscores
    try:
        with gzip.open(f'{os.path.join(os.path.dirname(__file__), name)}.pkl.gz', 'rb') as file:
            data = pickle.load(file)
            _fscores = {item[1]: float(item[0]) for item in data for i in range(1, len(item))}
    except Exception as e:
        print(f"Failed to load fragment scores: {e}")
        _fscores = {}


def numBridgeheadsAndSpiro(mol):
    return rdMolDescriptors.CalcNumBridgeheadAtoms(mol), rdMolDescriptors.CalcNumSpiroAtoms(mol)

def calculateScore(mol):
    if _fscores is None:
        readFragmentScores()

    # Get the Morgan fingerprint
    fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)  # radius=2
    fps = fp.GetNonzeroElements()
    score1 = sum(_fscores.get(bitId, -4) * count for bitId, count in fps.items()) / sum(fps.values())

    # Feature-based scores
    nAtoms = mol.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(mol)
    nMacrocycles = sum(1 for x in mol.GetRingInfo().AtomRings() if len(x) > 8)

    penalties = {
        'size': nAtoms**1.005 - nAtoms,
        'stereo': math.log10(nChiralCenters + 1),
        'spiro': math.log10(nSpiro + 1),
        'bridge': math.log10(nBridgeheads + 1),
        'macrocycle': math.log10(2) if nMacrocycles > 0 else 0
    }
    score2 = -(sum(penalties.values()))

    # Density correction
    score3 = 0.5 * math.log(float(nAtoms) / len(fps)) if nAtoms > len(fps) else 0

    # Aggregate score
    raw_score = score1 + score2 + score3
    scaled_score = scaleScore(raw_score)
    return scaled_score

def scaleScore(raw_score, min_score=-4.0, max_score=2.5):
    sascore = 11. - (raw_score - min_score + 1) / (max_score - min_score) * 9.
    sascore = min(max(sascore, 1), 10)
    if sascore > 8:
        sascore = 8. + math.log(sascore - 8 + 1)
    return sascore

def sa_score(smiles):
    """
    Check the synthetic accessibility score (https://doi.org/doi:10.1186/1758-2946-1-8) for a SMILES string
    using the RDKit SA score implementation, sascorer.

    :param smiles: str, SMILES string to be checked

    :return: float, smiles SA score
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        score = calculateScore(mol)

        return round(calculateScore(mol), 2)
    except:
        return None


def plot_molecules(smiles="c1ccccc1", path=''):
    # Check if the molecule is in PubChem
    pubchem_status = check_pubchem(smiles)
    
    # Generate RDKit molecule object from SMILES string
    rd_mol = Chem.MolFromSmiles(smiles)
    # Generate 2D coordinates for the molecule
    AllChem.Compute2DCoords(rd_mol)
    # Create an image of the molecule
    mol_img = Draw.MolToImage(rd_mol, size=(300, 300))

    if not pubchem_status:
        # Initialize drawing context
        draw = ImageDraw.Draw(mol_img)
        
        # Define text properties
        star_color = "red"
        text_position = (200, 270)  # Move text more to the right and bottom

        # Draw a simple star as an asterisk (*)
        star_position = (20, 20)  # Position of the star at the top left
        font_size = 10  # Adjust as needed for visibility
        try:
                # Try to use a default or specified font for the star and text
                font_star = ImageFont.truetype("arial.ttf", 6)
        except IOError:
            print('IOError')
            # If the specified font is not found, use default PIL font, adjusting size if possible
            font_star = ImageFont.load_default()
            

		# Draw star and text
        draw.text(star_position, "*", fill=star_color, font=font_star)
        
    
    # Save the image
    mol_img.save(path + '_molecule.png')



'''
    # Generate RDKit molecule object from SMILES string
    rd_mol = Chem.MolFromSmiles(smiles)
    # Generate 2D coordinates for the molecule
    _discard = AllChem.Compute2DCoords(rd_mol)
    # Use RDKit to create an image object of the molecule
    Draw.MolToFile(rd_mol, filename = path + '_molecule.png', size=(300,300), fitImage=False, imageType='png')'''
    
def filter_duplicates(df):
    """
    Filter out duplicates based on SMILES and molecular identity (using InChI).
    """
    # Ensure df_ is a copy to avoid SettingWithCopyWarning when modifying it later
    df_ = df.dropna(subset=['SMILES']).copy()

    # Remove duplicate SMILES strings
    df_no_dupes_smiles = df_.drop_duplicates(subset='SMILES', keep='first')

    # Function to safely convert SMILES to InChI, returning None for invalid molecules
    def safe_smiles_to_inchi(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return inchi.MolToInchi(mol)
        else:
            return None

    # Apply InChI conversion safely and directly using loc to avoid potential future SettingWithCopyWarning
    df_no_dupes_smiles['InChI'] = df_no_dupes_smiles['SMILES'].apply(safe_smiles_to_inchi)
    df_no_dupes_smiles = df_no_dupes_smiles.dropna(subset=['InChI'])  # Drop rows where InChI conversion failed

    # Remove duplicates based on InChI
    df_no_dupes = df_no_dupes_smiles.drop_duplicates(subset='InChI', keep='first').reset_index(drop=True)

    # Remove the InChI column as it's no longer needed
    df_no_dupes = df_no_dupes.drop(columns=['InChI'])

    return df_no_dupes

def sort_k_radius(generated_results, k):
    """
    Sort by 'radius', apply filter to remove duplicates, and return the top k+1 entries.
    """
    # Check if generated_results is a dict and convert it to DataFrame if true
    if isinstance(generated_results, dict):
        df = pd.DataFrame.from_dict(generated_results)
    elif not isinstance(generated_results, pd.DataFrame):
        raise TypeError("generated_results must be a dict or a pandas DataFrame")
    else:
        df = generated_results
    
    # Sort by 'radius'
    df_sorted = df.sort_values(by='radius').reset_index(drop=True)
    
    # Apply filter to remove duplicates
    df_filtered = filter_duplicates(df_sorted)
    
    # Select the top k+1 entries
    top_k_1 = df_filtered[:min(k+1, len(df_filtered))]
    
    return top_k_1


def plot_k_molecules_radius(generated_results,k,folder):
    top_k_1_generated_results = sort_k_radius(generated_results, k)
    
    # Ensure no unwanted index-like columns are saved
    # This assumes 'SMILES' and 'radius' are the only columns you want to save
    # Adjust the column list based on your actual DataFrame structure
    

    columns_to_save = ['dimension','direction', 'radius','SMILES','SELFIES']  # Add any other columns you need to save
    filtered_df = top_k_1_generated_results[columns_to_save]
    
    filtered_df.to_csv(folder + 'top_k_1_radius.csv', index=False)

    for i in range(1+k):
        path = folder + ('radius_original' if i == 0 else 'radius_'+ str(i))
        plot_molecules(top_k_1_generated_results['SMILES'].iloc[i], path)

def plot_1d_distance_ratio_line(dataframe,path,marker_size=10):
    distance_ratio_ = dataframe['distance_ratio']
    # Assuming 'distance_ratio_' is a Pandas Series
    last_value = distance_ratio_.iloc[-1]  # Correct way to access the last element in a Pandas Series
    distance_ratio = [x / last_value for x in distance_ratio_]

    plt.figure(figsize=(15, 6))  # Example: 10 inches by 6 inches
    plt.hlines(1, 0, 1)  # Draw a horizontal line
    plt.xlim(-0.3, 1.3)
    plt.ylim(0.5, 1.5)

    y = np.ones(np.shape(distance_ratio))  # Make all y values the same

    # Plot the dots and add labels
    for i, value in enumerate(distance_ratio):
        color = 'red' if i == 0 or i == len(distance_ratio) - 1 else 'blue'
        plt.plot(value, y[i], 'o', ms=marker_size, color=color)  # Plot dot
        plt.text(value, y[i] - 0.05, f'{value:.4f}', ha='center', va='top', fontsize=8, rotation=285)
  # Add label below dot

    plt.axis('off')
    plt.savefig(path + '/1d_distance_ratio.png')
    plt.close()

