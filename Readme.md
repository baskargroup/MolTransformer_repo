# **MolGen-Transformer: A Molecular Language Model for Generation and Latent Space Exploration**



## **Overview**
MolGen-Transformer is a **transformer-based generative AI model** designed for **molecular generation and latent space exploration**, specifically targeting **π-conjugated molecules**. By leveraging a **latent-space-centered approach** and a **SELFIES-based molecular representation**, MolGen-Transformer ensures **100% molecular reconstruction accuracy**, enabling robust and reliable generation of chemically meaningful molecules.

> **New!** MolGen-Transformer is trained with the **[OCELOT Plus dataset](https://huggingface.co/datasets/D3TaLES/OCELOT_Plus)**, providing an expanded dataset for more comprehensive coverage of π-conjugated molecules.

This repository provides the **MolGen-Transformer model, sampling methods, and analysis tools** for generative molecular design, facilitating AI-driven **chemical discovery, structure optimization, and property-based molecular exploration**.

This work is described in detail in our paper:  
[**MolGen-Transformer: A Molecule Language Model for the Generation and Latent Space Exploration of π-Conjugated Molecules**](https://chemrxiv.org/engage/chemrxiv/article-details/67bce95d81d2151a02e708ba) available on **ChemRxiv**.

---
## **Key Features**
MolGen-Transformer addresses major challenges in generative molecular AI, including **chemical space coverage, latent space interpretability, and generation reliability**, by implementing the following capabilities:

- **Diverse Molecular Generation**: Randomly samples molecules from latent space to ensure diverse structural outputs.
- **Controlled Molecular Generation**: Allows similarity-controlled generation for tuning molecular diversity and resemblance.
- **Molecular Interpolation**: Identifies intermediate structures between two molecules, aiding in **reaction pathway discovery**.
- **Local Molecular Generation**: Enables the **refinement and optimization** of molecules by manipulating latent space vectors locally.
- **Neighboring Search**: Iteratively searches neighboring molecules to **optimize a given molecular property** using a multi-fidelity model.
- **Molecular Evolution**: Evolves molecules along a path in latent space, allowing **progressive optimization from a starting molecule to a target structure**.
- **SMILES & SELFIES Conversion**: Smiles and Selfies Conversion**: Converts SMILES to latent space representations and vice versa.


## **Configuration Details**
- **Trained on**: ~198 million organic molecules
- **Latent Space Encoding**: SELFIES representation for guaranteed chemical validity
- **Computation Mode**: GPU acceleration supported for efficient molecular generation
- **Output Storage**: Customizable **report save path** for logs and results
- **GPU Mode**: Enables computations on a GPU to speed up processing.
- **Report Save Path**: Specifies the directory for saving outputs and logs.



## Table of Contents
- [GenerateMethods Usage Guide](#generatemethods-usage-guide)
- [Installation](#installation)
  - [1. Create a Conda Environment (Recommended)](#1-create-a-conda-environment-recommended)
  - [2. Install RDKit from conda-forge](#2-install-rdkit-from-conda-forge)
  - [3. Install the remaining dependencies via pip](#3-install-the-remaining-dependencies-via-pip)
  - [4. Install MolTransformer as a package](#4-install-moltransformer-as-a-package)
  - [5. Test Your Installation](#5-test-your-installation)
- [Quick Start](#quick-start)
  - [Example 1: Global Molecular Generation](#example-1-global-molecular-generation)
  - [Example 2: Local Molecular Generation](#example-2-local-molecular-generation)
  - [Example 3: Neighboring Search](#example-3-neighboring-search)
  - [Example 4: Custom Latent Space Manipulation](#example-4-custom-latent-space-manipulation)
  - [Example 5: Simplified Molecular Evolution Between Two Molecules](#example-5-simplified-molecular-evolution-between-two-molecules)
- [Configuration Files](#configuration-files)
  - [config.json](#configjson)
  - [global_config.json](#global_configjson)
- [Addition Information and Notes for Installation](#addition-information-and-notes-for-installation)



## **Installation**

### 1. Clone the Repository

```bash
git clone https://github.com/baskargroup/MolTransformer_repo.git
cd MolTransformer_repo
```

### 2. Set Up Conda Environment

If you don’t already have conda, [install Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.  
*Note: For systems without conda (e.g., using micromamba), see the [Additional Information and Notes](#addition-information-and-notes-for-installation).*  

Create and activate a new environment:

```bash
conda create -n moltransformer python=3.9
conda activate moltransformer
```

### 3. Install Dependencies

- **Install RDKit:**

  ```bash
  conda install -c conda-forge rdkit
  ```

- **Install PyTorch:**

  If you have an NVIDIA GPU:

  ```bash
  conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
  ```

  *(Adjust the cudatoolkit version based on your hardware.)*

  For CPU-only PyTorch:

  ```bash
  conda install pytorch torchvision torchaudio cpuonly -c pytorch
  ```

- **Install Remaining Dependencies:**

  ```bash
  pip install -r requirements.txt
  ```

### 4. Install MolTransformer (Pip Installable)

Install MolTransformer as an editable package (for development and latest updates):

```bash
pip install -e .
```

Or directly via pip (stable release):

```bash
pip install moltransformer
```

### 5. Test Installation

Verify your installation by running:

```python
import MolTransformer
print(MolTransformer.__file__)
```

If there are no errors, your setup is complete!
## Quick Start


#### Example 1: Global Molecular Generation
This example demonstrates how to generate a set number of molecular structures randomly across the latent space. Note that the number of unique molecules may be less than requested due to potential duplicates.
```python
from MolTransformer import GenerateMethods
GM = GenerateMethods(save=True)  # Set `save=True` to save results and logs
smiles_list, selfies_list = GM.global_molecular_generation(n_samples=100)
```

#### Example 2: Local Molecular Generation

Generates new molecules around an initial molecule by exploring its local latent space neighborhood. By default, it saves generated results and provides an option to select top-k closest molecules.

**Parameters:**

- `initial_smile` *(str, optional)*: The SMILES string of your reference molecule. If omitted, a random molecule is selected from the dataset.
- `report_save_path` *(str, optional)*: Path to save generated results. Default path: `output/GenerateMethods/`.

```python
from MolTransformer import GenerateMethods

GM = GenerateMethods(save=True)
generated_results = GM.local_molecular_generation(
    dataset='qm9', random_initial_smile=False, initial_smile='C1=CC=CC=C1', num_vector=30)
print("Generated SMILES:", generated_results['SMILES'])
print("Generated SELFIES:", generated_results['SELFIES'])
```

---

#### Example 3: Neighboring Search

Performs minimal perturbations using a binary search in the latent space around a provided molecule, generating closely related molecular variants. Ideal for iterative exploration and optimization of molecular structures. The perturbation magnitude can be controlled with parameters.

**Parameters:**

- `initial_smile` *(str)*: SMILES string of the reference molecule (required).
- `search_range` *(float, optional)*: Maximum perturbation range (default: `10.0`).
- `resolution` *(float, optional)*: Perturbation resolution (default: `0.1`).
- `report_save_path` *(str, optional)*: Path to save generated results.

```python
from MolTransformer import GenerateMethods

GM = GenerateMethods(save=True, report_save_path='./output/NeighborSearch/')
initial_smile = 'C1=CC=CC=C1'  # Benzene example
generated_results, fail_cases = GM.neighboring_search(initial_smile=initial_smile, num_vector=20)

print('Generated SMILES:', generated_results['SMILES'])
print('Generated SELFIES:', generated_results['SELFIES'])
```

#### Example 4: Custom Latent Space Manipulation

This example shows how to manually manipulate the latent space representation of a molecule to explore structural variations. 

**Example usage:**

```python
from MolTransformer import GenerateMethods

# Initialize generator
GM = GenerateMethods()

# Select or define an initial SMILES molecule
initial_smile = GM.random_smile(dataset='qm9')
print('Initial SMILE:', initial_smile)

# Convert SMILES to latent space
latent_space = GM.smiles_2_latent_space([initial_smile])
print('Latent Space Shape:', latent_space.shape)

# Manually modify latent space here if desired
# Example: latent_space += np.random.normal(0, 0.1, latent_space.shape)

# Convert the modified latent space back to SMILES/SELFIES
edit_results = GM.latent_space_2_strings(latent_space)
print('Edited SMILE:', edit_results['SMILES'][0])
print('Edited SELFIES:', edit_results['SELFIES'][0])

```

#### Example 5: Molecular Evolution Between Two Molecules

The `molecular_evolution` function generates intermediate molecules along the latent space pathway connecting two specified molecules.

**Parameters:**

- `start_molecule` *(str)*: SMILES string of the initial molecule.
- `end_molecule` *(str)*: SMILES string of the target molecule.
- `number` *(int)*: Number of intermediate molecules generated.

The function saves results automatically if `save=True`, including generated molecules, similarity scores, and visualizations.

**Example Usage:**

```python
from MolTransformer import GenerateMethods

# Initialize with output saving enabled
GM = GenerateMethods(report_save_path='/path/to/save/reports/', save=True)

# Define starting and target molecules (SMILES)
start_molecule = 'c1ccccc1'   # Benzene (example)
end_molecule = 'C1CCCCC1'     # Cyclohexane example

# Generate intermediate molecules
results_df = GM.molecular_evolution(start_molecule, end_molecule, number=100)

# Display generated molecules and similarities
print(results_df[['SMILES', 'distance_ratio', 'similarity_start', 'similarity_end']])
```

## Configuration

To customize and effectively utilize the MolGen-Transformer examples provided, edit the configuration parameters in the following files:

### `config.json`

- **`gpu_mode` (bool, default: false)**:

  Controls whether MolGen-Transformer uses parallel GPU computation.  
  - If `true`, the package runs in distributed parallel mode across multiple GPUs. Set to `true` only if parallel GPU execution is explicitly required.
  - If `false` (default), MolGen-Transformer automatically detects if CUDA is available and runs in single-GPU mode or CPU if GPU is unavailable.

- **`output_folder_name` (str)**:

  Specifies the directory name for saving generated outputs and reports.

Example:

```json
{
  "gpu_mode": false,
  "output_folder_name": "generated_molecules"
}
```

### `global_config.json`

- **`model_mode` (str)**:
  Determines the model used for generation. Default is `"SS"` (self-supervised).

- **`gpu_mode` (bool)**:
  Same functionality as described above (`config.json`).

- **`batch_size` (int)**:
  Specifies batch size for molecular generation and inference.

- **`report_save_path` (str)**:
  Path for saving logs and generated reports.

- **`model_save_folder` (str)**:
  Directory for storing trained model checkpoints.

- **`data_path` (dict)**:
  Specifies file paths for user-provided datasets for training/testing. Required only if using custom data.

Example:

```json
{
  "model_mode": "SS",
  "gpu_mode": false,
  "batch_size": 64,
  "report_save_path": "./output/reports",
  "model_save_folder": "./output/models",
  "data_path": {
    "train": [["train.csv"]],
    "test": [["test.csv"]]
  }
}
```

Adjust these configurations based on your computational resources and experimental needs.



## Addition Information and Notes for Installation

### Installation on Clusters Without Conda (Using micromamba)
Some HPC systems remove or disable conda, but provide micromamba as a lightweight conda-like tool. To install MolTransformer under micromamba, do:

- Load micromamba:

```bash
module purge
module load micromamba
eval "$(micromamba shell hook --shell=bash)"
```
       

- Create and activate a new environment:

```bash
micromamba create -p /path/to/my_moltransformer_env python=3.9 -c conda-forge
micromamba activate /path/to/my_moltransformer_env
```

If your cluster only offers Python 3.13, you may see a warning about the removed imp module. We’ve patched hostlist in our code to avoid this issue, but be aware that older versions of hostlist or ansiblecmdb might still reference imp.

#### Optional 

- Install GPU PyTorch:

```bash
micromamba install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

- Install RDKit (if needed):

```bash
micromamba install -c conda-forge rdkit
```

#### Install dependencies and MolTransformer:

- Option 1: Install from the repository:

```bash
pip install -r requirements.txt
pip install -e .
```

- Option 2: Install via PyPI 
```bash
pip install moltransformer
```

#### Test:

```bash
python -c "import MolTransformer; print(MolTransformer.__file__)"
```

### Notes on Python 3.13 and the imp Module
Python 3.13 fully removes the old imp module. If you see a ModuleNotFoundError: No module named 'imp', it usually means a library (e.g., ansiblecmdb, hostlist) hasn’t updated to importlib.
We’ve patched our references to hostlist so it no longer imports imp. If you still encounter problems, make sure you’re on our latest codebase.
If your cluster forcibly uses Python 3.13, you might need to manually patch or remove libraries that still depend on imp. Alternatively, if your HPC environment allows it, use Python 3.12 or earlier to avoid this issue entirely.

#### ModuleNotFoundError: No module named 'imp'
If the error occur to your systm, please do the following step:
- Open it in a text editor:
```bash
nano /work/mech-ai/bella/my_moltransformer_env_py312/lib/python3.13/site-packages/ansiblecmdb/render.py
```
- Remove or replace the line:

```python
import imp
```
to: 

```python
import importlib
```
