# MolTransformer



## GenerateMethods Usage Guide

# **MolGen-Transformer: A Molecular Language Model for Generation and Latent Space Exploration**


## **Overview**
MolGen-Transformer is a **transformer-based generative AI model** designed for **molecular generation and latent space exploration**, specifically targeting **π-conjugated molecules**. By leveraging a **latent-space-centered approach** and a **SELFIES-based molecular representation**, MolGen-Transformer ensures **100% molecular reconstruction accuracy**, enabling robust and reliable generation of chemically meaningful molecules.

This repository provides the **MolGen-Transformer model, sampling methods, and analysis tools** for generative molecular design, facilitating AI-driven **chemical discovery, structure optimization, and property-based molecular exploration**.

This work is described in detail in our paper:  
[**MolGen-Transformer: A Molecule Language Model for the Generation and Latent Space Exploration of π-Conjugated Molecules**](https://chemrxiv.org/engage/chemrxiv/article-details/67bce95d81d2151a02e708ba) available on **ChemRxiv**.

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
  - [Example 5: Optimistic Property-Driven Molecule Generation](#example-5-optimistic-property-driven-molecule-generation)
  - [Example 6: Simplified Molecular Evolution Between Two Molecules](#example-6-simplified-molecular-evolution-between-two-molecules)
- [BuildModel Configuratiom](#buildmodel-configuratiom)
- [DataLoader Configuration](#dataloader-configuration)
- [ModelOperator Configuration](#modeloperator-configuration)
  - [Configuration Guide for train_config.json](#configuration-guide-for-train_configjson)
- [Addition Information and Notes for Installation](#addition-information-and-notes-for-installation)



## **Installation**
Clone the repository to start using MolGen-Transformer for molecular generation and analysis:

```bash
git clone https://github.com/baskargroup/MolTransformer_repo.git
cd MolTransformer_repo
```

### 1. Create a Conda Environment (Recommended)

1. If you don’t already have conda, [install Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.  
**If your system does not support conda but has micromamba, and if you encounter error messages while testing MolTransformer, please see the [Addition Information and Notes for Installation](#addition-information-and-notes-for-installation).**

- 2. **Create and activate a new conda environment**  
  ```bash
  conda create -n moltransformer python=3.9
  conda activate moltransformer
  ```

- 3. **(Optional) Install GPU-Enabled PyTorch**
If you have a suitable NVIDIA GPU and drivers:
  ```bash
  conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
  ```
(Adjust the cudatoolkit version & channel according to your hardware and preferences.)

Skip this step if you don’t have a GPU or want CPU-only PyTorch.
In that case, you can install PyTorch via pip install torch or conda install pytorch -c pytorch (CPU version).


### 2. Install RDKit from conda-forge

  ```bash
  conda install -c conda-forge rdkit
  ```

### 3. Install the remaining dependencies via pip

  ```bash
  pip install -r requirements.txt
  ```

### 4. Install MolTransformer as a package

  ```bash
  pip install -e .
  ```

### 5. Test Your Installation

After installation, open a Python shell:
```python
import MolTransformer

print(MolTransformer.__file__)
```

If you see no import errors, you’re good to go!


## Quick Start


#### Example 1: Global Molecular Generation
This example demonstrates how to generate a set number of molecular structures randomly across the latent space. Note that the number of unique molecules may be less than requested due to potential duplicates.
```python
from MolTransformer import GenerateMethods
GM = GenerateMethods(save=True)  # Set `save=True` to save results and logs
smiles_list, selfies_list = GM.global_molecular_generation(n_samples=100)
```
####  Example 2: Local Molecular Generation
This example demonstrates how to generate molecules locally around a specified molecule. 

**Parameters:**

- `initial_smile` *(str, optional)*:  
  The SMILES string of the molecule around which new structures will be generated. If not provided, a random molecule from the dataset is selected.

- `report_save_path` *(str, optional)*:  
  The path to save the generated molecules and related reports. If not specified, results will be saved to a default location (`output/GenerateMethods/`).

```python
from MolTransformer import GenerateMethods
GM = GenerateMethods(save=True)
generated_results = GM.local_molecular_generation(dataset='qm9', num_vector=30)
print(generated_results['SMILES'])
print(generated_results['SELFIES'])
```
#### Example 3: Neighboring Search
This example starts with a random SMILE from a dataset and explores its molecular neighborhood. It's particularly useful for iterative exploration and optimization of molecular structures.
```python
from MolTransformer import GenerateMethods
GM = GenerateMethods(report_save_path='your_custom_path')
initial_smile = GM.random_smile(dataset='qm9')
print('Initial SMILE:', initial_smile)
generated_results, _ = GM.neighboring_search(initial_smile=initial_smile, num_vector=20)
print('Generated SMILES:', generated_results['SMILES'])
print('Generated SELFIES:', generated_results['SELFIES'])
```
#### Example 4: Custom Latent Space Manipulation
After generating a latent space from a SMILE, you can manually perturb it to explore slight variations and their properties.
```python
from MolTransformer import GenerateMethods
GM = GenerateMethods()
initial_smile = GM.random_smile(dataset='qm9')
print('Initial SMILE:', initial_smile)
ls = GM.smiles_2_latent_space([initial_smile])
print('Latent Space Shape:', ls.shape)
# Perform custom modifications to ls as needed
edit_results = GM.latent_space_2_strings(ls)
print('Edited SMILE:', edit_results['SMILES'][0])
print('Edited SELFIES:', edit_results['SELFIES'][0])
properties = GM.smiles_2_properties(edit_results['SMILES'])
print('properties_2: ', float(properties[0][0]))
print('properties_2: shape', properties.shape)
```
#### Example 5: Optimistic Property-Driven Molecule Generation
This example showcases property-driven generation aimed at optimizing molecule properties through iterative exploration of the latent space. The k parameter selects the top k most similar neighboring molecules in each iteration, from which the molecule showing the most significant improvement in properties is chosen for the next cycle. This process necessitates defining the dataset, the number of latent space vectors to sample, and the k parameter that dictates the extent of the neighborhood search.
```python
from MolTransformer import GenerateMethods
GM = GenerateMethods(save=True)  # Enable saving of results and logs
molecules_generation_record = GM.optimistic_property_driven_molecules_generation(dataset='qm9',k=30,num_vector=100) # you can also specify your initail_smile = 'your interested molecule'
print('Generated SMILES:', molecules_generation_record['SMILES'])
print('Properties:', molecules_generation_record['Property'])
```

#### Example 6: Simplified Molecular Evolution Between Two Molecules
The `molecular_evolution` function conducts a series of operations to transition from the structure of `start_molecule` to `end_molecule`. It explores the latent space to propose potential intermediates and applies property optimization techniques to refine the transformation pathway. This process helps in understanding how one molecular configuration can be converted into another, potentially uncovering viable synthetic routes or novel molecular structures.

```python
# Import the GenerateMethods class
from MolTransformer import GenerateMethods
# Initialize the GenerateMethods with an output path
GM = GenerateMethods(report_save_path='/path/to/save/reports/')
# Randomly pick one molecule from each dataset
start_molecule = 'c1ccccc1'  # Example SMILE from 'qm9'
end_molecule = 'c2ccc(c1ccccc1)cc2'  # Example SMILE from 'ocelot'

# Perform molecular evolution and observe the transformation
GM.molecular_evolution(start_molecule, end_molecule, number=100)
print('Molecular evolution completed from qm9 to ocelot molecule.')
```

## BuildModel Configuratiom;
### Overview
The BuildModel class simplifies the initialization and configuration of models tailored for different machine learning tasks in the MolTransformer project. It handles device setup, model initialization, and pre-loading of models with detailed customization options.

### Configuration Parameters
#### Basic Parameters
- **device**: The computation device (CPU or GPU) used for the model. Default is CPU.
- **model_mode**: Type of the model ('SS', 'HF', 'multiF_HF', 'SS_HF', 'Descriptors'). Determines the model's architecture and behavior.
- **gpu_mode**: Enables GPU acceleration if set to True. Improves performance and supports parallel processing.

#### Model Loading
- **train**: Indicates if the model is in training mode.
- **preload_model**: Specifies which model to load initially, defaults to the value of `model_mode`.
- **pretrain_model_file**: Path to a pre-trained model file.

#### Dataset Handling
- **dataset**: Dataset to use ('qm9' or 'ocelot'). Determines how the model is configured and initialized.
  - Default behavior: If `dataset` is not 'SS', `model_mode` will automatically adjust to 'multiF_HF'.

### Usage Scenarios
```python
from MolTransformer import BuildModel
# Example 1: Initialize a self-supervised model with default settings
build_model_instance = BuildModel(model_mode='SS')
model = build_model_instance.model
print("Loaded SS model")

# Example 2: Load a MultiF_HF model for the 'ocelot' dataset with GPU acceleration
build_model_instance = BuildModel(dataset='ocelot', gpu_mode=True)
model = build_model_instance.model
print("Loaded MultiF_HF ocelot model")

# Example 3: Load a MultiF_HF model for the 'qm9' dataset
build_model_instance = BuildModel(dataset='qm9')
model = build_model_instance.model
print("Loaded MultiF_HF qm9 model")

# Example 4: Initializing a MultiF_HF model with a pre-loaded SS model and user's pre-trained model file
build_model_instance = BuildModel(
    model_mode='MultiF_HF',
    preload_model='SS',
    pretrain_model_file='/path/to/user/pretrain_model.pt'
)
model = build_model_instance.model
print("Loaded MultiF_HF model with SS pre-training")

```

## DataLoader Configuration
### Overview
The DataLoader in the MolTransformer project is designed to facilitate the loading and handling of chemical datasets for machine learning models. This guide provides instructions on how to configure the DataLoader, including details on dataset selection, custom data integration, and GPU utilization.
### Configuration Details
#### Dataset Selection
Default Setting: If data_path is not specified, the DataLoader defaults to the 'qm9' dataset.
Options:
qm9: Utilizes 'lumo' as the default label for high-fidelity calculations.
ocelot: Uses 'aea' as the default label for high-fidelity calculations.
#### Data Path Configuration
Custom Data Usage: To use custom data, ensure it is in CSV format. The file must include a 'SELFIES' column. If using the model for property prediction, ensure the label specified in the label parameter exists in your CSV.
Setting Data Paths: Provide a dictionary with keys 'train' and 'test', each pointing to lists of your data file paths.
Example: data_path={'train': ['path/to/train1.csv', 'path/to/train2.csv'], 'test': ['path/to/test.csv']}
#### GPU Mode
Enabling GPU Mode: Set gpu_mode to True to enable processing on a GPU, enhancing computation speed and efficiency, particularly for parallel processing tasks.
### Important Notes
Ensure both 'train' and 'test' paths are specified when using custom data. Failing to specify both will default the DataLoader to use the preconfigured datasets ('qm9' or 'ocelot').
Explicitly define both paths to avoid default settings. The system will not infer missing paths.
### Example Usage of DataLoader


```python
from MolTransformer import DataLoader

# Example 1: Using the DataLoader with default settings for the 'qm9' dataset
data_loader = DataLoader(dataset='qm9',save = True) # save = True will auto save the histogram to printed path, or you can set report_save_path = 'your_path'

# Example 2: Using custom data with specified paths
custom_data_path = {
    'train': ['/path/to/your/train_data.csv'],
    'test': ['/path/to/your/test_data.csv']
}
data_loader = DataLoader(data_path=custom_data_path, label='your_label_here', gpu_mode=True)

#Example 3: Handling 'ocelot' dataset with GPU acceleration
data_loader = DataLoader(dataset='ocelot', gpu_mode=True)

```

## ModelOperator Configuration
### Overview
The ModelOperator in the MolTransformer project is designed for training and fine-tuning models across different modes such as Self-Supervised, High Fidelity, and Descriptors. This document details how to configure and utilize the ModelOperator effectively.

### Configuration Details
#### Model Training and Fine-Tuning
- **Primary Function:** The ModelOperator is primarily used to train new models or fine-tune existing models based on specific requirements.
- **Configuration File:** Ensure to configure the `train_config.json` file according to your training needs before initiating the ModelOperator.

#### Model Modes
- **Modes Available:** The ModelOperator supports various modes like 'SS' (Self-Supervised), 'HF' (High Fidelity), 'multiF_HF' (Multi Fidelity High Fidelity), 'SS_HF' (Self-Supervised High Fidelity), and 'Descriptors'.
- **Choosing a Mode:** Depending on the desired outcome, select an appropriate mode from the settings.

#### Dataset Configuration
- **Dataset Selection:** Choose between available datasets such as 'qm9' and 'ocelot' for training, unless `user_data` is true, which overrides the dataset selection with custom data provided by the user.

### Important Notes
- **Edit Configuration:** Before training, make sure to edit the `train_config.json` to match your specific model configuration and dataset requirements.
- **Pretrained Model Configuration:** If using a mode like 'multiF_HF' with a pretrained model, specify the path to the pretrained model in the `train_config.json`.

### Example Usage of ModelOperator

```python
# Example 1: Initialize and train a model using the default Self-Supervised mode
from MolTransformer import ModelOperator

MO = ModelOperator()
print('---------build model operator-----------')
MO.evaluate_decoder(num_batch=100)  # Use when model_mode == 'SS'
MO.train()

# Example 2: Training and evaluating a model in High Fidelity or Multi Fidelity mode
MO = ModelOperator()
MO.train()
MO.r_square(num_batch=100)  # Use when model_mode in ['HF', 'multiF_HF', 'SS_HF', 'Descriptors']
```

### Configuration Guide for train_config.json

#### Overview
This guide provides detailed instructions on how to configure the `train_config.json` file for training models within the MolTransformer framework. Editing this configuration file is essential when you wish to tailor the training process according to specific requirements.

#### Configuration Parameters
##### Model Mode
- **Parameter:** `model_mode`
- **Options:** ['SS', 'HF', 'multiF_HF', 'SS_HF', 'Descriptors']
- **Description:** Specifies the mode of the model to be trained. Choose 'SS' for Self-Supervised, 'HF' for High Fidelity, 'multiF_HF' for Multi Fidelity High Fidelity, 'SS_HF' for Self-Supervised High Fidelity, or 'Descriptors' for training models based on molecular descriptors.

##### Dataset
- **Parameter:** `dataset`
- **Options:** ['qm9', 'ocelot','SS']
- **Description:** Determines the dataset to use for training. This parameter is relevant only when `model_mode` is set to ['HF', 'multiF_HF', 'SS_HF', 'Descriptors']. The choice of dataset will not matter if `user_data` is set to true, as custom user data will override the preset datasets.

##### User Data
- **Parameter:** `user_data`
- **Options:** Boolean (true or false)
- **Description:** When set to true, the system will use user-provided datasets specified in the `data_path` configuration. Setting this to false will utilize the predefined datasets specified by the `dataset` parameter.

##### Pretrain Model Configuration
- **Parameters:** `pretrain_model_type`, `pretrain_model_file`
- **Description:** If `model_mode` is set to 'multiF_HF' and `pretrain_model_type` is 'HF', you must specify the path to a pre-trained High Fidelity model in `pretrain_model_file`. This setup facilitates transfer learning from a pre-trained High Fidelity model.

##### Training Locked Layers
- **Parameter:** `train_only_lock_layer`
- **Options:** ['Na', 'SS', 'fc1fc2', "SS_fc1fc2"]
- **Description:** Configures which layers of the model are locked during training. 'Na' indicates no layers are locked, 'SS' locks layers specific to the Self-Supervised mode, 'fc1fc2' locks the first two fully-connected layers, and 'SS_fc1fc2' locks layers specific to both Self-Supervised and the first two fully-connected layers.

##### Default Model Configurations
- **Example Configuration:** If `dataset` is set to 'qm9', the default high fidelity property to be modeled is 'lumo'. Similarly, if set to 'ocelot', the property is 'aea'. If `dataset` is set to 'SS', the defaulte model_mode is 'SS'.

##### Example Configurations in train_config.json

```json
{
  "model_mode": "multiF_HF",
  "dataset": "qm9",
  "user_data": false,
  "train_only_lock_layer": "Na",
  "pretrain_model_type": "HF",
  "pretrain_model_file": "/path/to/pretrain/hf_model.pt"
}
```


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

```bash
pip install -r requirements.txt
pip install -e .
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
