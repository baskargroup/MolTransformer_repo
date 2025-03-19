API:
pypi-AgENdGVzdC5weXBpLm9yZwIkYWIwYzVhM2QtOTMyMy00MDYxLTljNTAtZDlmYjZhZDNkMTliAAIqWzMsImUyZDQ1MGYzLTc4YTctNDAwNC04NzU4LTQ2YTdkNzljMGQ3OCJdAAAGICjlUyzwqxuEK63d-oOlioKgoAhj6fOCTwVjslsHGEOY



[testpypi]
  username = __token__
  password = pypi-AgENdGVzdC5weXBpLm9yZwIkYWIwYzVhM2QtOTMyMy00MDYxLTljNTAtZDlmYjZhZDNkMTliAAIqWzMsImUyZDQ1MGYzLTc4YTctNDAwNC04NzU4LTQ2YTdkNzljMGQ3OCJdAAAGICjlUyzwqxuEK63d-oOlioKgoAhj6fOCTwVjslsHGEOY



#### Example 4: Custom Latent Space Manipulation

This example shows how to manually manipulate the latent space representation of a molecule to explore structural variations. 

**Important:**  
Before generating or manipulating molecular structures, set the appropriate **property prediction model** using the `set_property_model` method.  
- For the QM9 dataset (`dataset='qm9'`), it will load a model predicting lumo to QM9 molecules.
- For the Ocelot dataset (`dataset='ocelot'`), it will load another model predicting AEA.

**Example usage:**

```python
from MolTransformer import GenerateMethods

# Initialize generator
GM = GenerateMethods()

# Set the property prediction model (required)
GM.set_property_model(dataset='qm9')  # or 'ocelot' depending on your data

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

# Compute and print molecular properties of the edited molecule
properties = GM.smiles_2_properties(edit_results['SMILES'])
print('Predicted Property Value:', float(properties[0][0]))
print('Properties Shape:', properties.shape)
```

- **Note**: Omitting `set_property_model()` will result in runtime errors. Ensure it is always called first for smooth execution.


#### Example 5: Optimistic Property-Driven Molecule Generation
This example showcases property-driven generation aimed at optimizing molecule properties through iterative exploration of the latent space. The k parameter selects the top k most similar neighboring molecules in each iteration, from which the molecule showing the most significant improvement in properties is chosen for the next cycle. This process necessitates defining the dataset, the number of latent space vectors to sample, and the k parameter that dictates the extent of the neighborhood search.
```python
from MolTransformer import GenerateMethods
GM = GenerateMethods(save=True)  # Enable saving of results and logs
molecules_generation_record = GM.optimistic_property_driven_molecules_generation(dataset='qm9',k=30,num_vector=100) # you can also specify your initail_smile = 'your interested molecule'
print('Generated SMILES:', molecules_generation_record['SMILES'])
print('Properties:', molecules_generation_record['Property'])
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
