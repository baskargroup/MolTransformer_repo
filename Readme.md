
This following section of README file will guide users through the process of setting up a Conda environment named "Env_MolTransformer" specific to your library's requirements.
# Setting Up Env_MolTransformer

## Step 1: Install Conda
If you haven't already installed Conda, download and install it from the [official Conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

## Step 2: Download Environment File
Ensure you have the `environment.yml` file from the MolTransformer library.

## Step 3: Create the Conda Environment
Open your terminal or command prompt and navigate to the directory containing the `environment.yml` file. Run the following command:

'''bash
conda env create -f environment.yml -n Env_MolTransformer

conda activate Env_MolTransformer
conda list
'''
# Quick Start of Running the code.
## Step 1: edit config.jason especially to path and model_mode
## Step 2: run the code
For cpu: python test_main.py
For gpu multi node: CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --nproc_per_node=4 test_main.py


# train_config.jason setting
you only need to edit train_config.jason if you want to train your own model.
model_mode chosse from: ['SS', 'HF','multiF_HF','SS_HF','Descriptors'] 

If model_mode = multiF_HF and pretrain_model_type = HF, please remember to edit a pretrain HF_path to pretrain_model_file for the multiF_HF to transfer learning from there.
dataset: only matter when model_mode in ['HF','multiF_HF','SS_HF','Descriptors'], choose from ['ocelot','qm9'], if user_data = true, then this dataset will not matter. 

Note: user_data will over-write dataset
train_only_lock_layer choose from : ['Na', 'SS','fc1fc2',"SS_fc1fc2"]

Now, the package offer the following model to ty and keep trying if you wish, please set user_data = false, and choose from the following comibination: [ "dataset" = 'qm9', "high_fidelity" = 'lumo' ; "dataset" = 'ocelot', "high_fidelity" = 'aea']

If you are using Generative method, please change config.jason


some use exaample 1:
from MolTransformer import *

print('imported')
MO = ModelOperator()
print('---------build model operator-----------')
#MO.evaluate_decoder(num_batch = 1)
#MO.train()
MO.r_square(num_batch = 100)
example 2:
from MolTransformer import *
build_model_instance = BuildModel(model_mode = 'SS')
model = build_model_instance.model
print("loaded SS model")

build_model_instance = BuildModel(dataset = 'ocelot')
model = build_model_instance.model
print("loaded multiHF_hf ocelot model")

build_model_instance = BuildModel(dataset = 'qm9')
model = build_model_instance.model
print("loaded multiHF_hf qm9 model")



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
### Example Usage
Example 1: Using the DataLoader with default settings for the 'qm9' dataset
data_loader = DataLoader(dataset='qm9')

Example 2: Using custom data with specified paths
custom_data_path = {
    'train': ['/path/to/your/train_data.csv'],
    'test': ['/path/to/your/test_data.csv']
}
data_loader = DataLoader(data_path=custom_data_path, label='your_label_here', gpu_mode=True)

Example 3: Handling 'ocelot' dataset with GPU acceleration
data_loader = DataLoader(dataset='ocelot', gpu_mode=True)
