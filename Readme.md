
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

### Dataset Selection
- **Default Behavior:** If `data_path` is not set, the DataLoader defaults to using the 'qm9' dataset.
- **Dataset Options:** You can specify `dataset` to either 'qm9' or 'ocelot' according to your requirements.
  - `qm9`: When using this dataset, the default label for high fidelity calculations is set to 'lumo'.
  - `ocelot`: For this dataset, the default high fidelity label is 'aea'.

### Data Path Configuration
- **Custom Data:** If you wish to use your own data, ensure it is formatted as a CSV file. The CSV file must include a column named 'SELFIES'. If your model mode involves property prediction, ensure that your data includes the necessary label column specified by the `label` parameter.
- **Parameter `data_path`:** This should be a dictionary with keys 'train' and 'test', each containing lists of paths to your training and testing CSV files, respectively.
  - Example: `data_path={'train': ['path/to/train.csv'], 'test': ['path/to/test.csv']}`

### GPU Mode
- **Parallel Computing:** Set `gpu_mode` to `True` to enable GPU support. This setting is recommended if your hardware supports it, as it facilitates faster computation and supports parallel processing.

### Important Notes
- Both 'train' and 'test' data paths must be provided if using custom data. If either is left unspecified, the DataLoader will default to using preconfigured datasets ('qm9' or 'ocelot').
- The system does not automatically infer missing paths; both must be explicitly defined.
- If you are training or using models that predict properties, ensure that your data includes all necessary labels as specified by your `label` parameter.

This configuration guide ensures that users can effectively prepare their data and set up the DataLoader to match the specific requirements of their computational tasks.

