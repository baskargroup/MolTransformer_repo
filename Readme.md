
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


# config.jason setting

model_mode chosse from: ['SS', 'HF','multiF_HF','SS_HF','Descriptors'] 

If model_mode = multiF_HF and pretrain_model_type = HF, please remember to edit a pretrain HF_path to pretrain_model_file for the multiF_HF to transfer learning from there.
dataset: only matter when model_mode in ['HF','multiF_HF','SS_HF','Descriptors'], choose from ['ocelot','qm9'], if user_data = true, then this dataset will not matter. 

Note: user_data will over-write dataset
train_only_lock_layer choose from : ['Na', 'SS','fc1fc2',"SS_fc1fc2"]

Now, the package offer the following model to ty and keep trying if you wish, please set user_data = false, and choose from the following comibination: [ "dataset" = 'qm9', "high_fidelity" = ['lumo'] ; "dataset" = 'ocelot', "high_fidelity" = ['aea']]

If you are using Generative method, please change Generative_configuration.jason
