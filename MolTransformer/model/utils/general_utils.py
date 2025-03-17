import os
import pickle
import torch # type: ignore
from torch.autograd.variable import Variable # type: ignore
from torch.utils.data import Dataset # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import selfies as sf # type: ignore
from .settings import  max_sequence_length, low_fidelity_label_list
import json
import logging



class Config:
    """
    Configuration loader for JSON files in the project's config directory.

    By default, loads 'global_config.json'. Provide a different filename to load a specific configuration.

    Example usage:
        global_config = Config()
        setting_value = global_config['some_key']  # Returns None if 'some_key' doesn't exist
    """

    def __init__(self, config_name=''):
        # Default config file name
        config_name = config_name if config_name else 'global_config.json'

        # Dynamically identify repo root (assuming this file is nested two levels deep, adjust accordingly)
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../../../')
        )

        # recommended config folder at project root
        config_folder = os.path.join(repo_root, 'config')

        # construct full path to the config file
        config_path = os.path.join(config_folder, config_name)

        # Debug print statements (optional, can be removed after debugging)
        print("Attempting to load configuration from:", config_folder)

        try:
            with open(config_path := os.path.join(config_folder, config_name), 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file '{config_name}' not found in '{config_folder}'."
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in configuration file '{config_name}': {str(e)}"
            )

    def __getitem__(self, key):
        # safe retrieval, returns None if key doesn't exist
        return self.config.get(key, None)


# Load global configuration from the JSON file

global_config = Config()


def log_file():
    # Dynamically determine the repository root (assume current file is at MolTransformer/model/utils/)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

    # Define the path to the logs directory within the output folder at the repo root
    log_dir = os.path.join(repo_root, 'output', 'logs')

    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Define the full path to the log file
    log_path = os.path.join(log_dir, "log_file.log")

    # Print the path clearly to verify it
    print('Log file will be located at:', log_path)

    # Configure Python logging to save logs to this file
    logging.basicConfig(
        filename=log_path,
        filemode='a',  # Append logs to existing file
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

# Call log_file() once during initialization to start logging
log_file()


class LoadIndex:
    """
    This class loads the character to index mappings and class weights for the model.
    It constructs the path to index files based on the location of this script.
    """
    def __init__(self):
        self.char2ind, self.ind2char, self.sos_indx, self.eos_indx, self.pad_indx, self.class_weight = self._load_int2chr()
        self.vocab_size = len(self.char2ind)
    
    def _load_int2chr(self):
        # Get the directory of the current file and construct the index_path
        current_dir = os.path.dirname(__file__)
        index_path = os.path.join(current_dir, '..', 'models', 'index_path')

        # Build the full paths to the files
        char2ind_path = os.path.join(index_path, 'char2ind.pkl')
        ind2char_path = os.path.join(index_path, 'ind2char.pkl')
        class_weight_path = os.path.join(index_path, 'class_weight')

        # Load the files
        with open(char2ind_path, "rb") as file:
            char2ind = pickle.load(file)
        with open(ind2char_path, "rb") as file:
            ind2char = pickle.load(file)
        with open(class_weight_path, "rb") as file:
            class_weight = pickle.load(file)
        
        # Extract special indices
        sos_indx = char2ind['G']
        eos_indx = char2ind['E']
        pad_indx = 0
        
        return char2ind, ind2char, sos_indx, eos_indx, pad_indx, class_weight


Index = LoadIndex()
class IndexConvert:
    def __init__(self):
        # Assuming Index is an object with attributes like sos_indx, eos_indx, pad_indx, etc.
        self.Index = LoadIndex()

    def compare_2_idxs(self, target_idx, reconstruction_output_idx):
        ignore_list = [self.Index.sos_indx, self.Index.eos_indx, self.Index.pad_indx]
        test_size = len(reconstruction_output_idx)
        molecular_correct = 0
        symbol = 0
        symbol_correct = 0
        for i in range(test_size):
            target_ = [v for v in target_idx[i] if (v not in ignore_list)]
            if reconstruction_output_idx[i] == target_:
                molecular_correct += 1
            for j in range(len(target_)):
                symbol += 1
                if j < len(reconstruction_output_idx[i]) and reconstruction_output_idx[i][j] == target_[j]:
                    symbol_correct += 1
        return molecular_correct, symbol, symbol_correct, test_size

    def index_2_selfies(self, list_of_index_list):
        return [''.join(self.Index.ind2char[v] for v in sublist if v not in {self.Index.sos_indx, self.Index.eos_indx, self.Index.pad_indx , self.Index.char2ind['[nop]']}) for sublist in list_of_index_list]

    def selfies_2_smile(self, selfies_list):
        smiles = [sf.decoder(selfies) for selfies in selfies_list]
        return smiles


def check_path(path):
    """
    Checks if a path exists, and creates it if it doesn't.
    Parameters:
        path (str): The path to check.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def plot_histogram(data1, data2, path = '',name = ''):
        range = (min(min(data1), min(data2)), max(max(data1), max(data2)))
        fig = plt.figure(figsize=(10, 5))
        n, bins, patches = plt.hist(data1, bins=20, range=range, edgecolor='black', alpha=0.75, color='blue')
        n, bins, patches = plt.hist(data2, bins=20, range=range, edgecolor='black', alpha=0.75, color='orange')
        mean1 = np.mean(data1)
        mean2 = np.mean(data2)
        std1 = np.std(data1)
        std2 = np.std(data2)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Values (min={min(range)}, max={max(range)}, mean1={mean1:.2f}, std1={std1:.2f}, mean2={mean2:.2f}, std2={std2:.2f})', fontsize=12)
        plt.legend()
        plt.savefig(path + name+  '_histogram.png')
        plt.close()


class dataset_building(Dataset): 
    '''
    main purpose:
    make data['input'], data['target'] or [ data['properties'] or data['high_f']  ] if needed
    '''
    def __init__(self,char2ind,data,label = ''):
        self.count_over_400 = 0
        self.count = 0
        self.largest = 0
        self.char2ind = char2ind
        self.data = data 
        self.max_seq_len = max_sequence_length  #add index G     
        self.len = len(data['SELFIES']) 
        self.smi2vec = self.vectorize_sequence()
        self.seq_len = self.get_len()   # add index G to the begining
        # for reading labels to the data
        if global_config['model_mode'] != 'SS':
            self.label = label
            if global_config['model_mode'] in  ['multiF_HF','Descriptors']:
                self.descriptors = self.data['descriptors']
            self.num = 1
        
    def __getitem__(self,index):
        seq_len = self.seq_len[index]
        if seq_len >= self.max_seq_len : 
            seq_len = torch.tensor(self.max_seq_len)

        #should still have end
        inputs= self.smi2vec[index][:seq_len]
        targets = self.smi2vec[index][:seq_len]
        G_index = self.char2ind['G']
        E_index = self.char2ind['E']

        # lookup index for selfies from the dictionary
        inputs_padd = Variable(torch.zeros((1, self.max_seq_len + 1 ))).long()
        inputs_padd[0,0] = Variable(torch.tensor(G_index)).long()
        inputs_padd[0,1:seq_len + 1] = torch.LongTensor(inputs)

        target_padd = Variable(torch.zeros((1, self.max_seq_len  + 1 ))).long()
        target_padd[0,:seq_len] = torch.LongTensor(targets)
        target_padd[0,seq_len] = Variable(torch.tensor(E_index)).long()

        if global_config['model_mode'] == 'SS':
            sample = {'input':    inputs_padd[0], 
                      'target':   target_padd[0],
                      'length':   seq_len}
        
        elif global_config['model_mode']  in ['multiF_HF','Descriptors']:
            descriptors = torch.zeros((266))
            descriptors[:] = torch.tensor(self.descriptors[index])
            sample = {'input':    inputs_padd[0], 
                      'target':   target_padd[0],
                      'length':   seq_len,
                      'descriptors': descriptors,
                      'high_f': self._get_labels(index)}

        else:
            sample = {'input':    inputs_padd[0], 
                      'target':   target_padd[0],
                      'length':   seq_len,
                      'high_f': self._get_labels(index)}
        
        return sample
        
    def __len__(self):
        return self.len
    
    def vectorize_sequence(self):        
        keys = self.char2ind.keys()
        vectorized_seqs = [[self.char2ind.get(char, self.char2ind['[nop]']) for char in list(sf.split_selfies(sel))] for sel in self.data['SELFIES']]

        return vectorized_seqs
    
    def get_len(self):
        seq_len = [len(list(sf.split_selfies(sel))) for  sel in self.data['SELFIES']]
        return torch.LongTensor(seq_len)
    
    def padding(self):
        seqlen = len(self.seq)
        seq_tensor = Variable(torch.zeros((1, self.max_seq_len))).long()
        seq_tensor[:seqlen] = torch.LongTensor(self.seq)
        return  seq_tensor

    def _get_labels(self,index):
        properties = Variable(torch.zeros((self.num)))
        properties[0] = torch.tensor(self.data[self.label][index])
        return properties


def get_index_path():
    # Absolute path to the directory where the current script is located
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Parent directory of the current script's directory
    parent_directory = os.path.dirname(current_file_directory)
    
    # Parent directory of the parent directory (up two levels)
    grandparent_directory = os.path.dirname(parent_directory)
    # Path to the sibling directory which is at the same level as the parent
    up_sibling_directory = os.path.join(grandparent_directory, "model")
    # Path to the sibling directory which is at the same level as the parent
    sibling_directory = os.path.join(up_sibling_directory, "models")
    
    # Path to the child directory inside the sibling directory
    child_directory_path = os.path.join(sibling_directory, "index_path")
    
    return child_directory_path

