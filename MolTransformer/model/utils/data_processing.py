import pickle
import numpy as np
import pandas as pd  # type: ignore
from collections import defaultdict
from .general_utils import plot_histogram
from rdkit import Chem # type: ignore
from .descriptors import molecule_descriptors
import selfies as sf # type: ignore
from .general_utils import Config, dataset_building, get_index_path
import os
import glob

# not hard core things as well need to change!

global_config = Config()

class DataProcess(): 
    def __init__(self,data_set_id = 0,data_name = '',user_data = True,dataset = 'qm9'):
        self.index_path = get_index_path()
        self.dataset_name = data_name
        if not user_data:
            base_train_path = os.path.join(os.path.dirname(__file__), '..', 'data',dataset, 'train')
            base_test_path = os.path.join(os.path.dirname(__file__), '..', 'data', dataset,'test')
            # List all CSV files in the training and testing directories
            self.train_path = glob.glob(os.path.join(base_train_path, '*.csv'))
            self.test_path = glob.glob(os.path.join(base_test_path, '*.csv'))
            # Print the paths to check them
            print('Train paths: ', self.train_path)
            print('Test paths: ', self.test_path)
        else:
            self.train_path = global_config['data_path']['train'][data_set_id] 
            self.test_path = global_config['data_path']['test'][data_set_id] 

        #########
        '''
        #following main process:
        #1. load selfies
        #2. load  HF if needed, then standlize + plot + save the parameters(add! to dataset+"train".npy to data_folder)
        #3. dataset_building (need clean up as well)
        '''
        #1. load selfies
        
        data_train_sel = self._load_selfies('train')
        print('number of train',len(data_train_sel['SELFIES']))
        data_test_sel = self._load_selfies('test')
        print('number of test',len(data_test_sel['SELFIES']))
        
        #2. load HF if needed
        if global_config['model_mode'] != 'SS':
            print('not ss')
            self.label_list = global_config['high_fidelity'] 
            fidelity = 'hf'
            data_train_sel = self._load_label_data('train',data_train_sel)
            data_test_sel = self._load_label_data('test',data_test_sel)
            label = self.label_list[0]
            plot_histogram(data1 = data_train_sel[label],data2 = data_test_sel[label],path = global_config['report_save_path'] ,name = label + '_original') 
            self.std_parameter =  defaultdict(lambda: defaultdict(float))
            data_train_sel,data_test_sel = self._std_data(data_train_sel,data_test_sel)
            if  global_config['model_mode'] in  ['multiF_HF','Descriptors' ]:
                print('start compute descriptors')
                data_train_sel = self._get_descriptors('train',data_train_sel)
                data_test_sel = self._get_descriptors('test',data_test_sel)

        #3. dataset_building
        print('start dataset_building')
        self.char2ind = self._load_char2ind()
        print('end char2ind')
        self.dataset_train = dataset_building(self.char2ind,data_train_sel)
        self.dataset_test = dataset_building(self.char2ind,data_test_sel)
        print('end dataset_test')
        
    
    def _load_char2ind(self):
        open_file = open(self.index_path + '/char2ind.pkl', "rb")
        char2ind = pickle.load(open_file)
        open_file.close()
        return char2ind

    def _load_selfies(self,file_type):
        paths =self.train_path if file_type == 'train' else self.test_path
        saver = defaultdict(list)
        for file in paths:
            print('in file: ',file)
            df = pd.read_csv(file)
            sel = np.asanyarray(df.SELFIES).tolist()
            saver['SELFIES'] += sel
        return saver
    
    def _load_label_data(self,file_type,saver):
        paths =self.train_path if file_type == 'train' else self.test_path
        for file in paths:
            df = pd.read_csv(file)
            for label in self.label_list:
                saver[label] += np.asanyarray(df[label]).tolist()
        
        return saver

    def _std_data(self,data_train_sel,data_test_sel):
        for label in global_config['high_fidelity']:
            data_train_sel[label],data_test_sel[label] = self.standardize_data(data_train_sel[label],data_test_sel[label],label)
            plot_histogram(data1 = data_train_sel[label],data2 = data_test_sel[label],path = global_config['report_save_path'],name = label)   
        return data_train_sel,data_test_sel

    
    def standardize_data(self,data1, data2, name = ''):
        '''if self.Args.load_std_parameter:
            mean1, std1, constant = np.load(self.Args.data_folder +self.Args.load_std_parameter_dataset_name + '_'+name + '_mean_std.npy')
            data1 = (data1 - mean1) / std1
            data2 = (data2 - mean1) / std1
            data1 = data1 + constant
            data2 = data2 + constant
        else:'''
        mean1 = np.mean(data1)
        std1 = np.std(data1)
        data1 = (data1 - mean1) / std1
        data2 = (data2 - mean1) / std1
        constant = np.ceil(np.abs(min(min(data1), min(data2))))
        data1 = data1 + constant
        data2 = data2 + constant
        np.save(global_config['report_save_path'] +self.dataset_name + '_'+ name + '_mean_std.npy', [mean1, std1,constant])
        self.std_parameter[name]['mean'] = mean1
        self.std_parameter[name]['std'] = std1
        self.std_parameter[name]['constant'] = constant
        return data1, data2

    def recover_standardized_data(self,data1, data2):
        mean1, std1, constant = np.load(global_config['report_save_path'] + 'mean_std.npy')
        data1 = data1 - constant
        data2 = data2 - constant
        data1 = data1 * std1 + mean1
        data2 = data2 * std1 + mean1
        return data1, data2
    def _get_descriptors(self,file_type,saver):
        paths =self.train_path if file_type == 'train' else self.test_path
        for file in paths:
            df = pd.read_csv(file)
            if 'SMILES' not in df.columns and 'SELFIES' in df.columns:
                df['SMILES'] = df['SELFIES'].apply(lambda x: sf.decoder(x))

            smiles = np.asanyarray(df.SMILES).tolist()
            for smi in smiles:
                molecule = Chem.MolFromSmiles(smi)
                descriptors = molecule_descriptors(molecule)
                saver['descriptors'].append(descriptors)
        return saver
