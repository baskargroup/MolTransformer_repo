import pickle
import numpy as np
import pandas as pd  # type: ignore
from collections import defaultdict
from .general_utils import plot_histogram
from rdkit import Chem # type: ignore
from .descriptors import molecule_descriptors
import selfies as sf # type: ignore
import os
from .general_utils import dataset_building, get_index_path
from .general_utils import check_path

class DataProcess(): 
    """
    data_path: in a form: {'train':['your csv file 1','your csv file 2'],'test':['your csv file']}
    high_fidelity_label: str, the label . note a column with the input label name is needed in the csv file

    """
    def __init__(self,model_mode,data_path,high_fidelity_label,report_save_path, save = False):
        self.save = save or bool(report_save_path)
        if  self.save  and not report_save_path:
            report_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            report_save_path = os.path.join(report_base_dir, 'output','user_output')
            print('Resault will be save to the following path: ', report_save_path)
            
        
        if self.save:
            self.report_save_path = report_save_path
            check_path(self.report_save_path)

        
        self.index_path = get_index_path()
        self.train_path = data_path['train'] 
        self.test_path = data_path['test'] 

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
        if model_mode != 'SS':
            print('not ss')
            self.label = high_fidelity_label
            data_train_sel = self._load_label_data('train',data_train_sel)
            data_test_sel = self._load_label_data('test',data_test_sel)
            plot_histogram(data1 = data_train_sel[self.label],data2 = data_test_sel[self.label],path = report_save_path ,name = self.label + '_original') 
            self.std_parameter =  defaultdict(lambda: defaultdict(float))
            data_train_sel,data_test_sel = self._std_data(data_train_sel,data_test_sel)
            if  model_mode in  ['multiF_HF','Descriptors' ]:
                print('start compute descriptors')
                data_train_sel = self._get_descriptors('train',data_train_sel)
                data_test_sel = self._get_descriptors('test',data_test_sel)
        else:
            self.label = ''
        #3. dataset_building
        print('start dataset_building')
        self.char2ind = self._load_char2ind()
        print('end char2ind')
        self.dataset_train = dataset_building(self.char2ind,data_train_sel,label = self.label)
        self.dataset_test = dataset_building(self.char2ind,data_test_sel,label = self.label)
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
            saver[self.label] += np.asanyarray(df[self.label]).tolist()
        return saver

    def _std_data(self,data_train_sel,data_test_sel):
        data_train_sel[self.label],data_test_sel[self.label] = self.standardize_data(data_train_sel[self.label],data_test_sel[self.label],self.label)
        if self.save:
            plot_histogram(data1 = data_train_sel[self.label],data2 = data_test_sel[self.label],path = self.report_save_path ,name = self.label)   
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
        if self.save:
            np.save(self.report_save_path + name + '_mean_std.npy', [mean1, std1,constant])
        self.std_parameter[name]['mean'] = mean1
        self.std_parameter[name]['std'] = std1
        self.std_parameter[name]['constant'] = constant
        return data1, data2

    def recover_standardized_data(self,data1, data2):
        if self.save:
            mean1, std1, constant = np.load(self.report_save_path + 'mean_std.npy')
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

