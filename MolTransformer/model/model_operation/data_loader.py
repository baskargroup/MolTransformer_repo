import glob
import os
import logging
import torch # type: ignore
from ..utils import DataProcess,check_path
from . import settings 

class DataLoader:
    """
    Initializes a DataLoader to manage dataset loading and processing.

    Parameters:
        model_mode (str): Specifies the model mode, default is 'SS'. Valid options are 'SS', 'HF', 'multiF_HF', 'SS_HF', or 'Descriptors'.
        gpu_mode (bool): Enables GPU mode if True.
        dataset (str): Specifies the predefined dataset to use. Options are 'qm9' or 'ocelot'.
        data_path (dict): Specifies paths for training and testing data.
        label (str): Column name for the label, dependent on dataset and model_mode.

    Attributes:
        user_data (bool): Indicates whether user-provided data is being used.

    Raises:
        ValueError: If only one of train or test data paths is specified.
    """
    def __init__(self, model_mode='', gpu_mode=False, dataset='SS', data_path={'train': [''], 'test': ['']}, label='',report_save_path = ''):
        self.model_mode = model_mode
        self.gpu_mode = gpu_mode
        self.dataset = dataset
        self.user_data = False  # Default to not using user data
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Check the content of data_path
        train_empty = data_path['train'] == ['']
        test_empty = data_path['test'] == ['']

        if train_empty and test_empty:
            self.user_data = False  # No user data provided
        elif train_empty or test_empty:
            raise ValueError(
                "Both train and test data paths must be specified if using custom data. Please ensure both 'train' and 'test' keys in 'data_path' contain paths."
                "Both train and test data paths must be specified in the form data_path={'train':['path/to/train1.csv', 'path/to/train2.csv'],"
                    "'test':['path/to/test.csv']} unless a predefined dataset ('SS', 'qm9' or 'ocelot') is used."
            )
        else:
            self.user_data = True  # User data provided and valid
        
        if self.user_data:
            # print and log warinig
            #please refine my messages 
            message = (
                        "please sepcify your  model_mode form 'SS', 'HF', 'multiF_HF', 'SS_HF', or 'Descriptors'.  and if the model_mode is not 'SS' make sure you have set label and the csv files you provided have corresponding column "
                    )
            print(message)
            logging.warning(message)
        else:  # no user_data
            if self.dataset == 'SS':
                message = (
                        "The 'model_mode' is set default 'SS'. Since the dataset is 'SS') "
                    )
                print(message)
                logging.info(message)
                self.model_mode = 'SS'
            else:
                if dataset not in ['qm9', 'ocelot']:
                    raise ValueError("Invalid dataset specified. Please choose either 'qm9' or 'ocelot'.")
                message = (
                        "The 'model_mode' is set to 'multiF_HF'. Since the dataset is 'qm9' or 'ocelot'. if you want to set model_mode to 'SS', 'HF', 'multiF_HF', 'SS_HF', or 'Descriptors', please do define model_load.ex data = DataLoader(dataset = 'ocelot',model_load = 'HF') ,please make sure you also define properly the label",
                        "the label setting will be overwrite to default label if no model_mode is define"
                    )
                print(message)
                logging.info(message)
                if not self.model_mode:
                    self.model_mode = 'multiF_HF'
                    label = 'lumo' if dataset == 'qm9' else 'aea'
                else:
                    if not label:
                        raise ValueError(' Please specify the label as you are defining model_mode, or please not define model_mode, then default seting for the dataset will be used.')
                  
            base_train_path = os.path.join(base_dir, 'model','data', dataset, 'train')
            base_test_path = os.path.join(base_dir,  'model','data', dataset, 'test')

            # List all CSV files in the training and testing directories
            train_path = glob.glob(os.path.join(base_train_path, '*.csv'))
            test_path = glob.glob(os.path.join(base_test_path, '*.csv'))

            data_path = {'train':train_path,'test':test_path}
        if not report_save_path:
            report_save_path = os.path.join(base_dir, 'output','user_output')
        
        check_path(report_save_path)

        Data = DataProcess(model_mode = self.model_mode ,data_path = data_path,high_fidelity_label =label ,save_path = report_save_path)
        logging.info("********train size :  " + str(len(Data.dataset_train)) + " ***************")
        logging.info("********test size :  " + str(len(Data.dataset_test)) + " ***************")
        if self.gpu_mode:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(Data.dataset_train)
            self.dataloader_train =  torch.utils.data.DataLoader(Data.dataset_train,batch_size=settings.batch_size,sampler=self.train_sampler ,num_workers=self.gpu_world_size,pin_memory=True)        
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(Data.dataset_test, shuffle=False)
            self.dataloader_test = torch.utils.data.DataLoader(Data.dataset_test, batch_size=settings.batch_size, sampler=self.test_sampler, num_workers=self.gpu_world_size, pin_memory=True)
            
        else:
            self.dataloader_train =  torch.utils.data.DataLoader(Data.dataset_train,batch_size=settings.batch_size,pin_memory=True) 
            self.dataloader_test = torch.utils.data.DataLoader(Data.dataset_test, batch_size=settings.batch_size,  pin_memory=True)
                          
        if self.model_mode != 'SS':
            self.std_parameter = Data.std_parameter   
        del Data

                




        

        
