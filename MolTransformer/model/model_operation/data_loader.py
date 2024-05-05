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
        model_mode (str): Model mode, default 'SS'. Options: 'SS', 'HF', 'multiF_HF', 'SS_HF', 'Descriptors'.
        gpu_mode (bool): If True, enables GPU mode.
        dataset (str): Predefined dataset to use ('qm9' or 'ocelot').
        data_path (dict): Paths for training and testing data.
        label (str): Label column name, depends on dataset and model_mode.
        report_save_path (str): Path to save reports.

    Attributes:
        user_data (bool): True if using user-provided data.

    Raises:
        ValueError: If both or one of the train/test paths are not specified when using custom data.
    """
    def __init__(self, model_mode='', gpu_mode=False, dataset='SS', data_path={'train': [''], 'test': ['']}, label='',report_save_path = '', save = False):
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
                        "Please specify your model_mode from 'SS', 'HF', 'multiF_HF', 'SS_HF', or 'Descriptors'. If the model_mode is not 'SS', ensure the label is defined and the label column exists in your CSV files."
                    )
            print(message)
            logging.warning(message)
        else:  # no user_data
            if self.dataset == 'SS':
                message = "The 'model_mode' is set to default 'SS'."
                print(message)
                logging.info(message)
                self.model_mode = 'SS'
            else:
                if dataset not in ['qm9', 'ocelot']:
                    raise ValueError("Invalid dataset specified. Please choose either 'qm9' or 'ocelot'.")
                message = "The 'model_mode' will default to 'multiF_HF'. To customize, specify model_mode explicitly, e.g., DataLoader(dataset='ocelot', model_mode='HF'). Ensure 'label' is set appropriately."

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
        self.save = save or bool(report_save_path)
        if  self.save  and not report_save_path:
            report_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            report_save_path = os.path.join(report_base_dir, 'output','user_output')
            print('Resault will be save to the following path: ', report_save_path)
        
        if self.save:
            check_path(report_save_path)

        Data = DataProcess(model_mode = self.model_mode ,data_path = data_path,high_fidelity_label =label ,report_save_path = report_save_path)
        logging.info("********train size :  " + str(len(Data.dataset_train)) + " ***************")
        logging.info("********test size :  " + str(len(Data.dataset_test)) + " ***************")
        if self.gpu_mode:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(Data.dataset_train)
            self.train =  torch.utils.data.DataLoader(Data.dataset_train,batch_size=settings.data_loader_batch_size,sampler=self.train_sampler ,num_workers=self.gpu_world_size,pin_memory=True)        
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(Data.dataset_test, shuffle=False)
            self.test = torch.utils.data.DataLoader(Data.dataset_test, batch_size=settings.data_loader_batch_size, sampler=self.test_sampler, num_workers=self.gpu_world_size, pin_memory=True)
            
        else:
            self.train =  torch.utils.data.DataLoader(Data.dataset_train,batch_size=settings.data_loader_batch_size,pin_memory=True) 
            self.test = torch.utils.data.DataLoader(Data.dataset_test, batch_size=settings.data_loader_batch_size,  pin_memory=True)
                          
        if self.model_mode != 'SS':
            self.std_parameter = Data.std_parameter 
        del Data

                




        

        
