from ..model_architecture import *
from ..utils import init_distributed_mode
import torch # type: ignore
import os
import logging

# Set up logging configuration
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


class BuildModel():
    """
    Initializes and configures a model based on the specified parameters, handling both training and pre-loaded scenarios.

    Parameters:
        device (torch.device): The device to use for the model, defaults to CPU.
        model_mode (str): The mode of the model, which can be 'SS', 'HF', 'multiF_HF', 'SS_HF', or 'Descriptors'.
        gpu_mode (bool): Flag indicating whether to use GPU for model operations, defaults to False.
        train (bool): Flag indicating whether the model is being initialized for training, defaults to False.
        preload_model (str): The model to preload, defaults to the value of model_mode.
        pretrain_model_file (str): Path to a pre-trained model file to load, defaults to empty.
        dataset (str): Specifies the dataset to use, defaults to 'qm9'. Valid options are 'qm9' and 'ocelot'.

    If a dataset is specified and model_mode is not 'multiF_HF', model_mode will be overridden to 'multiF_HF'.
    If a pretrain_model_file is provided but a dataset is also specified, the file will be ignored and the model
    will default to 'multiF_HF' settings.

    Raises:
        ValueError: If the specified dataset is not valid.

    This class supports initializing a model directly with specified configurations. It handles device placement,
    distributed training setup, and loading pre-trained models based on the configuration.

    Examples of usage:
        - For a simple Self-Supervised model: BuildModel(model_mode='SS')
        - For loading a specific pre-trained High Fidelity model: BuildModel(preload_model='HF', pretrain_model_file='path/to/model')

    """
    def __init__(self,device = torch.device("cpu"),model_mode = 'SS',gpu_mode = False ,train = False,preload_model = '',pretrain_model_file = '',dataset = ''):
        
        self.device = device
        self.model_mode = model_mode
        self.gpu_mode = gpu_mode
        if not preload_model:
            preload_model = model_mode
        if dataset:
            if dataset != 'SS':
                if self.model_mode != 'multiF_HF':
                    message = (
                        "Warning: The 'model_mode' is set to a value other than 'multiF_HF'. However, "
                        "since 'dataset' is specified and not 'SS' , 'model_mode' will be overridden to 'multiF_HF'."
                    )
                    print(message)
                    logging.warning(message)
                if pretrain_model_file != '':
                    message = (
                        "Warning: The 'pretrain_model_file' is provided, but it will be ignored since 'dataset' "
                        "is specified and not 'SS', the model configuration will load default 'multiF_HF' model."
                    )
                    print(message)
                    logging.warning(message)
                self.model_mode = 'multiF_HF'
                preload_model = 'multiF_HF'
                pretrain_model_file = ''
                if dataset not in ['qm9', 'ocelot']:
                    raise ValueError("Invalid dataset specified. Please choose either 'qm9' or 'ocelot'.")
            else:
                if self.model_mode != 'SS':
                    message = (
                        "Warning: The 'model_mode' is set to a value other than 'SS'. However, "
                        "since 'dataset' is specified to 'SS' , 'model_mode' will be overridden to 'SS'."
                    )
                    print(message)
                    logging.warning(message)
                if pretrain_model_file != '':
                    message = (
                        "Warning: The 'pretrain_model_file' is provided, but it will be ignored since 'dataset' "
                        "is specified to 'SS', the model will load default pretrain 'SS' model."
                    )
                    print(message)
                    logging.warning(message)
                self.model_mode = 'SS'
                preload_model = 'SS'
                pretrain_model_file = ''


        else: #dataset not define
            pass

        if self.model_mode == 'Descriptors':
            model = DescriptorHF()
        else:
            model = ChemTransformer(device,self.model_mode, train = train,gpu = gpu_mode)
        print('done intialized')

        #set DDP
        if gpu_mode:
            model.to(self.device)
            if not torch.distributed.is_initialized():
                self.gpu_world_size = init_distributed_mode()
                print('enter init_distributed_mode')
            print('distributied model in gpu')
            self.model = torch.nn.parallel.DistributedDataParallel(module=model, find_unused_parameters=True,)
        else:
            model.to("cpu")
            self.model = model
        if not train:
            preload_model = self.model_mode
        else:
            preload_model = preload_model
        if not pretrain_model_file:
            if  preload_model == 'SS':   
                pretrain_model_file = self._get_SS_path()
                print('loading SS model')
            else:
                # Log the message as a warning
                message = (
                    "Default configuration loaded: A MultiF_HF model trained on the QM9 dataset targeting the LUMO property. "
                    "To use a model trained on the OCELOT dataset targeting the AEA property, specify `BuildModel(dataset='ocelot')` "
                    "when initializing your model. If you wish to load the model in SS mode without specifying a dataset, "
                    "please ensure that the 'dataset' parameter is not defined. To specify the preload model according to your needs, "
                    "you can also initialize the model with `BuildModel(preload_model='your intended mode', pretrain_model_file='your_pretrained_model_path')`."
                )

                # Print and log the same message
                print(message)
                logging.warning(message)
                pretrain_model_file = self._get_multiF_HF_path(dataset)

        self._pre_load_model(preload_model,pretrain_model_file = pretrain_model_file)
        

    
    def _pre_load_model(self,preload_model,pretrain_model_file):
        if self.model_mode == 'Descriptors':
            return None
        ######load pretrain model and build new top models
        if preload_model == 'Na':
            if self.model_mode in ['SS_HF','HF']:
                self._add_top_model('HF')
            elif self.model_mode  == 'multiF_HF':
                self._add_top_model('multiF_HF') 
            else:
                pass
        elif preload_model == 'SS':
            self.load_pretrain_SS_model(pretrain_model_file = pretrain_model_file)  # change to: load pretrain ss model
            self.model.eval()
            if self.model_mode != 'SS':
                if self.model_mode in ['SS_HF','HF']:
                    self._add_top_model('HF')
                elif self.model_mode == 'multiF_HF':
                    self._add_top_model('multiF_HF') 
                    print('add multi and pretrain ss')
                else:
                    pass

        elif preload_model == 'HF':
            if self.model_mode in ['SS_HF','HF']:
                self._add_top_model('HF')
                self._load_model(path = pretrain_model_file)
                self.model.eval()
            elif self.model_mode == 'multiF_HF':
                self._add_top_model('HF')
                self._load_model(path = pretrain_model_file) # load a HF model!!!
                self.model.eval()
                self._add_top_model('multiF_HF',multi_from_HF = True)
                self._copy_top_layer_to_MultiF_from_HF()
                if self.gpu_mode:
                    self.model.module.high_fidelity_model = self.model.module.multi_high_fidelity_model
                    del self.model.module.multi_high_fidelity_model
                else:
                    self.model.high_fidelity_model = self.model.multi_high_fidelity_model
                    del self.model.multi_high_fidelity_model
                    #check
        elif preload_model == 'multiF_HF':
            self._add_top_model('multiF_HF')
            self._load_model(path = pretrain_model_file)
            self.model.eval()

    def _load_model(self,path ):
        model_path = path 
        if self.gpu_mode:
            self.model.load_state_dict(torch.load(model_path))
            print('load the gpu model!! ')
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print('load the cpu model!! ')


    def _add_top_model(self,top_model_type,multi_from_HF = False):
        if top_model_type == 'HF':
            model_highF = HighFidelity().to(self.device)
            model_highF.to(self.device)
            if self.gpu_mode:
                model_highF= torch.nn.parallel.DistributedDataParallel(module=model_highF, find_unused_parameters=True,)
                self.model.module.high_fidelity_model = model_highF
            else:
                self.model.high_fidelity_model = model_highF
        elif top_model_type == 'multiF_HF':    
            model_highF = MultiFidelity().to(self.device)
            model_highF.to(self.device)
            if self.gpu_mode:
                model_highF= torch.nn.parallel.DistributedDataParallel(module=model_highF, find_unused_parameters=True,)
            if not multi_from_HF:
                if self.gpu_mode:
                    self.model.module.high_fidelity_model = model_highF
                else:
                    self.model.high_fidelity_model = model_highF
            else:
                if self.gpu_mode:
                    self.model.module.multi_high_fidelity_model = model_highF
                else:
                    self.model.multi_high_fidelity_model = model_highF
    
    def _copy_top_layer_to_MultiF_from_HF(self):
        if self.gpu_mode:
            with torch.no_grad():
                self.model.module.multi_high_fidelity_model.module.fc1.weight.copy_(self.model.module.high_fidelity_model.module.fc1.weight)
                self.model.module.multi_high_fidelity_model.module.fc1.bias.copy_(self.model.module.high_fidelity_model.module.fc1.bias)
                self.model.module.multi_high_fidelity_model.module.fc2.weight.copy_(self.model.module.high_fidelity_model.module.fc2.weight)
                self.model.module.multi_high_fidelity_model.module.fc2.bias.copy_(self.model.module.high_fidelity_model.module.fc2.bias)
        else:
            with torch.no_grad():
                self.model.multi_high_fidelity_model.fc1.weight.copy_(self.model.module.high_fidelity_model.fc1.weight)
                self.model.multi_high_fidelity_model.fc1.bias.copy_(self.model.module.high_fidelity_model.fc1.bias)
                self.model.multi_high_fidelity_model.fc2.weight.copy_(self.model.module.high_fidelity_model.fc2.weight)
                self.model.multi_high_fidelity_model.fc2.bias.copy_(self.model.module.high_fidelity_model.fc2.bias)

    def _get_SS_path(self):
        """
        Constructs the full path to the best Self-Supervised (SS) model file based on the GPU mode.
        """
        current_dir = os.path.dirname(__file__)
        package_root = os.path.abspath(os.path.join(current_dir, '../..'))
        model_file = 'Best_SS_GPU.pt' if self.gpu_mode else 'Best_SS_CPU.pt'
        model_path = os.path.join(package_root, 'model', 'models', 'best_models', 'SS_model', model_file)
        return model_path

    def _get_multiF_HF_path(self, dataset):
        """
        Constructs the full path to the best multi-fidelity HF model file based on the GPU mode and dataset.
        """
        current_dir = os.path.dirname(__file__)
        package_root = os.path.abspath(os.path.join(current_dir, '../..'))
        model_file = 'R2_HF_best.pt' if self.gpu_mode else 'model_noneDDP.pt'
        model_folder = 'ocelot_aea' if dataset == 'ocelot' else 'qm9_lumo'
        model_path = os.path.join(package_root, 'model', 'models', 'best_models', 'MultiF_HF', model_folder, model_file)
        return model_path
    
    def load_pretrain_SS_model(self,pretrain_model_file):
        # Relative path from the current file to the best_models directory
        base_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_models', 'SS_model')
        if not pretrain_model_file:
            model_filename = 'Best_SS_GPU.pt' if self.gpu_mode else 'Best_SS_CPU.pt'
            model_path = os.path.join(base_path, model_filename)
        else:
            model_path = pretrain_model_file
        if self.gpu_mode:
            self.model.load_state_dict(torch.load(model_path))
            print('load the gpu model!! ')
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print('load the cpu model!! ')
