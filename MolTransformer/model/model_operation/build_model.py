from ..model_architecture import *
from ..utils import init_distributed_mode
import torch # type: ignore
import os

class BuildModel():
    def __init__(self,device = torch.device("cpu"),model_mode = 'SS',gpu_mode = False ,train = False,preload_model = 'SS',pretrain_model_file = ''):
        
        self.device = device
        self.model_mode = model_mode
        self.gpu_mode = gpu_mode

        if model_mode == 'Descriptors':
            model = DescriptorHF()
        else:
            model = ChemTransformer(device,model_mode, train = True,gpu = gpu_mode)
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
            self.load_pretrain_SS_model()  # change to: load pretrain ss model
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
        # Get the directory of the current script
        current_dir = os.path.dirname(__file__)
        # Construct the path relative to the current script
        # Move up to the package root (MolTransformer), assuming the current script is inside 'model/model_operation'
        package_root = os.path.abspath(os.path.join(current_dir, '../..'))
        # Determine the model file based on gpu_mode
        model_file = 'Best_SS_GPU.pt' if self.gpu_mode else 'Best_SS_CPU.pt'
        # Append the relative path to the target file
        model_path = os.path.join(package_root, 'models/best_models/SS_model', model_file)
        
        return model_path