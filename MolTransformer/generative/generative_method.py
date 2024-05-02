from ..model import BuildModel
from .generative_utils import *
import torch  # type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config('config')

class GenerateMolecules():
     def __init__(self):
        
        build_model_instance = BuildModel(device=device,model_mode='SS',
            gpu_mode = config["gpu_mode"] ,train = True, 
            preload_model=global_config['pretrain_model_type'], pretrain_model_file=global_config['pretrain_model_file'])
        self.model = build_model_instance.model
        #change test 
     
          
          

