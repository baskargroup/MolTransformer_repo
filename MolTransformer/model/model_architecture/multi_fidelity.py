# multi_f_high_fidelity.py
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from .mini_hf import mini_HF  
from . import settings 

class MultiFidelity(nn.Module):
    """
    MultiFidelity is a PyTorch module for predicting properties of chemical compounds,
    which considers both sequence information from embeddings and additional descriptors.
    It integrates these two sources of information to make a more informed prediction.
    
    Attributes:
        max_sequence_length (int): The maximum length of the sequence embeddings.
        embedding_size (int): The dimensionality of the sequence embeddings.
    """
    def __init__(self):
        """
        Initializes the MultiFidelity model with the necessary layers, including
        dropout for regularization, a layer normalization step, and an instance of a
        mini High Fidelity model for final prediction.
        
        Parameters:
            max_sequence_length (int): The maximum length of the sequence embeddings.
            embedding_size (int): The dimensionality of the sequence embeddings.
        """
        super().__init__()
        self.max_sequence_length = settings.max_sequence_length +1
        self.embedding_size = settings.embedding_size
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear((settings.max_sequence_length +1) * settings.embedding_size, 1500)  
        self.fc2 = nn.Linear(1500, 128)
        self.norm3 = nn.LayerNorm(266 + 128)
        self.hf0 = mini_HF()
        #self.hf = nn.Linear(266 + 128,1)
    def forward(self,memory_,descriptors):
        """
        Defines the forward pass of the MultiFHighFidelity model. Takes in both memory
        embeddings and descriptors, processes them, and produces a prediction.

        Parameters:
            memory_ (torch.Tensor): The input tensor containing the sequence of embeddings.
            descriptors (torch.Tensor): The additional input tensor containing descriptors.
        
        Returns:
            torch.Tensor: The output prediction tensor of the model.
        """
        x = F.relu(self.fc1(memory_.reshape((-1,self.max_sequence_length*self.embedding_size))))
        x = self.dropout(x) 
        x = F.relu(self.fc2(x))
        x = self.dropout(x) 
        x = torch.cat((x,descriptors.reshape((-1,266)) ),dim = 1)
        x = self.norm3(x)
        #prediction = F.relu(self.hf(x))
        prediction = self.hf0(x)
        return prediction