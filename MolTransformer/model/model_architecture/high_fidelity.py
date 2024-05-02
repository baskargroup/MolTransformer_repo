import torch # type: ignore 
import torch.nn as nn  # type: ignore
import torch.nn.functional as F # type: ignore
from . import settings 

class HighFidelity(nn.Module): # type: ignore
    """
    HighFidelity is a neural network module designed for high-fidelity prediction tasks. 
    The model includes two fully connected layers and a 
    final output layer, all interspersed with dropout for regularization.

    Attributes:
        max_sequence_length (int): The maximum length of the input sequences.
        embedding_size (int): The size of each embedding vector.
    """

    def __init__(self):
        """
        Initializes the HighFidelity model with the provided sequence length and embedding size,
        along with the fully connected layers and dropout.
        """
        super().__init__()
        self.max_sequence_length = settings.max_sequence_length +1
        self.embedding_size = settings.embedding_size
        self.fc1 = nn.Linear(self.max_sequence_length*self.embedding_size, 1500)  
        self.fc2 = nn.Linear(1500, 128)
        self.hf0 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.1)
                
    def forward(self,memory_):
        """
        Defines the forward pass of the HighFidelity model. Applies two linear transformations
        and a dropout after each, followed by a final linear layer to produce the prediction.

        Parameters:
            memory_ (torch.Tensor): The input tensor containing the sequences of embeddings.
        
        Returns:
            torch.Tensor: The output prediction of the model.
        """
        x = F.relu(self.fc1(memory_.reshape((-1,self.max_sequence_length*self.embedding_size))))
        x = self.dropout(x) 
        x = F.relu(self.fc2(x))
        x = self.dropout(x) 
        p0 = F.relu(self.hf0(x))
        return p0