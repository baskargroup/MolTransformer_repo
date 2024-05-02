import torch
import torch.nn as nn
import torch.nn.functional as F # type: ignore

class DescriptorHF(nn.Module):
    """
    DescriptorHF is a neural network module specifically designed for processing 
    descriptor data related to chemical compounds. It maps the descriptors through
    two linear transformations with a ReLU activation in between, ultimately outputting 
    a prediction based on the descriptor input.

    This model could be used for tasks such as predicting a single property of a molecule 
    given its descriptor vector.
    """
    def __init__(self):
        super().__init__()
        self.fc3 = nn.Linear(266, 32)
        self.fc4 = nn.Linear(32, 1)
    def forward(self,descriptors):
        """
        Forward pass of the DescriptorHF model. Applies two linear transformations with
        ReLU activations to the input descriptors.

        Parameters:
            descriptors (torch.Tensor): The input tensor containing molecular descriptors.
        
        Returns:
            torch.Tensor: The output prediction of the model.
        """
        x = F.relu(self.fc3(descriptors.reshape((-1,266))))
        x = F.relu(self.fc4(x))
        return x

