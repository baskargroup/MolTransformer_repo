# positional_encoding.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    PositionalEncoding module injects some information about the relative or absolute position 
    of the tokens in the sequence. The positional encodings have the same dimension as 
    the embeddings so that the two can be summed. This allows the model to make use of 
    the order of the sequence. The positional encoding is calculated using a specific 
    function of the position and the dimension.
    
    Attributes:
        d_model (int): The dimension of the embeddings.
        dropout (float): The dropout value.
        max_len (int): The maximum length of the sequence expected in the input.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initializes the PositionalEncoding module with the given model dimension, 
        dropout rate, and maximum length of the sequence.

        Parameters:
            d_model (int): The dimension of the embeddings.
            dropout (float): The dropout value.
            max_len (int): The maximum length of the sequence expected in the input.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        """
        Forward pass of the PositionalEncoding module. Adds the positional encoding 
        to the input tensor.

        Parameters:
            x (torch.Tensor): A tensor containing the embeddings for the input sequence.
        
        Returns:
            torch.Tensor: The tensor with added positional encodings.
        """
        
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)