import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore

class LowFidelity(nn.Module):
    """
    LowFidelity is a neural network module designed for lower-fidelity prediction tasks
    compared to HighFidelity and MultiFidelity models. It's suited for situations where a trade-off between
    performance and computational efficiency is required. This model processes sequences of
    embeddings and passes them through a series of fully connected layers.

    Attributes:
        max_sequence_length (int): The maximum length of the input sequences.
        embedding_size (int): The size of each embedding vector.
    """
    def __init__(self,max_sequence_length,embedding_size):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(max_sequence_length*embedding_size, 1500)  
        self.fc2 = nn.Linear(1500, 128)
        self.fc3 = nn.Linear(128, 8)
        self.relu = nn.ReLU()
    def forward(self,memory_):
        x = F.relu(self.fc1(memory_.reshape((-1,self.max_sequence_length*self.embedding_size))))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x