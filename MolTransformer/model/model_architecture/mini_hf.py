import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore

class mini_HF(nn.Module):
    """
    The mini_HF (mini High Fidelity) model is a compact neural network module designed for
    high-fidelity prediction tasks, with a smaller footprint than larger models. It is
    particularly useful when the input features include both sequence embeddings and
    additional descriptor information, allowing the model to predict based on a rich
    feature set. The module consists of fully connected layers with layer normalization
    and dropout for regularization.
    """
    def __init__(self):
        super().__init__()
        self.fc3 = nn.Linear(128 +266, 64)
        self.dropout = nn.Dropout(0.3)
        self.norm4 = nn.LayerNorm(64)
        self.fc4 = nn.Linear(64 , 1)
    def forward(self,m):
        x = F.relu(self.fc3(m))
        x = self.norm4(x)
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        return x
    