import torch
import torch.nn as nn

class MLPModel(nn.Module):
    """
    MLP Model: Flatten -> Linear -> ReLU -> Linear
    """
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(n_inputs, 256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256, n_outputs)

    def forward(self, x):
        output = self.flatten(x)
        output = self.linear1(output)
        output = self.relu(output)
        output = self.linear2(output)
        return output