import torch
import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)  # Define a linear layer with n_inputs and n_outputs
    
    def forward(self, x):
        output = torch.sigmoid(self.linear(x))  # Apply sigmoid activation to the linear transformation
        return output