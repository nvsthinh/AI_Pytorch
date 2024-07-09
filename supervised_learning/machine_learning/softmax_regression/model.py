import torch.nn as nn

class SoftmaxRegression(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)
    
    def forward(self, x):
        output = self.linear(x)
        return output