import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1) # 1 input and 1 output feature
    
    def forward(self, x):
        output = self.linear(x)
        return output