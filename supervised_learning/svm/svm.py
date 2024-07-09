import torch.nn as nn
import torch

batch_size = 64

class SVMModel(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        output = self.linear(x)
        return output
    
class SVMLoss(nn.modules.Module):
    def __init__(self):
        super(SVMLoss,self).__init__()
    def forward(self, outputs, labels):
         return torch.sum(torch.clamp(1 - outputs.t()*labels, min=0))/batch_size