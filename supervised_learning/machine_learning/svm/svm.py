import torch.nn as nn
import torch

class SVM(nn.Module):
    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)
    
class SVMLoss(nn.modules.Module):
    def __init__(self):
        super(SVMLoss,self).__init__()
    def forward(self, outputs, labels): 
        # Hinge Loss = max(0, 1 - t * y)
        return torch.mean(torch.clamp(1 - outputs * labels, min=0))