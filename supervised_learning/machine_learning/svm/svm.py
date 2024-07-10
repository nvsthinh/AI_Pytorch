import torch.nn as nn
import torch

class SVM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)
    
class SVMLoss(nn.modules.Module):
    def __init__(self):
        super(SVMLoss,self).__init__()
    def forward(self, outputs, labels): 
        # Hinge Loss = max(0, 1 - t * y)
        labels_one_hot = torch.eye(outputs.size(1))[labels].to(outputs.device)
        return torch.mean(torch.clamp(1 - outputs * labels_one_hot, min=0))