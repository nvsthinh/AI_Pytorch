import torch.nn as nn
import torch

class SVM(nn.Module):
    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Define a linear layer with input_dim as input features and 1 as output feature (for SVM)

    def forward(self, x):
        return self.linear(x)  # Output the raw score (decision function) from the linear layer

class SVMLoss(nn.Module):
    def __init__(self):
        super(SVMLoss, self).__init__()

    def forward(self, outputs, labels):
        # Hinge Loss = max(0, 1 - t * y)
        return torch.mean(torch.clamp(1 - outputs * labels, min=0))