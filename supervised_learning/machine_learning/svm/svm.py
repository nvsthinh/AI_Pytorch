import torch.nn as nn
import torch

class SVM(nn.Module):
    """
    Support Vector Machine (SVM) model implemented using PyTorch.
    
    Attributes:
        linear (torch.nn.Linear): Linear transformation layer.
    
    Args:
        input_dim (int): Number of input features.
    """
    
    def __init__(self, input_dim):
        """
        Initialize the SVM model.
        
        Args:
            input_dim (int): Number of input features.
        """
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Define a linear layer with input_dim as input features and 1 as output feature (for SVM)

    def forward(self, x):
        """
        Perform forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1), representing
                          the raw scores (decision function) from the linear layer.
        """
        return self.linear(x)  # Output the raw score (decision function) from the linear layer

class SVMLoss(nn.Module):
    """
    Hinge Loss function for SVM implemented using PyTorch.
    
    Args:
        outputs (torch.Tensor): Output tensor from the SVM model of shape (batch_size, 1).
        labels (torch.Tensor): True labels tensor of shape (batch_size,).
    """
    
    def __init__(self):
        """
        Initialize the SVMLoss function.
        """
        super(SVMLoss, self).__init__()

    def forward(self, outputs, labels):
        """
        Compute the Hinge Loss for SVM.
        
        Args:
            outputs (torch.Tensor): Output tensor from the SVM model of shape (batch_size, 1).
            labels (torch.Tensor): True labels tensor of shape (batch_size,).
        
        Returns:
            torch.Tensor: Mean Hinge Loss value over the batch.
        """
        # Hinge Loss = max(0, 1 - t * y)
        return torch.mean(torch.clamp(1 - outputs * labels, min=0))