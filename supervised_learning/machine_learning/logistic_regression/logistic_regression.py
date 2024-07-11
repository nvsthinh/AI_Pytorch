import torch
import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    """
    Logistic Regression model implemented using PyTorch.
    
    Attributes:
        linear (torch.nn.Linear): Linear transformation layer.
    
    Args:
        n_inputs (int): Number of input features.
        n_outputs (int): Number of output features, typically 1 for binary classification.
    """
    
    def __init__(self, n_inputs, n_outputs):
        """
        Initialize the Logistic Regression model.
        
        Args:
            n_inputs (int): Number of input features.
            n_outputs (int): Number of output features, typically 1 for binary classification.
        """
        super().__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)  # Define a linear layer with n_inputs and n_outputs
    
    def forward(self, x):
        """
        Perform forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_inputs).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_outputs), after applying
                          sigmoid activation to the linear transformation.
        """
        output = torch.sigmoid(self.linear(x))  # Apply sigmoid activation to the linear transformation
        return output
