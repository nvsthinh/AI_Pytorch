import torch.nn as nn

class SoftmaxRegressionModel(nn.Module):
    """
    Softmax Regression model implemented using PyTorch.
    
    Attributes:
        linear (torch.nn.Linear): Linear transformation layer.
    
    Args:
        n_inputs (int): Number of input features.
        n_outputs (int): Number of output classes.
    """
    
    def __init__(self, n_inputs, n_outputs):
        """
        Initialize the Softmax Regression model.
        
        Args:
            n_inputs (int): Number of input features.
            n_outputs (int): Number of output classes.
        """
        super().__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)  # Define a linear layer with n_inputs and n_outputs
    
    def forward(self, x):
        """
        Perform forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_inputs).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_outputs), containing
                          the logits (raw scores) for each class.
        """
        output = self.linear(x)  # Compute logits (raw scores) using the linear transformation
        return output
