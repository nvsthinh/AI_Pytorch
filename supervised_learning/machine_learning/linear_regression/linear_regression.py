import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    """
    Logistic Regression model implemented using PyTorch.
    
    Attributes:
        linear (torch.nn.Linear): Linear transformation layer.
    """
    
    def __init__(self):
        """
        Initialize the Logistic Regression model.
        
        Defines a linear layer with 1 input feature and 1 output feature.
        """
        super().__init__()
        self.linear = nn.Linear(1, 1)  # Define a linear layer with 1 input and 1 output feature
    
    def forward(self, x):
        """
        Perform forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1), after performing
                          the linear transformation.
        """
        output = self.linear(x)  # Perform the linear transformation
        return output