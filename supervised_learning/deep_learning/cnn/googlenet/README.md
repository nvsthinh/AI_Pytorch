# GoogleNet

## 1. Overview
GoogleNet, also known as Inception V1, is a convolutional neural network architecture that won the ImageNet Large Scale Visual Recognition Challenge in 2014. Designed by researchers at Google, GoogleNet introduced the concept of the Inception module, which allows for more efficient computation by using different filter sizes within the same module. This architecture significantly reduced the number of parameters compared to previous networks, improving both computational efficiency and performance.

## 2. Implementation
This implementation of GoogleNet uses PyTorch to build and train the neural network.

### 2.1. Files
- `googlenet.py`: Contains the PyTorch implementation of GoogleNet.
- `notebook/GoogleNet_Model_with_PyTorch_on_CIFAR10.ipynb`: A Jupyter notebook demonstrating the GoogleNet implementation with detailed explanations and visualizations.

## 3. How to Use
### 3.1. Running the Code
#### Step 1. Clone this repository
```bash
git clone https://github.com/nvsthinh/AI_Pytorch.git
cd AI_Pytorch
```
#### Step 2. Installation Requirements
This project relies on several key Python libraries. To ensure you have the correct versions, use the provided `requirements.txt` file to install the necessary dependencies.
```bash
pip install -r requirements.txt
```

### 3.2. Example
```bash
cd supervised_learning/deep_learning/cnn/googlenet
```
Here's a basic example of how to use the GoogleNet implementation:
```python
import torch
import torch.nn.functional as F
from googlenet import GoogLeNet

# Sample data (e.g., 3x224x224 color images, similar to ImageNet dimensions)
sample_data = torch.randn(5, 3, 224, 224)  # batch of 5 images

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GoogLeNet(num_classes=10).to(device)

# Making predictions
model.eval()
with torch.no_grad():
    sample_data = sample_data.to(device)
    output = model(sample_data)
print(f"Output shape: {output.shape}")
# Output: Output shape: torch.Size([5, 10])
```
## 3.3. Jupyter Notebook
For a more interactive demonstration, you can open the Jupyter notebook: [GoogleNet_Model_with_PyTorch_on_CIFAR10.ipynb](https://github.com/nvsthinh/AI_Pytorch/blob/main/supervised_learning/deep_learning/cnn/googlenet/notebook/GoogLeNet_Model_with_PyTorch_on_CIFAR10.ipynb)

## 4. Directory Structure
The directory structure for this repository should be organized as follows:
```csharp
googlenet/
├── googlenet.py
├── notebook/
│   └── GoogleNet_Model_with_PyTorch_on_CIFAR10.ipynb
└── README.md
```

## 5. License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/nvsthinh/AI_Pytorch/blob/main/LICENSE) file for details.