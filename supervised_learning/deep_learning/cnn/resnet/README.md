# ResNet

## 1. Overview
ResNet (Residual Network) is a convolutional neural network architecture that introduced the concept of residual learning. Developed by Kaiming He et al., ResNet won the ImageNet Large Scale Visual Recognition Challenge in 2015. The key innovation of ResNet is the use of skip connections, or shortcuts, that allow the gradient to bypass one or more layers. This helps in training very deep networks by mitigating the vanishing gradient problem. There are several variations of ResNet, including ResNet18, ResNet34, ResNet50, and ResNet101, which differ in the number of layers.

## 2. Implementation
This implementation of the ResNet models (ResNet18, ResNet34, ResNet50, and ResNet101) uses PyTorch to build and train the neural networks.

### 2.1. Files
- `resnet18.py`: Contains the PyTorch implementation of ResNet18.
- `resnet34.py`: Contains the PyTorch implementation of ResNet34.
- `resnet50.py`: Contains the PyTorch implementation of ResNet50.
- `resnet101.py`: Contains the PyTorch implementation of ResNet101.
- `notebook/ResNet18_Model_with_PyTorch_on_CIFAR10.ipynb`: A Jupyter notebook demonstrating the ResNet18 implementation with detailed explanations and visualizations. (Not yet implement)
- `notebook/ResNet34_Model_with_PyTorch_on_CIFAR10.ipynb`: A Jupyter notebook demonstrating the ResNet34 implementation with detailed explanations and visualizations. (Not yet implement)
- `notebook/ResNet50_Model_with_PyTorch_on_CIFAR10.ipynb`: A Jupyter notebook demonstrating the ResNet50 implementation with detailed explanations and visualizations. (Not yet implement)
- `notebook/ResNet101_Model_with_PyTorch_on_CIFAR10.ipynb`: A Jupyter notebook demonstrating the ResNet101 implementation with detailed explanations and visualizations. (Not yet implement)

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
cd supervised_learning/deep_learning/cnn/resnet
```
Here's a basic example of how to use the GoogleNet implementation:
```python
import torch
import torch.nn.functional as F
from resnet18 import ResNet18
from resnet34 import ResNet34
from resnet50 import ResNet50
from resnet101 import ResNet101

# Sample data (e.g., 3x224x224 color images, similar to ImageNet dimensions)
sample_data = torch.randn(5, 3, 224, 224)  # batch of 5 images

# Model initialization for ResNet50 (similarly for ResNet18, ResNet34, and ResNet101)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50(num_classes=10).to(device)

# Making predictions
model.eval()
with torch.no_grad():
    sample_data = sample_data.to(device)
    output = model(sample_data)
print(f"Output shape: {output.shape}")
# Output: Output shape: torch.Size([5, 10])
```
## 3.3. Jupyter Notebook
For a more interactive demonstration, you can open the Jupyter notebook: 
- [ResNet18_Model_with_PyTorch_on_CIFAR10.ipynb]() (not yet)
- [ResNet34_Model_with_PyTorch_on_CIFAR10.ipynb]() (not yet)
- [ResNet50_Model_with_PyTorch_on_CIFAR10.ipynb]() (not yet)
- [ResNet101_Model_with_PyTorch_on_CIFAR10.ipynb]() (not yet)

## 4. Directory Structure
The directory structure for this repository should be organized as follows:
```csharp
resnet/
├── resnet18.py
├── resnet34.py
├── resnet50.py
├── resnet101.py
├── notebook/
│   ├── ResNet18_Model_with_PyTorch_on_CIFAR10.ipynb (not yet)
│   ├── ResNet34_Model_with_PyTorch_on_CIFAR10.ipynb (not yet)
│   ├── ResNet50_Model_with_PyTorch_on_CIFAR10.ipynb (not yet)
│   └── ResNet101_Model_with_PyTorch_on_CIFAR10.ipynb (not yet)
└── README.md
```

## 5. License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/nvsthinh/AI_Pytorch/blob/main/LICENSE) file for details.