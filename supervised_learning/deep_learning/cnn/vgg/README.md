# VGG

## 1. Overview
The VGG (Visual Geometry Group) network is a convolutional neural network architecture that achieved excellent performance on the ImageNet dataset. Developed by the Visual Geometry Group at the University of Oxford, the VGG network is known for its simplicity and depth, using very small (3x3) convolution filters throughout the network. There are several variations of the VGG network, including VGG11, VGG16, and VGG19, which differ in the number of layers.

## 2. Implementation
This implementation of the VGG models (VGG11, VGG16, and VGG19) uses PyTorch to build and train the neural networks.

### 2.1. Files
- `vgg11.py`: Contains the PyTorch implementation of VGG11.
- `vgg16.py`: Contains the PyTorch implementation of VGG16.
- `vgg19.py`: Contains the PyTorch implementation of VGG19.
- `notebook/VGG11_Model_with_PyTorch_on_CIFAR10.ipynb`: A Jupyter notebook demonstrating the VGG11 implementation with detailed explanations and visualizations.
- `notebook/VGG16_Model_with_PyTorch_on_CIFAR10.ipynb`: A Jupyter notebook demonstrating the VGG16 implementation with detailed explanations and visualizations.
- `notebook/VGG19_Model_with_PyTorch_on_CIFAR10.ipynb`: A Jupyter notebook demonstrating the VGG19 implementation with detailed explanations and visualizations.

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
cd supervised_learning/deep_learning/cnn/vgg
```
Here's a basic example of how to use the VGG implementation:
```python
import torch
import torch.nn.functional as F
from vgg11 import VGG11
from vgg16 import VGG16
from vgg19 import VGG19

# Sample data (e.g., 3x224x224 color images, similar to ImageNet dimensions)
sample_data = torch.randn(5, 3, 224, 224)  # batch of 5 images

# Model initialization for VGG16 (similarly for VGG11 and VGG19)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16(num_classes=10).to(device)

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
- [VGG11_Model_with_PyTorch_on_CIFAR10.ipynb](https://github.com/nvsthinh/AI_Pytorch/blob/main/supervised_learning/deep_learning/cnn/vgg/notebook/VGG11_Model_with_PyTorch_on_CIFAR10.ipynb)
- [VGG16_Model_with_PyTorch_on_CIFAR10.ipynb](https://github.com/nvsthinh/AI_Pytorch/blob/main/supervised_learning/deep_learning/cnn/vgg/notebook/VGG16_Model_with_PyTorch_on_CIFAR10.ipynb)
- [VGG19_Model_with_PyTorch_on_CIFAR10.ipynb](https://github.com/nvsthinh/AI_Pytorch/blob/main/supervised_learning/deep_learning/cnn/vgg/notebook/VGG19_Model_with_PyTorch_on_CIFAR10.ipynb)

## 4. Directory Structure
The directory structure for this repository should be organized as follows:
```csharp
vgg/
├── vgg11.py
├── vgg16.py
├── vgg19.py
├── notebook/
│   ├── VGG11_Model_with_PyTorch_on_CIFAR10.ipynb
│   ├── VGG16_Model_with_PyTorch_on_CIFAR10.ipynb
│   └── VGG19_Model_with_PyTorch_on_CIFAR10.ipynb
└── README.md
```

## 5. License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/nvsthinh/AI_Pytorch/blob/main/LICENSE) file for details.