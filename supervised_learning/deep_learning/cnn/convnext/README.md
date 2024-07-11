# ConvNeXT

## 1. Overview
ConvNeXT is a modern convolutional neural network architecture inspired by the design principles of the Vision Transformer (ViT). Developed by Facebook AI Research (FAIR), ConvNeXT aims to combine the benefits of CNNs and Transformers, achieving state-of-the-art performance on various computer vision tasks. The ConvNeXT models are designed with simplicity and efficiency in mind, using standard convolutional operations while incorporating architectural innovations from Transformers.
- __ConvNeXT_tiny__: The smallest model in the ConvNeXT family, designed for efficient inference on devices with limited computational resources while maintaining strong performance. It consists of 28M parameters and achieves high accuracy with minimal computational overhead.
- __ConvNeXT_small__: Slightly larger than ConvNeXT_tiny, this model offers a better trade-off between performance and computational requirements, with 50M parameters. It is suitable for applications where slightly more computational power is available.
- __ConvNeXT_base__: The standard model in the ConvNeXT family, providing a good balance of accuracy and efficiency for most applications. It has 89M parameters and is designed for general-purpose use cases.
- __ConvNeXT_large__: A larger model designed for applications requiring higher accuracy and capable of leveraging more computational power, with 197M parameters. It performs exceptionally well on complex vision tasks.
- __ConvNeXT_xlarge__: The largest model in the ConvNeXT family, offering the highest accuracy with 350M parameters. It is intended for applications where computational resources are less of a concern and the highest performance is required.
## 2. Implementation
This implementation of the ResNet models (ResNet18, ResNet34, ResNet50, and ResNet101) uses PyTorch to build and train the neural networks.

### 2.1. Files
- `convnext_tiny.py`: Contains the PyTorch implementation of ConvNeXT_tiny.
- `convnext_small.py`: Contains the PyTorch implementation of ConvNeXT_small.
- `convnext_base.py`: Contains the PyTorch implementation of ConvNeXT_base.
- `convnext_large.py`: Contains the PyTorch implementation of ConvNeXT_large.
- `convnext_xlarge.py`: Contains the PyTorch implementation of ConvNeXT_xlarge.
- `notebook/ConvNeXT_tiny_Model_with_PyTorch_on_CIFAR10.ipynb`: A Jupyter notebook demonstrating the ConvNeXT_tiny implementation with detailed explanations and visualizations.
- `notebook/ConvNeXT_small_Model_with_PyTorch_on_CIFAR10.ipynb`: A Jupyter notebook demonstrating the ConvNeXT_small implementation with detailed explanations and visualizations.
- `notebook/ConvNeXT_base_Model_with_PyTorch_on_CIFAR10.ipynb`: A Jupyter notebook demonstrating the ConvNeXT_base implementation with detailed explanations and visualizations.
- `notebook/ConvNeXT_large_Model_with_PyTorch_on_CIFAR10.ipynb`: A Jupyter notebook demonstrating the ConvNeXT_large implementation with detailed explanations and visualizations.
- `notebook/ConvNeXT_xlarge_Model_with_PyTorch_on_CIFAR10.ipynb`: A Jupyter notebook demonstrating the ConvNeXT_xlarge implementation with detailed explanations and visualizations.

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
cd supervised_learning/deep_learning/cnn/convnext
```
Here's a basic example of how to use the ConvNeXT implementations:
```python
import torch
import torch.nn.functional as F
from convnext_tiny import ConvNeXT_tiny
from convnext_small import ConvNeXT_small
from convnext_base import ConvNeXT_base
from convnext_large import ConvNeXT_large
from convnext_xlarge import ConvNeXT_xlarge

# Sample data (e.g., 3x224x224 color images, similar to ImageNet dimensions)
sample_data = torch.randn(5, 3, 224, 224)  # batch of 5 images

# Model initialization for ConvNeXT_base (similarly for other variants)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNeXT_base(num_classes=10).to(device)

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
- [ConvNeXT_tiny_Model_with_PyTorch_on_CIFAR10.ipynb]() (not yet)
- [ConvNeXT_small_Model_with_PyTorch_on_CIFAR10.ipynb]() (not yet)
- [ConvNeXT_base_Model_with_PyTorch_on_CIFAR10.ipynb]() (not yet)
- [ConvNeXT_large_Model_with_PyTorch_on_CIFAR10.ipynb]() (not yet)
- [ConvNeXT_xlarge_Model_with_PyTorch_on_CIFAR10.ipynb]() (not yet)

## 4. Directory Structure
The directory structure for this repository should be organized as follows:
```csharp
convnext/
├── convnext_tiny.py
├── convnext_small.py
├── convnext_base.py
├── convnext_large.py
├── convnext_xlarge.py
├── notebook/
│   ├── ConvNeXT_tiny_Model_with_PyTorch_on_CIFAR10.ipynb (not yet)
│   ├── ConvNeXT_small_Model_with_PyTorch_on_CIFAR10.ipynb (not yet)
│   ├── ConvNeXT_base_Model_with_PyTorch_on_CIFAR10.ipynb (not yet)
│   ├── ConvNeXT_large_Model_with_PyTorch_on_CIFAR10.ipynb (not yet)
│   └── ConvNeXT_xlarge_Model_with_PyTorch_on_CIFAR10.ipynb (not yet)
└── README.md
```

## 5. License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/nvsthinh/AI_Pytorch/blob/main/LICENSE) file for details.