# MobileNet

## 1. Overview
MobileNet is a family of convolutional neural network architectures designed for efficient performance on mobile and embedded vision applications. MobileNetV1 introduces depthwise separable convolutions, which drastically reduce the number of parameters and computation compared to standard convolutions. MobileNetV2 builds on this with inverted residuals and linear bottlenecks, further improving performance and efficiency.

## 2. Implementation
This implementation of the MobileNet models (MobileNetV1 and MobileNetV2) uses PyTorch to build and train the neural networks.

### 2.1. Files
- `mobilenetv1.py`: Contains the PyTorch implementation of MobileNetV1.
- `mobilenetv2.py`: Contains the PyTorch implementation of MobileNetV2.
- `notebook/MobileNetV1_Model_with_PyTorch_on_CIFAR10.ipynb`: A Jupyter notebook demonstrating the MobileNetV1 implementation with detailed explanations and visualizations. (Not yet implement)
- `notebook/MobileNetV2_Model_with_PyTorch_on_CIFAR10.ipynb`: A Jupyter notebook demonstrating the MobileNetV2 implementation with detailed explanations and visualizations.(Not yet implement)

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
cd supervised_learning/deep_learning/cnn/mobilenet
```
Here's a basic example of how to use the MobileNet implementation:
```python
import torch
import torch.nn.functional as F
from mobilenetv1 import MobileNetV1
from mobilenetv2 import MobileNetV2

# Sample data (e.g., 3x224x224 color images, similar to ImageNet dimensions)
sample_data = torch.randn(5, 3, 224, 224)  # batch of 5 images

# Model initialization for MobileNetV2 (similarly for MobileNetV1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNetV2(num_classes=10).to(device)

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
- [MobileNetv1_Model_with_PyTorch_on_CIFAR10.ipynb]() (not yet)
- [MobileNetv2_Model_with_PyTorch_on_CIFAR10.ipynb]() (not yet)

## 4. Directory Structure
The directory structure for this repository should be organized as follows:
```csharp
mobilenet/
├── mobilenetv1.py
├── mobilenetv2.py
├── notebook/
│   ├── MobileNetV1_Model_with_PyTorch_on_CIFAR10.ipynb
│   └── MobileNetV2_Model_with_PyTorch_on_CIFAR10.ipynb
└── README.md
```

## 5. License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/nvsthinh/AI_Pytorch/blob/main/LICENSE) file for details.