# ZFNet

## 1. Overview
ZFNet, named after its creators Matthew Zeiler and Rob Fergus, is an extension of AlexNet that was designed to improve upon the performance of its predecessor. ZFNet made several modifications to the AlexNet architecture, including the use of a smaller convolutional stride and a larger number of filters in the first convolutional layer. These changes enhanced the model's ability to learn more complex features and led to better performance on image classification tasks.

## 2. Implementation
This implementation of ZFNet uses PyTorch to build and train the neural network.

### 2.1. Files
- `zfnet.py`: Contains the PyTorch implementation of ZFNet.
- `notebook/ZFNet_Model_with_PyTorch_on_CIFAR10.ipynb`: A Jupyter notebook demonstrating the ZFNet implementation with detailed explanations and visualizations.

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
cd supervised_learning/deep_learning/cnn/zfnet
```
Here's a basic example of how to use the ZFNet implementation:
```python
import torch
import torch.nn.functional as F
from zfnet import ZFNet

# Sample data (e.g., 3x224x224 color images, similar to ImageNet dimensions)
sample_data = torch.randn(5, 3, 224, 224)  # batch of 5 images

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ZFNet(num_classes=10).to(device)

# Making predictions
model.eval()
with torch.no_grad():
    sample_data = sample_data.to(device)
    output = model(sample_data)
print(f"Output shape: {output.shape}")
# Output: Output shape: torch.Size([5, 10])
```
## 3.3. Jupyter Notebook
For a more interactive demonstration, you can open the Jupyter notebook: [ZFNet_Model_with_PyTorch_on_CIFAR10.ipynb](https://github.com/nvsthinh/AI_Pytorch/blob/main/supervised_learning/deep_learning/cnn/zfnet/notebook/ZFNet_Model_with_PyTorch_on_CIFAR10.ipynb)

## 4. Directory Structure
The directory structure for this repository should be organized as follows:
```csharp
zfnet/
├── zfnet.py
├── notebook/
│   └── ZFNet_Model_with_PyTorch_on_CIFAR10.ipynb
└── README.md
```

## 5. License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/nvsthinh/AI_Pytorch/blob/main/LICENSE) file for details.