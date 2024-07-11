# AlexNet

## 1. Overview
AlexNet is a convolutional neural network architecture that won the ImageNet Large Scale Visual Recognition Challenge in 2012. Designed by Alex Krizhevsky et al., AlexNet introduced the use of ReLU activation, dropout, and data augmentation, significantly improving the performance of deep learning models on image classification tasks.

## 2. Implementation
This implementation of AlexNet uses PyTorch to build and train the neural network.

### 2.1. Files
- `lenet.py`: Contains the PyTorch implementation of LeNet.
- `notebook/LeNet_with_PyTorch_on_MNIST.ipynb`: A Jupyter notebook demonstrating the LeNet implementation with detailed explanations and visualizations.
- `alexnet.py`: Contains the PyTorch implementation of AlexNet.
- `notebook/AlexNet_with_PyTorch_on_MNIST.ipynb`: A Jupyter notebook demonstrating the AlexNet implementation with detailed explanations and visualizations.

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
cd supervised_learning/deep_learning/alexnet
```
Here's a basic example of how to use the Linear Regression implementation:
```python
import torch
import torch.nn.functional as F
from alexnet import AlexNet

# Sample data (e.g., 3x224x224 color images, similar to ImageNet dimensions)
sample_data = torch.randn(5, 3, 224, 224)  # batch of 5 images

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet().to(device)

# Making predictions
model.eval()
with torch.no_grad():
    sample_data = sample_data.to(device)
    output = model(sample_data)
    pred = output.argmax(dim=1, keepdim=True)
    print(f'Predicted: {pred.view(-1).tolist()}')
```
## 3.3. Jupyter Notebook
For a more interactive demonstration, you can open the Jupyter notebook: [AlexNet_Model_with_PyTorch_on_CIFAR10.ipynb](https://github.com/nvsthinh/AI_Pytorch/blob/main/supervised_learning/deep_learning/cnn/alexnet/notebook/AlexNet_Model_with_PyTorch_on_CIFAR10.ipynb)

## 4. Directory Structure
The directory structure for this repository should be organized as follows:
```csharp
alexnet/
├── alexnet.py
├── notebook/
│   └── AlexNet_Model_with_PyTorch_on_CIFAR10.ipynb
└── README.md
```

## 5. License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/nvsthinh/AI_Pytorch/blob/main/LICENSE) file for details.