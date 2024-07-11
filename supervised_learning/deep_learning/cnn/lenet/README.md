# LeNet

## 1. Overview
LeNet is one of the earliest convolutional neural networks, designed by Yann LeCun et al. It was primarily used for handwritten digit recognition (MNIST). LeNet-5, a popular variant, consists of two sets of convolutional and subsampling layers followed by fully connected layers.

## 2. Implementation
This implementation of LeNet uses PyTorch to build and train the neural network.

### 2.1. Files
- `lenet.py`: Contains the PyTorch implementation of LeNet.
- `notebook/LeNet_with_PyTorch_on_MNIST.ipynb`: A Jupyter notebook demonstrating the LeNet implementation with detailed explanations and visualizations.

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
cd supervised_learning/deep_learning/lenet
```
Here's a basic example of how to use the Linear Regression implementation:
```python
import torch
import torch.nn.functional as F
from lenet import LeNet

# Sample data (e.g., 1x28x28 grayscale images, similar to MNIST dimensions)
sample_data = torch.randn(5, 1, 28, 28)  # batch of 5 images

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)

# Making predictions
model.eval()
with torch.no_grad():
    sample_data = sample_data.to(device)
    output = model(sample_data)
    pred = output.argmax(dim=1, keepdim=True)
    print(f'Predicted: {pred.view(-1).tolist()}')
```
## 3.3. Jupyter Notebook
For a more interactive demonstration, you can open the Jupyter notebook: [LeNet_Model_with_PyTorch_on_CIFAR10.ipynb](https://github.com/nvsthinh/AI_Pytorch/blob/main/supervised_learning/deep_learning/cnn/lenet/notebook/LeNet_Model_with_PyTorch_on_CIFAR10.ipynb)

## 4. Directory Structure
The directory structure for this repository should be organized as follows:
```csharp
lenet/
├── lenet.py
├── notebook/
│   └── LeNet_with_PyTorch_on_MNIST.ipynb
└── README.md
```

## 5. License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/nvsthinh/AI_Pytorch/blob/main/LICENSE) file for details.