# MLP (Multi-Layer Perceptron)

## 1. Overview
A Multi-Layer Perceptron (MLP) is a class of feedforward artificial neural networks. An MLP consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer. Each node, except for the input nodes, is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training.

## 2. Implementation
This implementation of AlexNet uses PyTorch to build and train the neural network.

### 2.1. Files
- `mlp.py`: Contains the PyTorch implementation of MLP.
- `notebook/MLP_Model_with_Pytorch_on_FashionMNIST.ipynb`: A Jupyter notebook demonstrating the MLP implementation with detailed explanations and visualizations.

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
cd supervised_learning/deep_learning/mlp
```
Here's a basic example of how to use the Linear Regression implementation:
```python
import torch
import torch.nn.functional as F
from mlp import MLPModel

# Sample data (e.g., 28x28 grayscale images, similar to MNIST dimensions)
sample_data = torch.randn(5, 784)  # batch of 5 flattened images

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPModel(n_inputs=784, n_outputs=10).to(device)

# Making predictions
model.eval()
with torch.no_grad():
    sample_data = sample_data.to(device)
    output = model(sample_data)
    pred = output.argmax(dim=1, keepdim=True)
    print(f'Predicted: {pred.view(-1).tolist()}')
```
## 3.3. Jupyter Notebook
For a more interactive demonstration, you can open the Jupyter notebook: [MLP_Model_with_Pytorch_on_FashionMNIST.ipynb](https://github.com/nvsthinh/AI_Pytorch/tree/main/supervised_learning/deep_learning/mlp/notebook/MLP_Model_with_Pytorch_on_FashionMNIST.ipynb)

## 4. Directory Structure
The directory structure for this repository should be organized as follows:
```csharp
mlp/
├── mlp.py
├── notebook/
│   └── MLP_Model_with_Pytorch_on_FashionMNIST.ipynb
└── README.md
```

## 5. License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/nvsthinh/AI_Pytorch/blob/main/LICENSE) file for details.