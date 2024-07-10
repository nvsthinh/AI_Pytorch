# Support Vector Machine (SVM)

## 1. Overview
Support Vector Machines (SVM) are powerful, supervised learning models used for classification and regression tasks. They work by finding the hyperplane that best separates the classes in the feature space, maximizing the margin between the data points of different classes. SVMs are effective in high-dimensional spaces and are versatile due to different kernel functions.

## 2. Implementation
This implementation of SVM uses PyTorch to perform the necessary computations and optimizations.

### 2.1. Files
- `svm.py`: Contains the PyTorch implementation of Support Vector Machine.
- `notebook/SVM_Model_with_PyTorch_on_MNIST.ipynb`: A Jupyter notebook demonstrating the SVM implementation with detailed explanations and visualizations.

## 3. How to Use

### 3.1. Running the Code
#### Step 1: Clone this repository
```bash
git clone https://github.com/nvsthinh/AI_Pytorch.git
cd AI_Pytorch
```

#### Step 2: Installation Requirements
This project relies on several key Python libraries. To ensure you have the correct versions, use the provided `requirements.txt` file to install the necessary dependencies.
```bash
pip install -r requirements.txt
```

### 3.2. Example
Navigate to the relevant directory:
```bash
cd supervised_learning/machine_learning/svm
```

Here's a basic example of how to use the SVM implementation:
```python
import torch
from svm import SVM, SVMLoss
# Sample data
X = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [2.0, 1.0], [3.0, 2.0]])
y = torch.tensor([1, 1, 1, -1, -1], dtype=torch.float32)

# Model initialization
model = SVM(input_dim=2)

# Training the model
learning_rate = 0.01
num_epochs = 1000
criterion = SVMLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Making predictions
model.eval()
with torch.no_grad():
    predicted = model(X).sign()
print(predicted)
```

### 3.3. Jupyter Notebook
For a more interactive demonstration, you can open the Jupyter notebook: [SVM_Model_with_PyTorch_on_MNIST.ipynb](https://github.com/nvsthinh/AI_Pytorch/blob/main/supervised_learning/machine_learning/svm/notebook/SVM_Model_with_PyTorch_on_MNIST.ipynb)

## 4. Directory Structure
The directory structure for this repository is organized as follows:
```csharp
svm/
├── svm.py
├── notebook/
│   └── SVM_with_PyTorch_on_Sample.ipynb
├── requirements.txt
└── README.md
```
## 5. License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/nvsthinh/AI_Pytorch/blob/main/LICENSE) file for details.
