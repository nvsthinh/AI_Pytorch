# Linear Regression

## 1. Overview
Linear Regression is a simple but powerful algorithm used for predicting a continuous target variable based on one or more input features. It assumes a linear relationship between the input variables (X) and the single output variable (y). The goal is to find the line (or hyperplane in higher dimensions) that best fits the data.

## 2. Implementation
This implementation of Linear Regression uses PyTorch to perform gradient descent and optimize the weights.

### 2.1. Files
- `linear_regression.py`: Contains the PyTorch implementation of Linear Regression.
- `notebook/Linear_Regression_with_PyTorch_on_Sample.ipynb`: A Jupyter notebook demonstrating the Linear Regression implementation with detailed explanations and visualizations.


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
cd AI_Pytorch/supervised_learning/machine_learning/linear_regression
```
Here's a basic example of how to use the Linear Regression implementation:

```python
import torch
from linear_regression import LinearRegressionModel

# Sample data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])

# Model initialization
model = LinearRegressionModel()

# Training the model (not shown in the example)

# Making predictions
model.eval()
predicted = model(X).detach().numpy()
print(predicted)
```
### 3.3. Jupyter Notebook
For a more interactive demonstration, you can open the Jupyter notebook: [Linear_Regression_with_PyTorch_on_Sample.ipynb](https://github.com/nvsthinh/AI_Pytorch/blob/main/supervised_learning/machine_learning/linear_regression/notebook/Linear_Regression_with_PyTorch_on_Sample.ipynb)

## 4. Directory Structure
The directory structure for this repository should be organized as follows:

```csharp
machine_learning/
├── linear_regression.py
├── notebook/
│   └── Linear_Regression_with_PyTorch_on_Sample.ipynb
├── requirements.txt
└── README.md
```
## 5. License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/nvsthinh/AI_Pytorch/blob/main/LICENSE) file for details.


