# Logistic Regression

## 1. Overview
Logistic Regression is a widely-used algorithm for binary classification problems. Unlike linear regression, which predicts continuous outcomes, logistic regression predicts the probability of a binary outcome, using the logistic function to map predicted values to probabilities.

## 2. Implementation
This implementation of Logistic Regression uses PyTorch to perform gradient descent and optimize the weights.

### 2.1. Files
- `logistic_regression.py`: Contains the PyTorch implementation of Logistic Regression.
- `notebook/Logistic_Regression_Model_with_PyTorch_on_MNIST.ipynb`: A Jupyter notebook demonstrating the Logistic Regression implementation with detailed explanations and visualizations.

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
cd supervised_learning/machine_learning/logistic_regression
```
Here's a basic example of how to use the Logistic Regression implementation:

```python
import torch
from logistic_regression import LogisticRegressionModel

# Sample data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[0.0], [0.0], [1.0], [1.0], [1.0]])

# Model initialization
model = LogisticRegressionModel(n_inputs=1, n_outputs=1)

# Training the model (not shown in the example)

# Making predictions
model.eval()
predicted = model(X).detach().numpy()
print(predicted)
```
### 3.3. Jupyter Notebook
For a more interactive demonstration, you can open the Jupyter notebook: [Logistic_Regression_Model_with_PyTorch_on_MNIST.ipynb](https://github.com/nvsthinh/AI_Pytorch/blob/main/supervised_learning/machine_learning/logistic_regression/notebook/Logistic_Regression_Model_with_PyTorch_on_MNIST.ipynb)

## 4. Directory Structure
The directory structure for this repository should be organized as follows:

```csharp
logistic_regression/
├── logistic_regression.py
├── notebook/
│   └── Logistic_Regression_Model_with_PyTorch_on_MNIST.ipynb
├── requirements.txt
└── README.md
```
## 5. License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/nvsthinh/AI_Pytorch/blob/main/LICENSE) file for details.