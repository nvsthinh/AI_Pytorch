# Decision Tree

## 1. Overview
A Decision Tree is a supervised learning algorithm used for both classification and regression tasks. It partitions the data into subsets based on the feature values and predicts the target value at each node of the tree.

## 2. Implementation
This implementation of the Decision Tree model is built from scratch using PyTorch.

### 2.1. Files
- `decision_tree.py`: Contains the PyTorch implementation of the Decision Tree.
- `notebook/Decision_Tree_Model_with_PyTorch_on_MNIST.ipynb`: A Jupyter notebook demonstrating the Decision Tree implementation with detailed explanations and visualizations.

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
cd supervised_learning/machine_learning/tree_based/decision_tree
```
Here's a basic example of how to use the Linear Regression implementation:
```python
import torch
from decision_tree import DecisionTree

# Sample data (example with 10 samples and 4 features)
X_train = torch.tensor([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2],
    [5.0, 3.6, 1.4, 0.2],
    [7.0, 3.2, 4.7, 1.4],
    [6.4, 3.2, 4.5, 1.5],
    [6.9, 3.1, 4.9, 1.5],
    [5.5, 2.3, 4.0, 1.3],
    [6.5, 2.8, 4.6, 1.5]
], dtype=torch.float32)

y_train = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # Example labels (0 or 1)

# Model initialization
model = DecisionTree(depth=3, min_samples_split=2)
model.fit(X_train, y_train)

# Making predictions
sample_data = torch.tensor([
    [5.1, 3.5, 1.4, 0.2],   # Predicted as class 0
    [6.5, 2.8, 4.6, 1.5],   # Predicted as class 1
    [4.8, 3.0, 1.4, 0.1]    # Predicted as class 0
], dtype=torch.float32)

predictions = model.predict(sample_data)
print(f'Predictions: {predictions}')  # Output: Predictions: [0, 1, 0]
```
## 3.3. Jupyter Notebook
For a more interactive demonstration, you can open the Jupyter notebook: [Decision_Tree_Model_with_PyTorch_on_MNIST.ipynb](https://github.com/nvsthinh/AI_Pytorch/blob/main/supervised_learning/machine_learning/tree_based/decision_tree/notebook/Decision_Tree_Model_with_PyTorch_on_MNIST.ipynb)

## 4. Directory Structure
The directory structure for this repository should be organized as follows:
```csharp
decision_tree/
├── decision_tree.py
├── notebook/
│   └── Decision_Tree_Model_with_PyTorch_on_MNIST.ipynb
└── README.md
```

## 5. License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/nvsthinh/AI_Pytorch/blob/main/LICENSE) file for details.