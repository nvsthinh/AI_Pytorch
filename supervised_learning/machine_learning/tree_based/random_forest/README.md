# Random Forest

## 1. Overview
A Random Forest is an ensemble learning algorithm used for classification and regression tasks. It operates by constructing a multitude of decision trees during training and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

## 2. Implementation
This implementation of the Random Forest model is built from scratch using PyTorch.

### 2.1. Files
- `random_forest.py`: Contains the PyTorch implementation of the Random Forest, including the Decision Tree class.
- `notebook/Random_Forest_Model_with_PyTorch_on_MNIST.ipynb`: A Jupyter notebook demonstrating the Random Forest implementation with detailed explanations and visualizations.

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
cd supervised_learning/machine_learning/tree_based/random_forest
```
Here's a basic example of how to use the Random Forest implementation:
```python
import torch
from random_forest import RandomForest

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
model = RandomForest(n_estimators=10, max_depth=3, min_samples_split=2)
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
For a more interactive demonstration, you can open the Jupyter notebook: [Random_Forest_Model_with_PyTorch_on_MNIST.ipynb](https://github.com/nvsthinh/AI_Pytorch/blob/main/supervised_learning/machine_learning/tree_based/random_forest/notebook/Random_Forest_Model_with_PyTorch_on_MNIST.ipynb)

## 4. Directory Structure
The directory structure for this repository should be organized as follows:
```csharp
random_forest/
├── random_forest.py
├── notebook/
│   └── Random_Forest_Model_with_PyTorch_on_MNIST.ipynb
└── README.md
```

## 5. License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/nvsthinh/AI_Pytorch/blob/main/LICENSE) file for details.