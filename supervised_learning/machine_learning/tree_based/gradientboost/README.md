# Gradient Boost

## 1. Overview
Gradient Boosting is a powerful ensemble learning technique used for classification and regression tasks. It builds models sequentially, where each new model attempts to correct the errors made by the previous models. This approach combines multiple weak learners to create a strong learner. The most common type of weak learner used in Gradient Boosting is a decision tree.

## 2. Implementation
This implementation of the Gradient Boosting model is built from scratch using NumPy.

### 2.1. Files
- `gradient_boost.py`: Contains the NumPy implementation of the Gradient Boost model, including the Decision Tree class.
- `notebook/GradientBoost_Model_with_Numpy_on_Advertising.ipynb`: A Jupyter notebook demonstrating the Gradient Boost implementation with detailed explanations and visualizations.

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
cd supervised_learning/machine_learning/tree_based/adaboost
```
Here's a basic example of how to use the Decision Tree implementation:
```python
import numpy as np
from gradient_boost import GradientBoost

# Sample data (example with 10 samples and 4 features)
X_train = np.array([
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
])

y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # Example labels (0 or 1)

# Model initialization
model = GradientBoost(n_estimators=10, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

# Making predictions
sample_data = np.array([
    [5.1, 3.5, 1.4, 0.2],   # Predicted as class 0
    [6.5, 2.8, 4.6, 1.5],   # Predicted as class 1
    [4.8, 3.0, 1.4, 0.1]    # Predicted as class 0
])

predictions = model.predict(sample_data)
print(f'Predictions: {predictions}')  # Output: Predictions: [0, 1, 0]
```
## 3.3. Jupyter Notebook
For a more interactive demonstration, you can open the Jupyter notebook: [GradientBoost_Model_with_Numpy_on_Advertising.ipynb](https://github.com/nvsthinh/AI_Pytorch/blob/main/supervised_learning/machine_learning/tree_based/gradientboost/notebook/GradientBoost_Model_with_Numpy_on_Advertising.ipynb)

## 4. Directory Structure
The directory structure for this repository should be organized as follows:
```csharp
gradient_boost/
├── gradient_boost.py
├── notebook/
│   └── GradientBoost_Model_with_Numpy_on_Advertising.ipynb
└── README.md
```

## 5. License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/nvsthinh/AI_Pytorch/blob/main/LICENSE) file for details.