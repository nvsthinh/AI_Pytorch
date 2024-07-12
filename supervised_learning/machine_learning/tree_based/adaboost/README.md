# AdaBoost

## 1. Overview
AdaBoost, short for Adaptive Boosting, is an ensemble learning method that combines the predictions of multiple weak classifiers to produce a strong classifier. It is primarily used for classification tasks. The core idea of AdaBoost is to iteratively train weak classifiers on the data, each time adjusting the weights of the training samples to focus more on those that were misclassified by previous classifiers. The final prediction is made by a weighted majority vote of the weak classifiers.

## 2. Implementation
This implementation of the AdaBoost model is built from scratch using Numpy.

### 2.1. Files
- `adaboost.py`: Contains the Numpy implementation of the AdaBoost model, including the weak classifier class.
- `notebook/AdaBoost_Model_with_Numpy_on_Spam.ipynb`: A Jupyter notebook demonstrating the AdaBoost implementation with detailed explanations and visualizations.

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
from adaboost import AdaBoostClassifier

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
], dtype=np.float32)

y_train = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1], dtype=np.float32)  # Example labels (1 or -1)

# Model initialization
model = AdaBoostClassifier()
model.fit(X_train, y_train, M=10)

# Making predictions
sample_data = np.array([
    [5.1, 3.5, 1.4, 0.2],   # Predicted as class 1
    [6.5, 2.8, 4.6, 1.5],   # Predicted as class -1
    [4.8, 3.0, 1.4, 0.1]    # Predicted as class 1
], dtype=np.float32)

predictions = model.predict(sample_data)
print(f'Predictions: {predictions}')  # Output: Predictions: [ 1. -1.  1.]
```
## 3.3. Jupyter Notebook
For a more interactive demonstration, you can open the Jupyter notebook: [AdaBoost_Model_with_Numpy_on_Spam.ipynb](https://github.com/nvsthinh/AI_Pytorch/blob/main/supervised_learning/machine_learning/tree_based/adaboost/notebook/AdaBoost_Model_with_Numpy_on_Spam.ipynb)

## 4. Directory Structure
The directory structure for this repository should be organized as follows:
```csharp
adaboost/
├── adaboost.py
├── notebook/
│   └── AdaBoost_Model_with_Numpy_on_Spam.ipynb
└── README.md
```

## 5. License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/nvsthinh/AI_Pytorch/blob/main/LICENSE) file for details.