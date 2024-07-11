# K Nearest Neighbors (KNN)

## 1. Overview
K Nearest Neighbors (KNN) is a simple, instance-based learning algorithm used for both classification and regression tasks. It works by finding the 'k' training samples that are closest to the input data point and making predictions based on these nearest neighbors.

## 2. Implementation
This implementation of KNN uses PyTorch to find the nearest neighbors and make predictions.

### 2.1. Files
- `knn.py`: Contains the PyTorch implementation of K Nearest Neighbors.
- `notebook/K_Nearest_Neighbors_with_PyTorch_on_MNIST.ipynb`: A Jupyter notebook demonstrating the KNN implementation with detailed explanations and visualizations.

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
cd supervised_learning/machine_learning/knn
```
Here's a basic example of how to use the K Nearest Neighbors implementation:
```python
import torch
from knn import KNN

# Sample data
X_train = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_train = torch.tensor([0, 0, 1, 1, 1])
X_test = torch.tensor([[1.5], [3.5]])

# Model initialization
k = 3
knn_model = KNNClassifier(k)

# Making predictions
predictions = knn_model.predict(X_train, y_train, X_test)
print(predictions)
```
### 3.3. Jupyter Notebook
For a more interactive demonstration, you can open the Jupyter notebook: [K_Nearest_Neighbors_with_PyTorch_on_MNIST.ipynb](https://github.com/nvsthinh/AI_Pytorch/blob/main/supervised_learning/machine_learning/knn/notebook/K_Nearest_Neighbors_with_PyTorch_on_MNIST.ipynb)

## 4. Directory Structure
The directory structure for this repository should be organized as follows:
```csharp
knn/
├── knn.py
├── notebook/
│   └── K_Nearest_Neighbors_with_PyTorch_on_MNIST.ipynb
└── README.md
```

## 5. License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/nvsthinh/AI_Pytorch/blob/main/LICENSE) file for details.