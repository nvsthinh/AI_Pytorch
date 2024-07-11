import torch

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        """
        Initialize a node in the decision tree.

        Args:
        - feature_index (int): Index of the feature for splitting.
        - threshold (float): Threshold value for splitting the feature.
        - left (Node): Left child node.
        - right (Node): Right child node.
        - value (int): Leaf value (predicted label).

        Note:
        - If `value` is not None, the node is a leaf node; otherwise, it is an internal node.
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    """
    Decision Tree model implemented using PyTorch.

    Attributes:
    - depth (int): Maximum depth of the tree.
    - min_samples_split (int): Minimum number of samples required to split a node.
    - tree (Node): Root node of the decision tree.

    Methods:
    - fit(X, y): Build the decision tree recursively using input features X and target labels y.
    - predict(X): Make predictions using the trained decision tree on input features X.
    """

    def __init__(self, depth=10, min_samples_split=2):
        """
        Initialize the Decision Tree model.

        Args:
        - depth (int): Maximum depth of the tree.
        - min_samples_split (int): Minimum number of samples required to split a node.
        """
        self.depth = depth
        self.min_samples_split = min_samples_split
        self.tree = None  # Initialize the root of the decision tree
    
    def fit(self, X, y):
        """
        Build the decision tree recursively.

        Args:
        - X (Tensor): Input features of shape (num_samples, num_features).
        - y (Tensor): Target labels of shape (num_samples,).

        Returns:
        - None
        """
        self.tree = self._build_tree(X, y, depth=0)
        
    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.

        Args:
        - X (Tensor): Input features of shape (num_samples, num_features).
        - y (Tensor): Target labels of shape (num_samples,).
        - depth (int): Current depth of the tree.

        Returns:
        - Node: Root node of the subtree.
        """
        num_samples, num_features = X.shape
        
        # Check if termination conditions are met
        if num_samples >= self.min_samples_split and depth <= self.depth:
            best_split = self._best_split(X, y, num_samples, num_features)
            if best_split["info_gain"] > 0:
                # Recursively build left and right subtrees
                left_tree = self._build_tree(best_split["X_left"], best_split["y_left"], depth + 1)
                right_tree = self._build_tree(best_split["X_right"], best_split["y_right"], depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"], left_tree, right_tree)
        
        # Create a leaf node if termination conditions are met
        leaf_value = self._calculate_leaf_value(y)
        return Node(value=leaf_value)
    
    def _best_split(self, X, y, num_samples, num_features):
        """
        Find the best split for the current node.

        Args:
        - X (Tensor): Input features of shape (num_samples, num_features).
        - y (Tensor): Target labels of shape (num_samples,).
        - num_samples (int): Number of samples in the current node.
        - num_features (int): Number of features.

        Returns:
        - dict: Dictionary containing information about the best split.
        """
        best_split = {}
        max_info_gain = -float("inf")
        
        # Iterate over each feature to find the best split
        for feature_index in range(num_features):
            feature_values = X[:, feature_index]
            possible_thresholds = torch.unique(feature_values)
            for threshold in possible_thresholds:
                # Split the data based on the current feature and threshold
                X_left, y_left, X_right, y_right = self._split(X, y, feature_index, threshold)
                if len(X_left) > 0 and len(X_right) > 0:
                    # Calculate information gain for the current split
                    info_gain = self._information_gain(y, y_left, y_right)
                    if info_gain > max_info_gain:
                        # Update best split if information gain is greater
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["X_left"] = X_left
                        best_split["y_left"] = y_left
                        best_split["X_right"] = X_right
                        best_split["y_right"] = y_right
                        best_split["info_gain"] = info_gain
                        max_info_gain = info_gain
        return best_split
    
    def _split(self, X, y, feature_index, threshold):
        """
        Split the data based on a feature and threshold.

        Args:
        - X (Tensor): Input features of shape (num_samples, num_features).
        - y (Tensor): Target labels of shape (num_samples,).
        - feature_index (int): Index of the feature to split on.
        - threshold (float): Threshold value for splitting the feature.

        Returns:
        - tuple: Tuple containing split data (X_left, y_left, X_right, y_right).
        """
        left_indices = torch.where(X[:, feature_index] <= threshold)[0]
        right_indices = torch.where(X[:, feature_index] > threshold)[0]
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]
        return X_left, y_left, X_right, y_right
    
    def _information_gain(self, y, y_left, y_right):
        """
        Calculate information gain for a split.

        Args:
        - y (Tensor): Parent node labels.
        - y_left (Tensor): Left child node labels.
        - y_right (Tensor): Right child node labels.

        Returns:
        - float: Information gain.
        """
        weight_left = len(y_left) / len(y)
        weight_right = len(y_right) / len(y)
        gain = self._entropy(y) - (weight_left * self._entropy(y_left) + weight_right * self._entropy(y_right))
        return gain
    
    def _entropy(self, y):
        """
        Calculate entropy of a set of labels.

        Args:
        - y (Tensor): Labels.

        Returns:
        - float: Entropy.
        """
        class_labels = torch.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(torch.where(y == cls)[0]) / len(y)
            if p_cls > 0:
                entropy += -p_cls * torch.log2(torch.tensor(p_cls, dtype=torch.float32))
        return entropy
    
    def _calculate_leaf_value(self, y):
        """
        Calculate the leaf value (most common label) for a leaf node.

        Args:
        - y (Tensor): Labels.

        Returns:
        - int: Leaf value.
        """
        return torch.mode(y)[0].item()
    
    def predict(self, X):
        """
        Make predictions using the trained decision tree.

        Args:
        - X (Tensor): Input features of shape (num_samples, num_features).

        Returns:
        - List: Predicted labels.
        """
        predictions = [self._make_prediction(x, self.tree) for x in X]
        return predictions
    
    def _make_prediction(self, x, tree):
        """
        Recursively make predictions using the decision tree.

        Args:
        - x (Tensor): Input features for a single sample.
        - tree (Node): Current node of the decision tree.

        Returns:
        - int: Predicted label.
        """
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self._make_prediction(x, tree.left)
        else:
            return self._make_prediction(x, tree.right)