import numpy as np
from sklearn.tree import DecisionTreeRegressor

class AIVNGradientBooster:

    def __init__(self, max_depth=8, min_samples_split=5, min_samples_leaf=5, max_features=3, lr=0.1, num_iter=50):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.lr = lr
        self.num_iter = num_iter
        self.y_mean = 0

    def __calculate_loss(self,y, y_pred):
        loss = (1/len(y)) * 0.5 * np.sum(np.square(y-y_pred))
        return loss

    def __take_gradient(self, y, y_pred):
        # grad = -(y-y_pred) # cho dự đoán = pred - alpha * residual
        grad = (y - y_pred)  # cho dự đoán = pred + alpha * residual
        return grad

    def __create_base_model(self, X, y):
        base = DecisionTreeRegressor(max_depth=self.max_depth,
                                    min_samples_split=self.min_samples_split,
                                    min_samples_leaf=self.min_samples_leaf,
                                    max_features=self.max_features)
        base.fit(X,y)
        return base

    def predict(self,models,y,X):
        pred_0 = np.array([self.y_mean] * len(X))
        pred = pred_0.reshape(len(pred_0),1)

        for i in range(len(models)):
            temp = (models[i].predict(X)).reshape(len(X),1)
            # pred -= self.lr * temp #cho dự đoán = pred - alpha * residual
            pred += self.lr * temp #cho dự đoán = pred + alpha * residual

        return pred

    def train(self, X, y):
        models = []
        losses = []
        self.y_mean = np.mean(y)
        pred_0 = np.array([np.mean(y)] * len(y))
        pred = pred_0.reshape(len(pred_0),1)
        # print("pred_0", pred_0)

        for epoch in range(self.num_iter):
            loss = self.__calculate_loss(y, pred)
            # print("loss: epoch", epoch, "=", loss)
            losses.append(loss)
            grads = self.__take_gradient(y, pred)
            # print("gradi", grads)
            base = self.__create_base_model(X, grads)
            r = (base.predict(X)).reshape(len(X),1)
            # print("r", r)
            # pred -= self.lr * r
            pred += self.lr * r
            models.append(base)

        return models, losses, pred_0