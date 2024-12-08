import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from scipy.special import softmax
import pandas as pd

class BackpropagationModel(BaseEstimator, ClassifierMixin):
    def __init__(self, layer_conf, max_epoch=1000, max_error=0.1, learn_rate=0.5, print_per_epoch=100):
        self.layer_conf = layer_conf
        self.max_epoch = max_epoch
        self.max_error = max_error
        self.learn_rate = learn_rate
        self.print_per_epoch = print_per_epoch if print_per_epoch > 0 else 100 
        self.w = None
        self.epoch = 0
        self.mse = 1
        self.classes_ = None
        self.mse_history = []

    def _initialize_weights(self):
        # Initialize weights including bias for each layer
        np.random.seed(1)
        self.w = [
            np.random.rand(self.layer_conf[i] + 1, self.layer_conf[i + 1]) * 0.1  # +1 for bias
            for i in range(len(self.layer_conf) - 1)
        ]

    @staticmethod
    def sig(X):
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def sigd(X):
        s = BackpropagationModel.sig(X)
        return s * (1 - s)

    def _softmax(self, x):
        return softmax(x, axis=0)

    def bp_fit(self, X, target):
        # Add bias term to input
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        self.classes_ = np.unique(target)
        is_binary = len(self.classes_) == 2  # Determine binary vs. multi-class

        # Convert target to NumPy array if it is a Series
        target = np.array(target) if isinstance(target, pd.Series) else target

        # One-hot encoding for multi-class, add axis for binary
        if not is_binary:
            y = np.eye(len(self.classes_))[target]
        else:
            y = target.reshape(-1, 1)

        self._initialize_weights()
        epoch, mse = 0, 1

        while (self.max_epoch == -1 or epoch < self.max_epoch) and mse > self.max_error:
            epoch += 1
            mse = 0

            for r in range(len(X)):
                # **Initialize n**
                n = [X[r]]  # Start with the input layer (including bias)

                # Forward pass
                for L in range(len(self.w)):
                    activation = np.dot(n[L], self.w[L])  # Weighted sum
                    if L < len(self.w) - 1:  # Hidden layers
                        layer_output = self.sig(activation)
                        n.append(np.append(layer_output, 1))  # Add bias
                    else:  # Output layer
                        n.append(self.sig(activation) if is_binary else self._softmax(activation))

                # Calculate error and MSE
                e = y[r] - n[-1]
                mse += np.sum(e ** 2)

                # Backward pass with weight updates
                d = e * (self.sigd(np.dot(n[-2], self.w[-1])) if is_binary else e)
                for L in range(len(self.w) - 1, -1, -1):
                    dw = self.learn_rate * np.outer(n[L], d)
                    self.w[L] += dw
                    if L > 0:
                        d = np.dot(d, self.w[L][:-1].T) * self.sigd(np.dot(n[L-1], self.w[L-1]))

            mse /= len(X)

            # Append the MSE for this epoch
            self.mse_history.append(mse)

            # Avoid division by zero by checking print_per_epoch > 0
            if self.print_per_epoch > 0 and epoch % self.print_per_epoch == 0:
                print(f"Epoch {epoch}, MSE: {mse:.6f}")

        self.epoch = epoch
        self.mse = mse

    def bp_predict(self, X):
        X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias term to input
        predictions = []

        for x in X:
            n = x
            for layer_weights in self.w:
                activation = np.dot(n, layer_weights)
                n = (self.sig(activation) if layer_weights is not self.w[-1] 
                        else (self.sig(activation) if len(self.classes_) == 2 else self._softmax(activation)))
                n = np.append(n, 1) if layer_weights is not self.w[-1] else n
            predictions.append(n)

            return np.array(predictions)

    def fit(self, X, y):
        self.bp_fit(X, y)
        return self

    def predict(self, X):
        X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias term to input
        predictions = []

        for x in X:
            n = x
            for layer_weights in self.w:
                activation = np.dot(n, layer_weights)
                n = (self.sig(activation) if layer_weights is not self.w[-1] 
                     else (self.sig(activation) if len(self.classes_) == 2 else self._softmax(activation)))
                n = np.append(n, 1) if layer_weights is not self.w[-1] else n
            predictions.append(n)

        return np.array(predictions)

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def get_params(self, deep=True):
        return {
            'layer_conf': self.layer_conf,
            'max_epoch': self.max_epoch,
            'max_error': self.max_error,
            'learn_rate': self.learn_rate,
            'print_per_epoch': self.print_per_epoch
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
