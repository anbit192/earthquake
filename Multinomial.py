import numpy as np
import pandas as pd

class MultinomialLogisticRegression:
    def __init__(self, num_classes, learning_rate=0.01, num_iterations=1000):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.W = None
        self.b = None

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def initialize_parameters(self, num_features):
        self.W = np.random.randn(num_features, self.num_classes)
        self.b = np.zeros((1, self.num_classes))

    def fit(self, X, y):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        y = y.to_numpy() if isinstance(y, pd.Series) else y

        num_samples, num_features = X.shape
        self.initialize_parameters(num_features)

        for _ in range(self.num_iterations):
            Z = np.dot(X, self.W) + self.b
            A = self.softmax(Z)

            # Compute gradient
            dZ = A - np.eye(self.num_classes)[y]
            dW = (1 / num_samples) * np.dot(X.T, dZ)
            db = (1 / num_samples) * np.sum(dZ, axis=0, keepdims=True)

            # Update parameters
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

    def predict(self, X):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        Z = np.dot(X, self.W) + self.b
        A = self.softmax(Z)
        return np.argmax(A, axis=1)

# Example usage:
X_train = pd.DataFrame({'feature1': [1, 2, 3, 4], 'feature2': [2, 3, 4, 5]})
y_train = pd.Series([0, 1, 0, 1])  # Example labels, assuming two classes (0 and 1)

model = MultinomialLogisticRegression(num_classes=2)
model.fit(X_train, y_train)

# Example predictions:
X_test = pd.DataFrame({'feature1': [1.5, 3.5], 'feature2': [2.5, 4.5]})
predictions = model.predict(X_test)
print("Predictions:", predictions)
