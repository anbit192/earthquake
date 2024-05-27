import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split  # Use sklearn only for splitting the data
from collections import defaultdict

class MultinomialLogisticRegression:
    def __init__(self, n_components=None, learning_rate=0.01, max_iter=1000):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def standardize(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return (X - self.mean) / self.std

    def fit_pca(self, X):
        X = self.standardize(X)
        covariance_matrix = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_idx]
        
        if self.n_components is None or self.n_components >= X.shape[1]:
            self.eigenvectors = None  # No PCA is applied
        else:
            self.eigenvectors = sorted_eigenvectors[:, :self.n_components]

    def transform_pca(self, X):
        X = (X - self.mean) / self.std
        if self.eigenvectors is not None:
            X = np.dot(X, self.eigenvectors)
        return X

    def one_hot_encode(self, y):
        n_classes = len(np.unique(y))
        return np.eye(n_classes)[y]

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))

    def fit(self, X_train, y_train):
        self.fit_pca(X_train)
        X_train_transformed = self.transform_pca(X_train)
        y_train_encoded = self.one_hot_encode(y_train)
        
        n_samples, n_features = X_train_transformed.shape
        n_classes = y_train_encoded.shape[1]

        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))

        for i in range(self.max_iter):
            linear_model = np.dot(X_train_transformed, self.weights) + self.bias
            y_pred = self.softmax(linear_model)

            gradient_w = (1 / n_samples) * np.dot(X_train_transformed.T, (y_pred - y_train_encoded))
            gradient_b = (1 / n_samples) * np.sum(y_pred - y_train_encoded, axis=0, keepdims=True)

            self.weights -= self.learning_rate * gradient_w
            self.bias -= self.learning_rate * gradient_b

    def predict(self, X_test):
        X_test_transformed = self.transform_pca(X_test)
        linear_model = np.dot(X_test_transformed, self.weights) + self.bias
        y_pred = self.softmax(linear_model)
        return np.argmax(y_pred, axis=1)

    def evaluate(self, y_test, y_pred):
        accuracy = np.mean(y_test == y_pred)
        print(f'Accuracy: {accuracy}')

        precision_scores = []
        recall_scores = []
        f1_scores = []
        for cls in np.unique(y_test):
            tp = np.sum((y_test == cls) & (y_pred == cls))
            fp = np.sum((y_test != cls) & (y_pred == cls))
            fn = np.sum((y_test == cls) & (y_pred != cls))
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        precision = np.mean(precision_scores)
        recall = np.mean(recall_scores)
        f1 = np.mean(f1_scores)

        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')

        return accuracy, precision, recall, f1

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = defaultdict(int)
        for t, p in zip(y_test, y_pred):
            cm[(t, p)] += 1

        cm_matrix = np.zeros((len(np.unique(y_test)), len(np.unique(y_test))), dtype=int)
        for (t, p), count in cm.items():
            cm_matrix[t, p] = count

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()


X = df.drop('damage_grade', axis=1).values
y = df['damage_grade'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


mlr = MultinomialLogisticRegression(n_components=None, learning_rate=0.01, max_iter=1000)  # Use n_components=None to keep all components


mlr.fit(X_train, y_train)


y_pred = mlr.predict(X_test)


mlr.evaluate(y_test, y_pred)


mlr.plot_confusion_matrix(y_test, y_pred)
