import random
import pandas as pd
import numpy as np

class LinearRegression:
    def __init__(self, Learning_rate=0.001, n_iterations=100):
        self.Learning_Rate = Learning_rate
        self.n_iterations = n_iterations
        self.weights = [1.0, 1.3]
        self.bias = 0.0

    def predict(sefl, x, y):
        return (sefl.weights[0] * x + sefl.weights[1] * y + sefl.bias)
    
    def compute_loss(self, y_true, y_pred):
        n = len(y_true)
        return np.sum((y_true - y_pred) ** 2) / n
    
    def compute_gradient(self, x1, x2, y_true, y_pred):
        n = len(y_true)
        dw0 = -2 * np.sum((y_true - y_pred) * x1) / n
        dw1 = -2 * np.sum((y_true - y_pred) * x2) / n
        db = -2 * np.sum((y_true - y_pred)) / n
        return dw0, dw1, db
    
    def update_parameters(self, dw0, dw1, db):
        self.weights[0] -= self.Learning_Rate * dw0
        self.weights[1] -= self.Learning_Rate * dw1
        self.bias -= self.Learning_Rate * db

    def fit(self, x1, x2, y):
        for iteration in range(self.n_iterations):
            y_pred = self.predict(x1,x2)
            loss = self.compute_loss(y, y_pred)
            dw0, dw1, db = self.compute_gradient(x1, x2, y, y_pred)
            self.update_parameters(dw0, dw1, db)
            if (iteration+1) % 10 == 0:
                print(f"Iteration {iteration+1}: Loss = {loss}")
    
    def evaluate(self, x1, x2, y):
        y_pred = self.predict(x1, x2)
        loss = self.compute_loss(y, y_pred)
        return (f"Final Loss: {loss}")


def main():
    # Generate synthetic data
    np.random.seed(42)
    x1 = np.random.rand(100) * 10
    x2 = np.random.rand(100) * 10
    y = 3 * x1 + 2 * x2 + np.random.randn(100) * 2

    # Create and train the model
    model = LinearRegression(Learning_rate=0.01, n_iterations=100)
    model.fit(x1, x2, y)

    # Evaluate the model
    result = model.evaluate(x1, x2, y)
    print(result)

if __name__ == "__main__":
    main()