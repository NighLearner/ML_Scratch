import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _add_bias_column(self, X):
        """Add bias column to feature matrix for vectorized computation"""
        return np.column_stack([np.ones(X.shape[0]), X])

    def predict(self, X):
        """Vectorized prediction using matrix multiplication"""
        if self.weights is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Add bias column and compute predictions
        X_with_bias = self._add_bias_column(X)
        return X_with_bias @ self.weights

    def compute_loss(self, y_true, y_pred):
        """Compute Mean Squared Error loss"""
        return np.mean((y_true - y_pred) ** 2)

    def compute_gradients(self, X, y_true, y_pred):
        """Vectorized gradient computation"""
        n = len(y_true)
        X_with_bias = self._add_bias_column(X)
        
        # Vectorized gradient computation: X^T * (y_pred - y_true) / n
        gradients = X_with_bias.T @ (y_pred - y_true) / n
        return gradients

    def fit(self, X, y):
        """
        Train the linear regression model
        X: feature matrix of shape (n_samples, n_features)
        y: target vector of shape (n_samples,)
        """
        # Convert to numpy arrays if not already
        X = np.array(X)
        y = np.array(y)
        
        # Handle both 1D and 2D input
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Initialize weights (including bias as first weight)
        self.weights = np.random.normal(0, 0.01, n_features + 1)
        
        # Training loop
        for iteration in range(self.n_iterations):
            # Forward pass
            y_pred = self.predict(X)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # Compute gradients
            gradients = self.compute_gradients(X, y, y_pred)
            
            # Update weights
            self.weights -= self.learning_rate * gradients
            
            # Print progress
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: Loss = {loss:.6f}")

    def evaluate(self, X, y):
        """Evaluate model performance"""
        y_pred = self.predict(X)
        loss = self.compute_loss(y, y_pred)
        
        # Calculate R-squared
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        
        return {
            'mse': loss,
            'rmse': np.sqrt(loss),
            'r2_score': r2_score
        }

    def get_coefficients(self):
        """Return the learned coefficients"""
        if self.weights is None:
            raise ValueError("Model must be trained first")
        
        return {
            'bias': self.weights[0],
            'coefficients': self.weights[1:]
        }


def main():
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Create feature matrix
    X = np.random.rand(n_samples, 2) * 10
    
    # Create target with known relationship: y = 3*x1 + 2*x2 + noise
    true_coeffs = [3, 2]
    true_bias = 1
    y = true_bias + X @ true_coeffs + np.random.randn(n_samples) * 0.5
    
    # Split data into train/test
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create and train the model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate on training and test sets
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)
    
    # Get learned coefficients
    coeffs = model.get_coefficients()
    
    # Print results
    print(f"\nTraining Metrics:")
    print(f"MSE: {train_metrics['mse']:.6f}")
    print(f"RMSE: {train_metrics['rmse']:.6f}")
    print(f"R²: {train_metrics['r2_score']:.6f}")
    
    print(f"\nTest Metrics:")
    print(f"MSE: {test_metrics['mse']:.6f}")
    print(f"RMSE: {test_metrics['rmse']:.6f}")
    print(f"R²: {test_metrics['r2_score']:.6f}")
    
    print(f"\nLearned Coefficients:")
    print(f"Bias: {coeffs['bias']:.4f} (True: {true_bias})")
    print(f"Feature 1: {coeffs['coefficients'][0]:.4f} (True: {true_coeffs[0]})")
    print(f"Feature 2: {coeffs['coefficients'][1]:.4f} (True: {true_coeffs[1]})")


if __name__ == "__main__":
    main()