import random
import math

class SimpleLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def _initialize_weights(self, n_features):
        """Initialize weights and bias based on number of features"""
        self.weights = [random.uniform(-0.01, 0.01) for _ in range(n_features)]
        self.bias = random.uniform(-0.01, 0.01)
    
    def _predict_single(self, features):
        """Predict for a single sample"""
        prediction = self.bias
        for i in range(len(features)):
            prediction += self.weights[i] * features[i]
        return prediction
    
    def predict(self, X):
        """Predict for multiple samples"""
        if self.weights is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        for features in X:
            predictions.append(self._predict_single(features))
        return predictions
    
    def _compute_loss(self, y_true, y_pred):
        """Compute Mean Squared Error"""
        n = len(y_true)
        total_error = 0
        for i in range(n):
            error = y_true[i] - y_pred[i]
            total_error += error * error
        return total_error / n
    
    def _compute_gradients(self, X, y_true, y_pred):
        """Compute gradients for weights and bias"""
        n = len(X)
        n_features = len(X[0])
        
        # Initialize gradients
        weight_gradients = [0.0 for _ in range(n_features)]
        bias_gradient = 0.0
        
        # Compute gradients
        for i in range(n):
            error = y_pred[i] - y_true[i]
            bias_gradient += error
            
            for j in range(n_features):
                weight_gradients[j] += error * X[i][j]
        
        # Average the gradients
        bias_gradient /= n
        for j in range(n_features):
            weight_gradients[j] /= n
            
        return weight_gradients, bias_gradient
    
    def fit(self, data):
        """
        Train the linear regression model
        data: 2D list where each row is [feature1, feature2, ..., target]
        """
        # Extract features and target
        X = []
        y = []
        
        for row in data:
            X.append(row[:-1])  # All columns except last
            y.append(row[-1])   # Last column is target
        
        # Get number of features from first row
        n_features = len(X[0])
        
        # Initialize weights based on number of features
        self._initialize_weights(n_features)
        
        print(f"Training with {len(X)} samples and {n_features} features")
        
        # Training loop
        for iteration in range(self.n_iterations):
            # Forward pass
            y_pred = self.predict(X)
            
            # Compute loss
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # Compute gradients
            weight_gradients, bias_gradient = self._compute_gradients(X, y, y_pred)
            
            # Update weights and bias
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.bias -= self.learning_rate * bias_gradient
            
            # Print progress
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}: Loss = {loss:.6f}")
    
    def evaluate(self, data):
        """Evaluate model performance"""
        # Extract features and target
        X = []
        y_true = []
        
        for row in data:
            X.append(row[:-1])
            y_true.append(row[-1])
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        mse = self._compute_loss(y_true, y_pred)
        rmse = math.sqrt(mse)
        
        # Calculate R-squared
        y_mean = sum(y_true) / len(y_true)
        ss_tot = sum((y - y_mean) ** 2 for y in y_true)
        ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2_score
        }
    
    def get_coefficients(self):
        """Return the learned coefficients"""
        if self.weights is None:
            raise ValueError("Model must be trained first")
        
        return {
            'bias': self.bias,
            'weights': self.weights.copy()
        }


def generate_synthetic_data(n_samples=1000, n_features=2, noise_level=0.5):
    """
    Generate synthetic data with related features
    Feature 1: Random
    Feature 2: N1 * Feature1 + C1
    Feature 3: N2 * Feature2 + C2
    And so on...
    """
    random.seed(42)
    
    # Generate multipliers (N) and constants (C) for feature relationships
    multipliers = []
    constants = []
    for i in range(n_features - 1):  # n_features-1 relationships
        multiplier = random.uniform(-5, 5)
        constant = random.uniform(-20, 20)
        multipliers.append(multiplier)
        constants.append(constant)
    
    # True coefficients for final target - dynamically generate based on n_features
    true_weights = [random.uniform(1.0, 5.0) for _ in range(n_features)]
    true_bias = random.uniform(-10, 10)
    
    print(f"Feature relationships:")
    print(f"Feature 1: Random values")
    for i in range(len(multipliers)):
        print(f"Feature {i+2}: {multipliers[i]:.3f} * Feature{i+1} + {constants[i]:.3f}")
    
    data = []
    for _ in range(n_samples):
        features = []
        
        # Generate first feature randomly
        feature1 = random.uniform(0, 10)
        features.append(feature1)
        
        # Generate subsequent features based on previous feature
        current_feature = feature1
        for i in range(n_features - 1):
            next_feature = multipliers[i] * current_feature + constants[i]
            features.append(next_feature)
            current_feature = next_feature
        
        # Calculate target with some noise
        target = true_bias
        for i in range(n_features):
            target += true_weights[i] * features[i]
        target += random.gauss(0, noise_level)  # Add noise
        
        # Combine features and target
        row = features + [target]
        data.append(row)
    
    return data, true_weights, true_bias


def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    data, true_weights, true_bias = generate_synthetic_data(n_samples=1000, n_features=2)
    
    # Split data into train/test (80/20 split)
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    print(f"Training data: {len(train_data)} samples")
    print(f"Test data: {len(test_data)} samples")
    
    # Create and train the model
    model = SimpleLinearRegression(learning_rate=0.01, n_iterations=1000)
    
    print("\nTraining model...")
    model.fit(train_data)
    
    # Evaluate on training and test sets
    print("\nEvaluating model...")
    train_metrics = model.evaluate(train_data)
    test_metrics = model.evaluate(test_data)
    
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
    for i, weight in enumerate(coeffs['weights']):
        print(f"Feature {i+1}: {weight:.4f} (True: {true_weights[i]})")


if __name__ == "__main__":
    main()