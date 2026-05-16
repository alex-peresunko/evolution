import numpy as np

class NeuralNetwork:
    """
    A simple feed-forward neural network for multi-class classification.

    Attributes:
        layer_dims (list): A list containing the number of neurons in each layer,
                           starting with the input layer.
        parameters (dict): A dictionary holding the weights (W) and biases (b)
                           for each layer.
        learning_rate (float): The learning rate for gradient descent.
    """

    def __init__(self, input_size, hidden_layers, output_size):
        """
        Initializes the Neural Network.

        Args:
            input_size (int): The number of features in the input data.
            hidden_layers (list): A list of integers, where each integer is the
                                  number of neurons in a hidden layer.
                                  e.g., [10, 5] for two hidden layers with 10 and 5 neurons.
            output_size (int): The number of output classes.
        """
        self.layer_dims = [input_size] + hidden_layers + [output_size]
        self.parameters = {}
        self.cache = {}
        self.gradients = {}
        self._initialize_parameters()

    def _initialize_parameters(self):
        """
        Initializes the weights and biases for each layer.
        Weights are initialized with small random numbers to break symmetry.
        Biases are initialized to zero.
        """
        np.random.seed(42)  # for reproducibility
        for l in range(1, len(self.layer_dims)):
            # Weights: (size_of_current_layer, size_of_previous_layer)
            self.parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
            # Biases: (size_of_current_layer, 1)
            self.parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))

    # --- Activation Functions ---

    def _relu(self, Z):
        """ReLU activation function."""
        return np.maximum(0, Z)

    def _softmax(self, Z):
        """Softmax activation function."""
        # Subtract max for numerical stability (prevents overflow)
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    # --- Forward Propagation ---

    def forward(self, X):
        """
        Performs the forward pass through the network.

        Args:
            X (np.array): Input data of shape (input_size, number_of_examples).

        Returns:
            np.array: The output of the last layer (after softmax), of shape
                      (output_size, number_of_examples).
        """
        self.cache = {}
        A = X
        self.cache['A0'] = X
        L = len(self.parameters) // 2  # Number of layers in the network

        # Loop through hidden layers with ReLU activation
        for l in range(1, L):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(W, A) + b
            A = self._relu(Z)
            self.cache[f'Z{l}'] = Z
            self.cache[f'A{l}'] = A
        
        # Output layer with Softmax activation
        WL = self.parameters[f'W{L}']
        bL = self.parameters[f'b{L}']
        ZL = np.dot(WL, A) + bL
        AL = self._softmax(ZL)
        self.cache[f'Z{L}'] = ZL
        self.cache[f'A{L}'] = AL

        return AL

    # --- Loss Function ---

    def _compute_loss(self, Y_one_hot, AL):
        """
        Computes the cross-entropy loss.

        Args:
            Y_one_hot (np.array): One-hot encoded true labels, shape (output_size, num_examples).
            AL (np.array): The predictions from the forward pass (output of softmax).

        Returns:
            float: The cross-entropy loss.
        """
        m = Y_one_hot.shape[1]
        # Add a small epsilon for numerical stability to avoid log(0)
        loss = - (1 / m) * np.sum(Y_one_hot * np.log(AL + 1e-9))
        return np.squeeze(loss)

    # --- Backward Propagation ---

    def _relu_backward(self, dA, Z):
        """Computes the gradient of the ReLU function."""
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def backward(self, Y_one_hot):
        """
        Performs the backward pass to compute gradients.

        Args:
            Y_one_hot (np.array): One-hot encoded true labels, shape (output_size, num_examples).
        """
        self.gradients = {}
        m = Y_one_hot.shape[1]
        L = len(self.parameters) // 2 # Number of layers
        
        # Get the final activation from cache
        AL = self.cache[f'A{L}']

        # 1. Gradient for the output layer (Softmax)
        # The derivative of Cross-Entropy Loss w.r.t ZL for Softmax is simple: AL - Y
        dZL = AL - Y_one_hot
        
        A_prev = self.cache[f'A{L-1}']
        self.gradients[f'dW{L}'] = (1 / m) * np.dot(dZL, A_prev.T)
        self.gradients[f'db{L}'] = (1 / m) * np.sum(dZL, axis=1, keepdims=True)
        
        # 2. Loop backwards through the hidden layers (ReLU)
        dAPrev = self.parameters[f'W{L}'].T @ dZL
        
        for l in reversed(range(1, L)):
            dZ = self._relu_backward(dAPrev, self.cache[f'Z{l}'])
            A_prev = self.cache[f'A{l-1}']
            
            self.gradients[f'dW{l}'] = (1 / m) * np.dot(dZ, A_prev.T)
            self.gradients[f'db{l}'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            
            if l > 1:
                dAPrev = self.parameters[f'W{l}'].T @ dZ
    
    # --- Update Parameters ---
    
    def _update_parameters(self, learning_rate):
        """
        Updates the network's parameters using gradient descent.
        """
        L = len(self.parameters) // 2
        for l in range(1, L + 1):
            self.parameters[f'W{l}'] -= learning_rate * self.gradients[f'dW{l}']
            self.parameters[f'b{l}'] -= learning_rate * self.gradients[f'db{l}']

    # --- Training and Prediction ---

    def _one_hot(self, Y, num_classes):
        """Converts a 1D array of labels into a one-hot encoded matrix."""
        return np.eye(num_classes)[Y.reshape(-1)].T

    def train(self, X, Y, epochs, learning_rate, print_loss_every=100):
        """
        Trains the neural network.

        Args:
            X (np.array): Input data, shape (num_features, num_examples).
            Y (np.array): True labels, shape (1, num_examples).
            epochs (int): Number of passes through the entire dataset.
            learning_rate (float): Step size for gradient descent.
            print_loss_every (int): How often to print the loss.
        """
        num_classes = len(np.unique(Y))
        Y_one_hot = self._one_hot(Y, num_classes)

        for i in range(epochs):
            # Forward pass
            AL = self.forward(X)
            
            # Compute loss
            loss = self._compute_loss(Y_one_hot, AL)
            
            # Backward pass
            self.backward(Y_one_hot)
            
            # Update parameters
            self._update_parameters(learning_rate)
            
            if (i % print_loss_every == 0) or (i == epochs - 1):
                print(f"Epoch {i}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """
        Makes predictions on new data.

        Args:
            X (np.array): Input data, shape (num_features, num_examples).

        Returns:
            np.array: The predicted class for each example.
        """
        AL = self.forward(X)
        predictions = np.argmax(AL, axis=0)
        return predictions

### Example Usage

if __name__ == '__main__':
    # Use scikit-learn to generate sample classification data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # 1. Generate Data
    # 1000 samples, 20 features, 3 classes, 15 informative features
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                               n_informative=15, n_redundant=5, random_state=42)
    
    # The network expects features as rows and examples as columns, so we transpose X.
    # The network expects labels as a row vector.
    X = X.T
    y = y.reshape(1, -1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.2, random_state=42)
    
    # Transpose back to the network's required format
    X_train, X_test = X_train.T, X_test.T
    y_train, y_test = y_train.T, y_test.T

    print(f"Training data shape: {X_train.shape}") # (num_features, num_examples)
    print(f"Training labels shape: {y_train.shape}")   # (1, num_examples)
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # 2. Configure and Create the Neural Network
    input_features = X_train.shape[0]
    output_classes = len(np.unique(y_train))
    
    # A network with 2 hidden layers: 1st with 15 neurons, 2nd with 8 neurons
    hidden_layer_config = [15, 8] 
    
    nn = NeuralNetwork(
        input_size=input_features, 
        hidden_layers=hidden_layer_config, 
        output_size=output_classes
    )

    # 3. Train the Network
    print("\n--- Starting Training ---")
    nn.train(X_train, y_train, epochs=1500, learning_rate=0.05, print_loss_every=100)
    print("--- Training Finished ---\n")

    # 4. Evaluate the Network
    train_predictions = nn.predict(X_train)
    test_predictions = nn.predict(X_test)

    train_accuracy = accuracy_score(y_train.flatten(), train_predictions.flatten())
    test_accuracy = accuracy_score(y_test.flatten(), test_predictions.flatten())

    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
