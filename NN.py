import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def reluDerivative(x):
    return np.where(x > 0, 1, 0)

class NeuralNetwork:
    def __init__(self, x, y, hidden_layers=[128, 64], learning_rate=0.001):
        self.input = x
        self.y = y
        self.learning_rate = learning_rate

        # Initialize weights and biases with Xavier/Glorot initialization
        layer_sizes = [self.input.shape[1]] + hidden_layers + [self.y.shape[1]]
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / (layer_sizes[i] + layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]
        self.biases = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]

    def feedforward(self):
        self.layers = [self.input]
        for i in range(len(self.weights) - 1):
            self.layers.append(relu(np.dot(self.layers[-1], self.weights[i]) + self.biases[i]))
        self.output = np.dot(self.layers[-1], self.weights[-1]) + self.biases[-1]  # Linear activation for output layer

    def backprop(self):
        d_output = 2 * (self.y - self.output)  # Derivative of linear activation is 1
        deltas = [d_output]
        
        for i in reversed(range(len(self.weights) - 1)):
            deltas.append(np.dot(deltas[-1], self.weights[i+1].T) * reluDerivative(self.layers[i+1]))
        deltas.reverse()

        for i in range(len(self.weights)):
            grad_w = np.dot(self.layers[i].T, deltas[i])
            grad_b = np.sum(deltas[i], axis=0, keepdims=True)
            
            # Clip gradients to prevent exploding gradients (outside of computational bounds)
            grad_w = np.clip(grad_w, -1, 1)
            grad_b = np.clip(grad_b, -1, 1)
            
            self.weights[i] += self.learning_rate * grad_w
            self.biases[i] += self.learning_rate * grad_b

    def train(self, iterations):
        for i in range(iterations):
            self.feedforward()
            self.backprop()
            if i % 1 == 0:
                loss = np.mean(np.square(self.y - self.output))
                print(f"Iteration {i}, Loss: {loss}")

    def predict(self, x):
        layer = x
        for i in range(len(self.weights) - 1):
            layer = relu(np.dot(layer, self.weights[i]) + self.biases[i])
        output = np.dot(layer, self.weights[-1]) + self.biases[-1]  # Linear activation for output layer
        return output

if __name__ == "__main__":
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Normalize and reshape data
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    
    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Initialize and train the neural network
    nn = NeuralNetwork(X_train, y_train, hidden_layers=[128, 64], learning_rate=0.001)
    nn.train(300)
    
    # Predict on test data
    predictions = nn.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_labels == true_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Visualize some predictions
    num_images = 25
    plt.figure(figsize=(10, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"P:{predicted_labels[i]}\nT:{true_labels[i]}")
        plt.axis('off')
    plt.show()

