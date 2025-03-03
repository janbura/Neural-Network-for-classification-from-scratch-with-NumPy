import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
#NUMPY FUNCTIONS
def initialize_parameters():
    W1 = np.random.uniform(-0.5, 0.5, (10, 784))
    b1 = np.random.uniform(-0.5, 0.5, (10, 1))
    W2 = np.random.uniform(-0.5, 0.5, (10, 10))
    b2 = np.random.uniform(-0.5, 0.5, (10, 1))
    return W1, b1, W2, b2


def ReLU(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def one_hot_encode(y, num_classes = 10):
    one_hot = np.zeros((num_classes, y.size))
    one_hot[y, np.arange(y.size)] = 1
    return one_hot


def ReLU_deriv(Z):
    return (Z > 0).astype(float)


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = X.shape[1]
    one_hot_Y = one_hot_encode(Y)

    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis = 1, keepdims = True)

    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis = 1, keepdims = True)

    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)

def gradient_descent(X, Y, learning_rate, iterations):
    W1, b1, W2, b2 = initialize_parameters()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        if i % 100 == 0:  # Print every 10 iterations
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration {i} - Accuracy: {accuracy:.4f}")

    return W1, b1, W2, b2






#PYTORCH FUNCTIONS
class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.fc1 = nn.Linear(784, 10)  # First layer (Input: 784, Hidden: 10)
        self.fc2 = nn.Linear(10, 10)  # Second layer (Hidden: 10, Output: 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)  # Apply ReLU
        x = self.fc2(x)
        return torch.softmax(x, dim=1)  # Apply softmax


def train_pytorch_model(model, train_loader, epochs=100, learning_rate=0.1):
    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # SGD Optimizer

    accuracies = []

    for epoch in range(epochs):
        correct, total = 0, 0
        for images, labels in train_loader:
            images = images.view(-1, 784)  # Flatten images
            optimizer.zero_grad()  # Clear gradients

            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss

            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Track accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        accuracies.append(accuracy)

        if epoch % 100 == 0:
            print(f"PyTorch Model - Epoch {epoch}, Accuracy: {accuracy:.4f}")

    return accuracies






#EVALUATION/VISUALISATION FUNCTIONS
def visualize_samples(train_dataset, num_samples=10):
    """
    Visualizes a row of sample images from the dataset.
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        image, label = train_dataset[i]  # image is a tensor (1, 28, 28)
        image = image.squeeze().numpy()  # Remove channel dimension and convert to NumPy
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def evaluate_numpy_model(W1, b1, W2, b2, X_test, Y_test):
    """
    Evaluates the NumPy model, prints accuracy, and returns predictions.

    Args:
        W1, b1, W2, b2: Trained model parameters.
        X_test (np.ndarray): Test dataset features.
        Y_test (np.ndarray): True labels.

    Returns:
        float: Accuracy
        np.ndarray: Model predictions
    """
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X_test.T)
    predictions = get_predictions(A2)
    accuracy = get_accuracy(predictions, Y_test)
    print(f"NumPy Model Test Accuracy: {accuracy:.4f}")
    return accuracy, predictions


def evaluate_pytorch_model(model, test_loader):
    """
    Evaluates the PyTorch model, prints accuracy, and returns predictions.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        test_loader (DataLoader): PyTorch DataLoader for test dataset.

    Returns:
        float: Accuracy
        np.ndarray: Model predictions
    """
    correct, total = 0, 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 784)  # Flatten images
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.numpy())  # Collect predictions
            all_labels.extend(labels.numpy())  # Collect actual labels

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"PyTorch Model Test Accuracy: {accuracy:.4f}")
    return accuracy, np.array(all_predictions), np.array(all_labels)


def plot_heatmap(y_true, y_pred, model_name="Model"):
    """
    Generates and plots a confusion matrix heatmap for classification performance.
    """
    cm = confusion_matrix(y_true, y_pred)  # Compute confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    # Print classification report
    print(f"Classification Report for {model_name}:")
    print(classification_report(y_true, y_pred))

