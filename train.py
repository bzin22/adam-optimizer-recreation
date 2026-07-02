import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision

from optimizers import Adam, AdaGrad, SGD_Nesterov
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

"""
Load MNIST data and convert to numpy arrays
"""
# Convert images to tensors and normalize for faster processing)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST global mean and std dev
])

# Download and import the training dataset
train_dataset = datasets.MNIST(
    root='./data',      # Directory where data will be stored
    train=True,         # Load training data (60,000 samples)
    download=True,      # Download from internet if missing
    transform=transform # Apply the predefined transforms
)

# 3. Download and import the testing dataset
test_dataset = datasets.MNIST(
    root='./data', 
    train=False,        # Load testing data (10,000 samples)
    download=True,
    transform=transform # Apply the predefined transforms
)

# Wrap datasets in DataLoaders for batching and shuffling
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

tensor_images_train, tensor_labels_train = next(iter(train_loader)) # Get first batch of 64 images and labels
tensor_images_test, tensor_labels_test = next(iter(test_loader)) # Get first batch of 1000 images and labels

# Convert to numpy objects
train_images = tensor_images_train.numpy()
train_labels = tensor_labels_train.numpy()

test_images = tensor_images_test.numpy()
test_labels = tensor_labels_test.numpy()

"""
Verify the download shape
images, labels = next(iter(train_loader))
print(f"Batch Image Shape: {train_images.shape}")  # Output: torch.Size([128, 1, 28, 28])
print(f"Batch Label Shape: {train_labels.shape}")  # Output: torch.Size([128])
print(f"Batch Image Shape: {test_images.shape}")  # Output: torch.Size([1000, 1, 28, 28])
print(f"Batch Label Shape: {test_labels.shape}")  # Output: torch.Size([1000])
"""

# -----Helpers----------------------------------------------------------------------

def one_hot(classes, batch): 
    vec = np.zeros((classes, batch.shape[0]))
    for i in range(batch.shape[0]): 
        vec[batch[i]][i] = 1 
    return vec

def test_MNIST_logRegression(w_optimizer_fn, b_optimizer_fn, epochs):
    def forward(weights, biases, inputs):
        def softmax(z):
            return np.exp(z - np.max(z, axis=0, keepdims=True)) / np.sum(np.exp(z - np.max(z, axis=0, keepdims=True)), axis=0, keepdims=True)

        return softmax(np.dot(weights, inputs.T)+ biases)

    def backward(weights, biases, inputs, labels, lam):
        p = forward(weights, biases, inputs)
        loss = -np.mean(np.sum(labels * np.log(p), axis=0))
        grad_w = (1 / p.shape[1]) * np.dot((p-labels),inputs) + 2 * lam * weights
        grad_b = (1 / p.shape[1]) * np.sum((p-labels), axis=1, keepdims=True)

        return (loss, grad_w, grad_b)
    
    weights = np.random.randn(10,784) * 0.01 
    biases = np.random.randn(10,1) * 0.01 

    w_optimizer = w_optimizer_fn(weights)
    b_optimizer = b_optimizer_fn(biases)

    history = []

    for e in range(epochs):
        epoch_loss = []
        
        if hasattr(w_optimizer, 'step_epoch'):
            w_optimizer.step_epoch()
        if hasattr(b_optimizer, 'step_epoch'):
            b_optimizer.step_epoch()

        for images, labels in train_loader:
            train_imgs = images.numpy().reshape(images.shape[0], -1) # 128x784
            train_labels = one_hot(10, labels.numpy()) # 10x128

            loss, grad_w, grad_b = backward(weights, biases, train_imgs, train_labels, lam = 0.0001)
            weights = w_optimizer.update(grad_w)
            biases = b_optimizer.update(grad_b)

            epoch_loss.append(loss)
        history.append(np.mean(epoch_loss))
        print(f"Epoch {e+1}/{epochs}, loss: {history[-1]:.4f}")
        
    return history

def test_simple(stepsize, decay_1, decay_2, epsilon = 1e-8, maxiter = 1500):
    weights = np.ones((1,1))
    optimizer = Adam(stepsize, weights, decay_1, decay_2, epsilon)
    history = []

    for i in range(maxiter):
        grad = 2*weights
        weights = optimizer.update(grad)
        history.append(optimizer.weights.item())
    
    return history

def plot_results(experiments, title, xlabel, ylabel):
    plt.figure(figsize=(10, 5))
    for y, x in experiments:
        plt.plot(y, label=x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()