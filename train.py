import torch, torchvision

from utils import *
from optimizers import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def train(loss_and_gradients, params, optimizer, inputs, labels, epochs, batch_size=128, seed=0, lam=0.0001):
    """
    loss_and_gradients: fn(params, inputs, labels) -> (cost, weight gradients, bias gradients)
    params: list of arrays [W, b]
    optimizer: single instance whose .weights is the packed vector
    inputs: (N, features), labels: (classes, N)
    """

    shapes = [p.shape for p in params]
    theta = pack(params) 
    optimizer.weights = theta # optimizer has flattened weights & biases

    rng = np.random.default_rng(seed)
    N = inputs.shape[0]
    history = []

    for e in range(epochs):
        epoch_loss = []
        shuffled_input = rng.permutation(N)

        if hasattr(optimizer, 'step_epoch'):
            optimizer.step_epoch()
        
        for i in range(0, N, batch_size):
            idx = shuffled_input[i:i+batch_size]
            batch_input, batch_label = inputs[idx], labels[:,idx]

            current_params = unpack(optimizer.weights, shapes) 
            loss, grads = loss_and_gradients(current_params, batch_input, batch_label, lam)
            grad_vec = pack(grads)
            optimizer.update(grad_vec)

            epoch_loss.append(loss)

        history.append(np.mean(epoch_loss))
        print(f"Epoch {e+1}/{epochs}, loss: {history[-1]:.4f}")
        
    return history

def load_MNIST():
    """
    Load MNIST data and convert to numpy arrays
    """

    # Download and import the training dataset
    train_dataset = datasets.MNIST(
        root='./data',      # Directory where data will be stored
        train=True,         # Load training data (60,000 samples)
        download=True,      # Download from internet if missing
    )

    # Download and import the testing dataset
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False,        # Load testing data (10,000 samples)
        download=True,
    )

    # 4. Convert images/labels to np arrays and one hot encode labels
    train_images = train_dataset.data.numpy().reshape(-1, 784).astype(np.float32)
    train_images = (train_images / 255.0 - 0.1307) / 0.3081
    train_labels = one_hot(10, train_dataset.targets.numpy())

    return train_images, train_labels