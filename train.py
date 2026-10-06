import torch, torchvision
import os
import time

from utils import *
from optimizers import *
from torchvision import datasets, transforms

def train(backwards_fn, params, optimizer, inputs, labels, epochs, batch_size=128, seed=0, lam=0.0,
          record_batches=False):
    """
    loss_and_gradients: fn(params, inputs, labels) -> (cost, weight gradients, bias gradients)
    params: list of arrays [W, b]
    optimizer: single instance whose .weights is the packed vector
    inputs: (N, features), labels: (classes, N)

    record_batches=True also returns the per-minibatch loss, so a figure can be
    plotted at finer resolution than one point per epoch. Figure 3(a) needs this:
    the epoch mean averages over ~391 minibatches starting from ln(10), which
    hides the early separation between optimizers.
    """
    start = time.time()

    shapes = [p.shape for p in params]
    theta = pack(params) 
    optimizer.weights = theta # optimizer has flattened weights & biases

    rng = np.random.default_rng(seed)
    N = inputs.shape[0]
    history = []
    batch_losses = []

    for e in range(epochs):
        epoch_start = time.time()
        epoch_loss = []
        shuffled_input = rng.permutation(N)

        # if hasattr(optimizer, 'step_epoch'):
        #     optimizer.step_epoch()
        
        # Mini-batches
        for i in range(0, N, batch_size):
            idx = shuffled_input[i:i+batch_size]
            batch_input, batch_label = inputs[idx], labels[:,idx]

            current_params = unpack(optimizer.weights, shapes) 
            loss, grads = backwards_fn(current_params, batch_input, batch_label, lam)
            grad_vec = pack(grads)
            optimizer.update(grad_vec)

            epoch_loss.append(loss)
            if record_batches:
                batch_losses.append(loss)

        history.append(np.mean(epoch_loss))
        
        print(f"Epoch {e+1}/{epochs}, loss: {history[-1]:.4f}, "
          f"epoch: {time.time()-epoch_start:.1f}s, total: {(time.time()-start)/60:.1f}min")

    if record_batches:
        return history, np.array(batch_losses)
    return history

def load_CIFAR10(whiten=True, cache='./cifar10_zca.npy'):
    """
    Load CIFAR-10 as (N, 3, 32, 32) float32 and one-hot labels.

    The paper trains Figure 3 on whitened CIFAR-10, so whiten=True applies global
    contrast normalization (per image, subtract the mean and divide by the std over
    all 3072 values) followed by ZCA. The 3072x3072 eigendecomposition takes a few
    seconds, so the result is cached.
    """

    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True
    )

    train_labels = one_hot(10, np.array(train_dataset.targets))

    if whiten and cache and os.path.exists(cache):
        return np.load(cache), train_labels

    train_images = train_dataset.data.astype(np.float32) / 255
    train_images = np.transpose(train_images, (0,3,1,2))

    if not whiten:
        return train_images, train_labels

    N = train_images.shape[0]
    X = train_images.reshape(N, -1).astype(np.float64)

    # Global contrast normalization
    X -= X.mean(axis=1, keepdims=True)
    X /= (X.std(axis=1, keepdims=True) + 1e-8)

    # ZCA: W = E diag(1/sqrt(s + eps)) E.T
    X -= X.mean(axis=0, keepdims=True)
    cov = np.dot(X.T, X) / N
    s, E = np.linalg.eigh(cov)
    W = np.dot(E * (1.0 / np.sqrt(s + 1e-5)), E.T)
    X = np.dot(X, W)

    train_images = X.reshape(N, 3, 32, 32).astype(np.float32)

    if cache:
        np.save(cache, train_images)

    return train_images, train_labels

def load_MNIST(standardize=True):
    """
    Load MNIST data and convert to numpy arrays

    standardize=False leaves pixels in [0,1], which is what the VAE in Figure 4
    needs since it treats them as Bernoulli probabilities.
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
    train_images = train_dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    if standardize:
        train_images = (train_images - 0.1307) / 0.3081
    train_labels = one_hot(10, train_dataset.targets.numpy())

    return train_images, train_labels