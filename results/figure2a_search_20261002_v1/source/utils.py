import matplotlib.pyplot as plt
import numpy as np

def relu(z):
    return np.maximum(0,z)

def sigmoid(z):
    # Branchless stable form: exp() is only ever called on non-positive values.
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out

def softplus(z):
    # log(1 + exp(z)), stable for large |z|. Its derivative is sigmoid(z).
    return np.logaddexp(0.0, z)

def softmax(z):
    z_shift = z- np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def one_hot(classes, batch): 
    vec = np.zeros((classes, batch.shape[0]))
    for i in range(batch.shape[0]): 
        vec[batch[i]][i] = 1 
    return vec

def pack(params):
    return np.concatenate([p.ravel() for p in params])

def unpack(vec, shapes):
    result = []
    idx = 0
    for shape in shapes:
        size = int(np.prod(shape))
        result.append(vec[idx:idx+size].reshape(shape))
        idx += size
    assert idx == vec.size, "vector size does not match shapes."
    return result

def plot_results(experiments, title, xlabel, ylabel, save=None):
    plt.figure(figsize=(10, 5))
    for y, x in experiments:
        plt.plot(y, label=x)
    plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    plt.show()