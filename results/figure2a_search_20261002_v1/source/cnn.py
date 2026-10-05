"""
CIFAR-10 ConvNet for Figure 3.

This is the one model in the project that does not hand-code its backward pass.
Convolution gradients come from torch autograd; every parameter update still goes
through the NumPy optimizers in optimizers.py. The bridge is make_cnn_backward,
which matches the backwards_fn contract train() already expects:

    backwards_fn(params, inputs, labels, lam) -> (loss, grads)

so the CNN runs through the same pack/unpack/optimizer.update path as logreg and
the MLP, with no changes to train().

Architecture is the paper's Section 6.3 net: c1 5x5x64, s1 3x3 max pool stride 2,
c2 5x5x64, s2 pool, c3 5x5x128, s3 pool, then 1000 ReLU units and a 10-way output.
On 32x32 input the three pools give 32 -> 15 -> 7 -> 3, so the flatten is
128*3*3 = 1152 and the whole net is 1,475,266 parameters.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FLATTEN_DIM = 128 * 3 * 3   # 1152
N_PARAMS = 1_475_266


def build_cnn(dropout=True, seed=0):
    """Returns the module. Does not apply it."""
    torch.manual_seed(seed)

    layers = []
    if dropout:
        layers.append(nn.Dropout(0.2))      # matches the MLP's 0.8 input keep rate

    layers += [
        nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        nn.Linear(FLATTEN_DIM, 1000),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))

    # No softmax here. F.cross_entropy applies its own log-softmax.
    layers.append(nn.Linear(1000, 10))

    return nn.Sequential(*layers)


def cnn_params(model):
    """The module's parameters as a list of NumPy arrays, for pack()/unpack()."""
    return [p.detach().cpu().numpy().astype(np.float64) for p in model.parameters()]


def make_cnn_backward(model, device):
    """
    Closes over the module and returns a function with train()'s signature.

    Each call writes the optimizer's NumPy parameters into the module, runs one
    forward/backward, and hands the gradients back as NumPy. The optimizer vector
    is the source of truth; the module is just the thing that computes gradients.
    """
    tensors = list(model.parameters())
    model.to(device)

    def cnn_backward(params, inputs, labels, lam):
        with torch.no_grad():
            for tensor, param in zip(tensors, params):
                tensor.copy_(torch.from_numpy(np.ascontiguousarray(param)))

        model.zero_grad(set_to_none=True)

        x = torch.from_numpy(np.ascontiguousarray(inputs)).float().to(device)
        # train() carries labels as one-hot (classes, N); cross_entropy wants indices.
        y = torch.from_numpy(labels.argmax(axis=0)).long().to(device)

        loss = F.cross_entropy(model(x), y)
        if lam:
            loss = loss + (lam / 2) * sum((t**2).sum() for t in tensors if t.dim() > 1)
        loss.backward()

        grads = [t.grad.detach().cpu().numpy().astype(np.float64) for t in tensors]
        return float(loss.item()), grads

    return cnn_backward


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    model = build_cnn(dropout=True)
    n = sum(p.numel() for p in model.parameters())
    print("params:", n, "expected:", N_PARAMS)
    assert n == N_PARAMS
    print("output shape:", model(torch.randn(4, 3, 32, 32)).shape)
