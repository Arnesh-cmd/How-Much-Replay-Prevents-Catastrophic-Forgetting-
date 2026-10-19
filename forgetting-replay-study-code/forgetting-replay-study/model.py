"""
model.py: the neural network.

A small "multilayer perceptron" (MLP): layers of artificial neurons where
every neuron connects to every neuron in the next layer.

    784 pixels  ->  hidden layer  ->  hidden layer  ->  10 scores (one per digit)

Each artificial neuron computes:  (input1 x weight1) + (input2 x weight2) + ... + bias
then ReLU sets negative results to 0. Training adjusts the weights and biases.

The width (neurons per hidden layer) is a setting, so we can test whether
bigger networks forget less.
"""

import torch.nn as nn


def make_model(hidden_width=100):
    return nn.Sequential(
        nn.Flatten(),                          # 28x28 grid -> list of 784 numbers
        nn.Linear(784, hidden_width),          # hidden layer 1
        nn.ReLU(),
        nn.Linear(hidden_width, hidden_width), # hidden layer 2
        nn.ReLU(),
        nn.Linear(hidden_width, 10),           # one score per digit 0-9
    )


def count_weights(model):
    """How many adjustable numbers the network has (useful for your paper)."""
    return sum(p.numel() for p in model.parameters())
