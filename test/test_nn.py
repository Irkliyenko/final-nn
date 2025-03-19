import numpy as np
import pytest
from typing import List
from numpy.typing import ArrayLike
from nn.nn import NeuralNetwork
from nn.preprocess import sample_seqs, one_hot_encode_seqs

# Sample architecture for testing
nn_arch = [
    {"input_dim": 4, "output_dim": 3, "activation": "relu"},
    {"input_dim": 3, "output_dim": 1, "activation": "sigmoid"}
]
    
@pytest.fixture
def dummy_nn():
    return NeuralNetwork(nn_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function='binary_cross_entropy')

def test_single_forward(dummy_nn):
    W = np.array([[0.2, -0.5, 0.3, 0.8], [-0.2, 0.5, -0.3, -0.8], [0.1, -0.1, 0.1, -0.1]])
    b = np.array([[0.1], [0.2], [-0.1]])
    A_prev = np.array([[1], [0], [1], [0]])

    A_curr, Z_curr = dummy_nn._single_forward(W, b, A_prev, activation='relu')

    assert A_curr.shape == (3, 1)
    assert (A_curr >= 0).all()  # ReLU should output non-negative values

def test_forward(dummy_nn):
    X = np.array([[0.5], [0.1], [0.3], [0.9]])
    output, cache = dummy_nn.forward(X)

    assert output.shape == (1, 1)
    assert "A0" in cache and "Z1" in cache

def test_single_backprop(dummy_nn):
    W = np.random.randn(3, 4)
    b = np.random.randn(3, 1)
    Z = np.random.randn(3, 1)
    A_prev = np.random.randn(4, 1)
    dA = np.random.randn(3, 1)

    dA_prev, dW, db = dummy_nn._single_backprop(W, b, Z, A_prev, dA, activation_curr='relu')

    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

def test_predict(dummy_nn):
    X = np.array([[0.2], [0.4], [0.6], [0.8]])
    y_hat = dummy_nn.predict(X)

    assert y_hat.shape == (1, 1)

def test_binary_cross_entropy(dummy_nn):
    y = np.array([[1], [0], [1]])
    y_hat = np.array([[0.9], [0.1], [0.8]])

    loss = dummy_nn._binary_cross_entropy(y, y_hat)

    assert loss > 0  # Binary cross-entropy should be positive

def test_binary_cross_entropy_backprop(dummy_nn):
    y = np.array([[1], [0]])
    y_hat = np.array([[0.9], [0.1]])

    dA = dummy_nn._binary_cross_entropy_backprop(y, y_hat)

    assert dA.shape == y.shape

def test_mean_squared_error(dummy_nn):
    y = np.array([[1], [0], [1]])
    y_hat = np.array([[0.9], [0.1], [0.8]])

    loss = dummy_nn._mean_squared_error(y, y_hat)

    assert loss > 0  # MSE should be positive

def test_mean_squared_error_backprop(dummy_nn):
    y = np.array([[1], [0]])
    y_hat = np.array([[0.9], [0.1]])

    dA = dummy_nn._mean_squared_error_backprop(y, y_hat)

    assert dA.shape == y.shape

def test_sample_seqs():
    seqs = ["ATG", "GCT", "TGA", "CGA"]
    labels = [True, False, True, True]

    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)

    assert len(sampled_seqs) == len(sampled_labels)
    assert sampled_labels.count(True) == sampled_labels.count(False)

def test_one_hot_encode_seqs():
    seqs = ["ATG"]
    encoded = one_hot_encode_seqs(seqs)

    expected_encoding = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]])  # A = [1,0,0,0], T = [0,1,0,0], G = [0,0,0,1]

    assert np.array_equal(encoded, expected_encoding)

