import numpy as np


class SnakeModel:
    def __init__(self,
                 input_units,
                 hidden_units,
                 output_units):
        self.l1 = np.random.randn(
            input_units, hidden_units) * np.sqrt(1/input_units)
        self.l1_b = np.zeros((1, hidden_units))
        self.l2 = np.random.randn(
            hidden_units, output_units) * np.sqrt(1/input_units)
        self.l2_b = np.zeros((1, output_units))

    def forward(self, state):
        x = (state.T @ self.l1) + self.l1_b
        x = relu(x)
        x = (x @ self.l2) + self.l2_b
        return softmax(x)

    def __call__(self, x):
        return self.forward(x)


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
