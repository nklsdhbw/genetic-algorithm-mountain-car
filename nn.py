import numpy as np

class NeuralNetwork():
    def __init__(self, hidden_size, state_size, action_size):
        self.w1 = np.random.randn(state_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        self.w2 = np.random.randn(hidden_size, action_size)
        self.b2 = np.random.randn(action_size)

    def forward(self, state):
        z1 = np.dot(state, self.w1) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        return z2

    def copy(self):
        new_nn = NeuralNetwork(hidden_size=self.w1.shape[1], state_size=self.w1.shape[0], action_size=self.w2.shape[1])
        new_nn.w1 = np.copy(self.w1)
        new_nn.b1 = np.copy(self.b1)
        new_nn.w2 = np.copy(self.w2)
        new_nn.b2 = np.copy(self.b2)
        return new_nn