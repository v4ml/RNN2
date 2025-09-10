import sys
sys.path.append('..')
from common.np import *
from common.layers import *

from layer import TimeEmbedding

class Lm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (np.random.randn(V, D)/100).astype('f')

        self.layers = [
            TimeEmbedding(embed_W)
        ]

    def forward(self, xs):
        for layer in self.layers:
            loss = layer.forward(xs)
        return loss

    def backward(self):
        pass

