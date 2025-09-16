import sys
sys.path.append('..')
from common.np import *
from common.layers import *

from layer import *

class Lm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (np.random.randn(V, D)/100).astype('f')
        LSTM_Wx = (np.random.randn(D, 4*H)/np.sqrt(H)).astype('f')
        LSTM_Wh = (np.random.randn(H, 4*H)/np.sqrt(H)).astype('f')
        LSTM_b = np.zeros(H*4).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(LSTM_Wx, LSTM_Wh, LSTM_b)
        ]

    def forward(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

