import sys
sys.path.append('..')
from common.np import *
from common.layers import *

from layer import *

class Lm:
    def __init__(self, vocab_size, wordvec_size, hidden_size, batch_size, time_size):


        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (np.random.randn(V, D)/100).astype('f')
        LSTM_Wx = (np.random.randn(D, 4*H)/np.sqrt(H)).astype('f')
        LSTM_Wh = (np.random.randn(H, 4*H)/np.sqrt(H)).astype('f')
        LSTM_b = np.zeros(H*4).astype('f')
        Affine_W = (np.random.randn(H, V)/np.sqrt(V)).astype('f')
        Affine_b = np.zeros(V).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(LSTM_Wx, LSTM_Wh, LSTM_b, batch_size, time_size),
            TimeAffine(Affine_W, Affine_b),
        ]
        self.lossLayer = TimeSoftmaxWithLoss(vocab_size)

        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        
        loss = self.lossLayer.forward(xs, ts)

        return loss

    def backward(self, dout=1):
        dout = self.lossLayer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

