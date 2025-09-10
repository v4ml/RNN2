import sys
sys.path.append('..')
from common.np import *
from common.layers import *
from common.functions import sigmoid

class TimeEmbedding:
    def __init__(self, W):
        # W : V, D
        self.params = [W]
        pass

    def forward(self, xs):
        # xs : N, T
        W = self.params[0]
        V, D = W.shape
        N, T = xs.shape

        wordvecs = np.zeros((N, T, D), dtype='float64')
        for t in range(T):
            print(xs[:, t])
            a = W[xs[:, t], :]
            wordvecs[:, t, :] = W[xs[:, t], :]

        return wordvecs

    def backward(self, dout):
        # dout : N, T, D
        # dout = dloss/dwordvecs
        # ? = dwordvecs/dW = 
        self.grads = [dout]
