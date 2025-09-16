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

        self.xs = xs

        return wordvecs

    def backward(self, dout):
        # dout : N, T, D
        # dout = dloss/dwordvecs
        # ? = dwordvecs/dW = 
        W = self.params[0]
        xs = self.xs
        N, T = xs.shape
        V, D = W.shape
        
        dW = np.zeros((V, D), dtype='float64')
        for t in range(T):
            dW[xs[:, t], :] += dout[:, t, :]

        self.grads[0] = dW


class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        H, _ = Wh.shape
        prev_h = np.zeros((H, H), dtype='float64')
        prev_c = np.zeros((H, H), dtype='float64')

    
    def forward(self, x, prev_h, prev_c):
        Wx, Wh, b = self.params
        D, H = Wx.shape
        H = H/4

        A = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
        # hs = xs * Wx + prev_hs*Wh + b
        f = sigmoid(A[:, 0:H])
        g = np.tanh(A[:, H:2*H])
        i = sigmoid(A[:, 2*H:3*H])
        o = sigmoid(A[:, 3*H:4*H])

        next_c = f*prev_c + g*i
        next_h = np.tanh(next_c)*o
        
        return next_c, next_h

    def backward(self, xs):
        pass

class TimeLSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]


    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H, _ = Wh.shape
        prev_h = np.zeros((H, H), dtype='float64')
        prev_c = np.zeros((H, H), dtype='float64')
        for t in range(T):
            layer = LSTM(Wx, Wh, b)
            prev_h, prev_c = layer.forward(xs[:, t, :], prev_h, prev_c)

        return prev_h
        


    
