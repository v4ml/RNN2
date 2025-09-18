import sys
sys.path.append('..')
from common.np import *
from common.layers import *
from common.functions import *

class TimeEmbedding:
    def __init__(self, W):
        # W : V, D
        self.params = [W]
        self.grads = []
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

        self.grads= [dW]


class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = []
        H, _ = Wh.shape
        #prev_h = np.zeros((H, H), dtype='float64')
        #prev_c = np.zeros((H, H), dtype='float64')

    
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
        self.next_h, self.next_c = next_h, next_c
        self.x, self.prev_h, self.prev_c  = x, prev_h, prev_c
        self.A = np.concatenate((f, g, i, o), axis=1)
        
        return next_h, next_c

    def backward(self, dh, dc):
        N, H = dh.shape
        Wx, Wh, b = self.params
        f, g, i, o = np.split(self.A, 4, axis=1)
        next_h, next_c = self.next_h, self.next_c
        df = (dc + dh * o)*next_c
        dg = (dc + dh * o)*i
        di = (dc + dh * o)*g
        do = dh * np.tanh(next_c)
        dc = (dc + dh * o) * f
        dx = np.dot(sigmoid_derivative(f), Wx.T[0:H, :]) + np.dot(tanh_derivative(g), Wx.T[H:2*H, :]) + np.dot(sigmoid_derivative(i), Wx.T[2*H:3*H, :]) + np.dot(sigmoid_derivative(o), Wx.T[3*H:4*H, :])
        dh = np.dot(sigmoid_derivative(f), Wh.T[0:H, :]) + np.dot(tanh_derivative(g), Wh.T[H:2*H, :]) + np.dot(sigmoid_derivative(i), Wh.T[2*H:3*H, :]) + np.dot(sigmoid_derivative(o), Wh.T[3*H:4*H, :])
        #dWx = np.dot(self.x.T, sigmoid_derivative(f)) + np.dot(self.x.T, tanh_derivative(g)) + np.dot(self.x.T, sigmoid_derivative(i)) + np.dot(self.x.T, sigmoid_derivative(o))
        dWx = np.concatenate((np.dot(self.x.T, sigmoid_derivative(f)), np.dot(self.x.T, tanh_derivative(g)), np.dot(self.x.T, sigmoid_derivative(i)), np.dot(self.x.T, sigmoid_derivative(o))), axis=1)
        #dWh = np.dot(self.h.T, sigmoid_derivative(f)) + np.dot(self.h.T, tanh_derivative(g)) + np.dot(self.h.T, sigmoid_derivative(i)) + np.dot(self.h.T, sigmoid_derivative(o))
        dWh = np.concatenate((np.dot(self.prev_h.T, sigmoid_derivative(f)), np.dot(self.prev_h.T, tanh_derivative(g)), np.dot(self.prev_h.T, sigmoid_derivative(i)), np.dot(self.prev_h.T, sigmoid_derivative(o))), axis=1)
        self.grads = [dWx, dWh, b]
        self.grads = [dWx, dWh, b]

        return dx
        

class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful = True):
        self.params = [Wx, Wh, b]
        self.grads = []
        self.stateful = stateful
        N,H = 20, 15
        self.dh = np.ones((N, H), dtype='float64')
        self.dc = np.ones((N, H), dtype='float64')

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H, _ = Wh.shape
        self.prev_h = np.zeros((N, H), dtype='float64')
        self.prev_c = np.zeros((N, H), dtype='float64')
        


        self.layers = []
        for t in range(T):
            layer = LSTM(Wx, Wh, b)
            self.prev_h, self.prev_c = layer.forward(xs[:, t, :], self.prev_h, self.prev_c)
            self.layers.append(layer)
        return self.prev_h # N, H
    
    def backward(self, dout):
        Wx, Wh, b = self.params
        N, H = dout.shape
        _, D = Wx.shape



        if(self.stateful):
            dh = self.dh
            dc = self.dc


        dh = dout + dh
        dc = dc
        dxs = np.empty((N, len(self.layers), D//4), dtype='float64')
        for t, layer in enumerate(reversed(self.layers)):
            dx = layer.backward(dh, dc)
            dxs[:, t, :] = dx

        self.dh = dh
        self.dc = dc

        return dxs

    def set_h(self, h):
        self.h = h
    def set_c(self, c):
        self.c = c




        


    
