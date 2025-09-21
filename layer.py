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

    def backward(self, dW):
        # dout : N, T, D
        # dout = dloss/dwordvecs
        # ? = dwordvecs/dW = 
        W = self.params[0]
        xs = self.xs
        N, T = xs.shape
        V, D = W.shape
        
        dWs = np.zeros((V, D), dtype='float64')
        for t in range(T):
            dWs[xs[:, t], :] += dW[:, t, :]

        self.grads= [dWs]


class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = []
        H, _ = Wh.shape
        #prev_h = np.zeros((H, H), dtype='float64')
        #prev_c = np.zeros((H, H), dtype='float64')

    
    def forward(self, x, prev_h, prev_c):
        Wx, Wh, b = self.params
        self.x, self.prev_h, self.prev_c  = x, prev_h, prev_c
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
        db = np.concatenate((df, dg, di, do), axis=1)
        db = np.sum(db, axis=0)
        self.grads = [dWx, dWh, db]
        self.dx = dx

        return dh
        

class TimeLSTM:
    def __init__(self, Wx, Wh, b, batch_size, time_size, stateful = True):
        self.params = [Wx, Wh, b]
        self.grads = []
        self.stateful = stateful
        D, H = Wx.shape
        H = H//4
        N, T = batch_size, time_size
    
    
        self.prev_h = np.zeros((N, H), dtype='float64')
        self.prev_c = np.zeros((N, H), dtype='float64')
        self.hs = np.zeros((N, T, H), dtype='float64')
        self.cs = np.zeros((N, T, H), dtype='float64')        
        self.dh = np.ones((N, H), dtype='float64')
        self.dc = np.ones((N, H), dtype='float64')

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H, _ = Wh.shape

        self.layers = []
        for t in range(T):
            layer = LSTM(Wx, Wh, b)
            self.prev_h, self.prev_c = layer.forward(xs[:, t, :], self.prev_h, self.prev_c)
            self.layers.append(layer)
            self.hs[:, t, :] = self.prev_h
            self.cs[:, t, :] = self.prev_c

        return self.hs # N, T, H
    
    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        if(self.stateful):
            dh = self.dh
            dc = self.dc

        #dhs = np.empty((N, len(self.layers), H//4), dtype='float64')
        dcs = np.empty((N, len(self.layers), H//4), dtype='float64')
        dxs = np.empty((N, len(self.layers), D), dtype='float64')
        dWxs = np.zeros((D, H), dtype='float64')
        dWhs = np.zeros((H//4, H), dtype='float64')
        dbs = np.zeros((H), dtype='float64')
        for t, layer in enumerate(reversed(self.layers)):
            dh = dhs[:, t, :] + dh
            dh = layer.backward(dh, dc)
            dxs[:, t, :] = layer.dx
            dhs[:, t, :] = dh
            dcs[:, t, :] = dc
            dWxs += layer.grads[0]
            dWhs += layer.grads[1]
            dbs += layer.grads[2]
        self.dh = dh
        self.dc = dc

        return dxs

    def set_h(self, h):
        self.h = h
    def set_c(self, c):
        self.c = c




        


    
