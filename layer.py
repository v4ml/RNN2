import sys
sys.path.append('..')
from common.np import *
from common.layers import *
from common.functions import *
import matplotlib.pyplot as plt

class TimeEmbedding:
    def __init__(self, W):
        # W : V, D
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        pass

    def forward(self, xs):
        # xs : N, T
        W = self.params[0]
        V, D = W.shape
        N, T = xs.shape

        wordvecs = np.zeros((N, T, D), dtype='float32')
        for t in range(T):
            #print(xs[:, t])
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
        
        dWs = np.zeros((V, D), dtype='float32')
        for t in range(T):
            print(xs[:, t])
            dWs[xs[:, t], :] += dW[:, t, :]

        self.grads= [dWs]


class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = []
        H, _ = Wh.shape
        #prev_h = np.zeros((H, H), dtype='float32')
        #prev_c = np.zeros((H, H), dtype='float32')

    
    def forward(self, x, prev_h, prev_c):
        Wx, Wh, b = self.params
        D, H = Wx.shape
        H = H//4

        A = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
        # hs = xs * Wx + prev_hs*Wh + b
        f = sigmoid(A[:, 0:H])
        g = np.tanh(A[:, H:2*H])
        i = sigmoid(A[:, 2*H:3*H])
        o = sigmoid(A[:, 3*H:4*H])

        next_c = f*prev_c + g*i
        next_h = np.tanh(next_c)*o
        self.cache = [x, prev_h, prev_c, next_c, np.concatenate((f, g, i, o), axis=1)]
        #self.c = next_c
        
        #self.A = np.concatenate((f, g, i, o), axis=1)
        
        return next_h, next_c

    def backward(self, dh, dc):
        N, H = dh.shape
        Wx, Wh, b = self.params
        x, prev_h, prev_c, next_c, A = self.cache
        f, g, i, o = np.split(A, 4, axis=1)
        #c = self.c
        #temp = dc + dh * o
        ds = dc + (dh*o)*(1-next_c**2)
        df = ds * prev_c
        dg = ds * i
        di = ds * g
        do = dh * np.tanh(next_c)
        df = df*f*(1-f)
        dg = dg*(1-g**2)
        di = di*i*(1-i)
        do = do*o*(1-o)

        dA = np.concatenate((df, dg, di, do), axis=1)
        dc = ds*f
        #dx = np.dot(df*f*(1-f), Wx.T[0:H, :]) + np.dot(dg*(1-g**2), Wx.T[H:2*H, :]) + np.dot(di*i*(1-i), Wx.T[2*H:3*H, :]) + np.dot(do*o*(1-o), Wx.T[3*H:4*H, :])
        dx = np.dot(dA, Wx.T) # N, H x H, D
        #dh = np.dot(df*f*(1-f), Wh.T[0:H, :]) + np.dot(dg*(1-g**2), Wh.T[H:2*H, :]) + np.dot(di*i*(1-i), Wh.T[2*H:3*H, :]) + np.dot(do*o*(1-o), Wh.T[3*H:4*H, :])
        dh = np.dot(dA, Wh.T)
        #dWx = np.concatenate((np.dot(x.T, df*f*(1-f)), np.dot(x.T, dg*(1-g**2)), np.dot(x.T, di*i*(1-i)), np.dot(x.T, do*o*(1-o))), axis=1)
        dWx = np.dot(x.T, dA)
        #dWh = np.concatenate((np.dot(prev_h.T, df*f*(1-f)), np.dot(prev_h.T, dg*(1-g**2)), np.dot(prev_h.T, di*i*(1-i)), np.dot(prev_h.T, do*o*(1-o))), axis=1)
        dWh = np.dot(prev_h.T, dA)
        #db = np.concatenate((df*f*(1-f), dg*(1-g**2), di*i*(1-i), do*o*(1-o)), axis=1)
        db = np.sum(dA, axis=0)
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        #self.dx = dx # TimeLSTM에서 사용
        #self.dc = dc
        return dx, dh, dc
     

class TimeLSTM:
    def __init__(self, Wx, Wh, b, batch_size, time_size, stateful = True):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.stateful = stateful
        D, H = Wx.shape
        H = H//4
        N, T = batch_size, time_size
    
        self.dh = np.ones((N, H), dtype='float32')
        self.dc = np.ones((N, H), dtype='float32')

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H, _ = Wh.shape

        prev_h = np.zeros((N, H), dtype='float32')
        prev_c = np.zeros((N, H), dtype='float32')
        hs = np.zeros((N, T, H), dtype='float32')
        cs = np.zeros((N, T, H), dtype='float32')  
        
        self.layers = []
        
        for t in range(T):
            layer = LSTM(Wx, Wh, b)
            prev_h, prev_c = layer.forward(xs[:, t, :], prev_h, prev_c)
            self.layers.append(layer)
            hs[:, t, :] = prev_h
            cs[:, t, :] = prev_c

        return hs # N, T, H
    
    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape
        H = H//4

        if(self.stateful):
            dh = self.dh
            dc = self.dc
        else:
            dh = np.ones((N, H), dtype='float32')
            dc = np.ones((N, H), dtype='float32')

        #dhs = np.empty((N, len(self.layers), H//4), dtype='float32')
        #dcs = np.empty((N, len(self.layers), H//4), dtype='float32')
        dxs = np.empty((N, len(self.layers), D), dtype='float32')
        dWx = np.zeros((D, H*4), dtype='float32')
        dWh = np.zeros((H, H*4), dtype='float32')
        db = np.zeros((H*4), dtype='float32')
        for t, layer in enumerate(reversed(self.layers)):
            dh = dhs[:, -1-t, :] + dh
            dx, dh, dc = layer.backward(dh, dc)
            dxs[:, t, :] = dx
            #dhs[:, t, :] = dh
            #dcs[:, t, :] = dc
            dWx += layer.grads[0]
            dWh += layer.grads[1]
            db += layer.grads[2]
        
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        self.dh = dh
        self.dc = dc

        return dxs

    def set_h(self, h):
        self.h = h
    def set_c(self, c):
        self.c = c



class TimeAffine:
    def __init__(self, Wh, b):
        self.params = [Wh, b]
        self.grads = [np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, hs):
        # hs : N,T,H
        Wh, b = self.params
        N, T, H = hs.shape
        self.cache = [hs]
        vs = np.dot(hs, Wh)+b
        return vs # N, T, V
        

    def backward(self, dvs): # N, T, V
        Wh, b = self.params
        hs = self.cache[0]
        N, T, V = dvs.shape
        H = hs.shape[2]
                 #     V    VH
        dhs = np.dot(dvs, Wh.T)
                #      NTH   NTV
        dWh = np.dot((hs.reshape(-1, H)).T, dvs.reshape(-1, V)) # H, V
        # nth ntv -> hv
        # htn ntv
        db = np.sum(dvs, axis=(0,1))

        self.grads[0][...] = dWh
        self.grads[1][...] = db

        return dhs # N, T, H
     

class TimeSoftmaxWithLoss:
    def __init__(self, vocab_size):
        self.params = [vocab_size]
        self.cache = None
        #self.t = t
        pass

    def forward(self, vs, ts):
        vocab_size = self.params[0]

        N, T, V = vs.shape
        max_exp = np.max(vs, axis=2, keepdims=True)
        exp = np.exp(vs-max_exp) # N,T,V
        sum_exp = np.sum(exp, axis=2) # N,T
        #temp = temp.reshape(-1, V)
        #tempAll = tempAll.reshape(N, T, 1) 
        #ys = np.empty((N, T, V), dtype='float32')
        #for i in range(N*T):
        ys = exp/sum_exp[..., None]  # 

        # ts의 shape를 (20, 5)에서 (20, 5, vocab_size)로 변경
        #ts = np.eye(vocab_size)[ts] # N, T, V
        delta = 1e-7
        loss = -np.sum(np.log(ys[np.arange(N)[:, None], np.arange(T), ts]+delta))/(N*T)
        #sum_loss = -np.sum(loss, axis=(0,1,2))/(N*T)

        self.cache = [ys, ts, N, T]
        # targets = np.zeros((N, T), dtype='int')

        # for t in range(T):
        #     idx = np.argmax(dvs[:, t, :], axis=1) # N
        #     targets[:, t] = idx
        return loss # N, T

    def backward(self, dloss=-1):
        ys, ts, N, T = self.cache
        dloss = ys.copy()
        dloss[np.arange(N)[:, None], np.arange(T), ts] -= 1
        return dloss/(N*T)


