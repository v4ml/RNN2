import sys
sys.path.append('..')
from common.np import *
from common.layers import *
#from layer import *
from layer_new import *
from common.base_model import BaseModel

class Lm(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size, dropout_ratio=0.5):


        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (np.random.randn(V, D)/100).astype('f')
        LSTM_Wx0 = (np.random.randn(D, 4*H)/np.sqrt(D)).astype('f')
        LSTM_Wh0 = (np.random.randn(H, 4*H)/np.sqrt(H)).astype('f')
        LSTM_Wx1 = (np.random.randn(D, 4*H)/np.sqrt(D)).astype('f')
        LSTM_Wh1 = (np.random.randn(H, 4*H)/np.sqrt(H)).astype('f')
        LSTM_b0 = np.zeros(H*4).astype('f')
        
        LSTM_b1 = np.zeros(H*4).astype('f')
        Affine_W = (np.random.randn(H, V)/np.sqrt(H)).astype('f')
        Affine_b = np.zeros(V).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(LSTM_Wx0, LSTM_Wh0, LSTM_b0),
            TimeDropout(dropout_ratio),
            #TimeLSTM(LSTM_Wx1, LSTM_Wh1, LSTM_b1),
            #TimeDropout(dropout_ratio),
            #TimeAffine(Affine_W, Affine_b),
            TimeAffine(embed_W.T, Affine_b)
        ]
        self.lossLayer = TimeSoftmaxWithLoss()
        #self.lstm_layers = [self.layers[2], self.layers[4]]
        self.lstm_layers = [self.layers[2]]

        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):
        xs = self.predict(xs)
        loss = self.lossLayer.forward(xs, ts)

        return loss
    
    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def backward(self, dout=1):
        dout = self.lossLayer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()