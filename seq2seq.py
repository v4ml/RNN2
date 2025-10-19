import sys
sys.path.append('..')
from dataset import sequence

from common.np import *
#from layer_new import *
from common.time_layers import*
#from common.layers import *
#from common.time_layers import *
from dataset import ptb

from common import config
config.GPU = True

from seq2seqLm import Lm
import pickle
import os



class Seq2seq:
    def __init__(self, batch_size, vocab_size, wordvec_siez, hidden_size, length, char_to_id, id_to_char):
        self.encoder = Encoder(vocab_size, wordvec_siez, hidden_size)
        self.decoder = Decoder(batch_size, vocab_size, wordvec_siez, hidden_size)
        self.params, self.grads = [], []
        self.params += self.encoder.params + self.decoder.params
        self.grads += self.encoder.grads + self.decoder.grads
        self.id_to_char = id_to_char
        self.char_to_id = char_to_id
        self.loss_layer = TimeSoftmaxWithLoss()
        self.length = length

    def predict(self, xs):
        h = self.encoder.forward(xs) # encoder의 lstm_layers에서 array로 h 리턴
        ts = self.decoder.predict(h, self.length)
        # encoder에서 받은 h를 decoder에 설정
        # for i, layer in enumerate(self.decoder.lstm_layers):
        #     layer.set_state(h[i])
        result = np.argmax(ts, axis=2)
        #for i in range(ts.shape[0]):
        #    for k, input in enumerate(result[i]):
        #        print(self.id_to_char[input], end = '')
        
        return ts
        

    def forward(self, xs, ts):
        h = self.encoder.forward(xs) #self.predict(xs)
        wordvecs = self.decoder.predict(h, self.length)
        
        loss = self.loss_layer.forward(wordvecs, ts)

        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        self.decoder.backward(dout)
        dh = np.zeros((45000, 7, 128), dtype='f')
        dh[:, -1, :] = self.decoder.lstm_layers[0].dh
        #for layer in self.decoder.lstm_layers:
        #    self.encoder.backward( layer.dh )
        self.encoder.backward(dh)
        pass

    def reset_state(self):
        self.encoder.reset_state()
        self.decoder.reset_state()

    def save_params(self, file_name=None):
        if not file_name:
            file_name = self.__class__.__name__ + '.pkl'

        params = [p.astype(np.float16) for p in self.params]
        if GPU:
            params = [self.to_cpu(p) for p in params]

        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
    
    def to_cpu(self, x):
        import numpy
        if type(x) == numpy.ndarray:
            return x
        return np.asnumpy(x)    
    def to_gpu(self, x):
        import cupy
        if type(x) == cupy.ndarray:
            return x
        return cupy.asarray(x)
        

    def load_params(self, file_name=None):
        if not file_name:
            file_name = self.__class__.__name__ + '.pkl'

        if '/' in file_name:
            file_name = file_name.replace('/', os.sep)

        if not os.path.exists(file_name):
            raise IOError('No file: ' + file_name)

        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        params = [p.astype('f') for p in params]
        if GPU:
            params = [self.to_gpu(p) for p in params]

        for i, param in enumerate(self.params):
            param[...] = params[i]         

class Encoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (np.random.randn(V, D)/100).astype('f')
        LSTM_Wx0 = (np.random.randn(D, 4*H)/np.sqrt(D)).astype('f')
        LSTM_Wh0 = (np.random.randn(H, 4*H)/np.sqrt(H)).astype('f')
        LSTM_b0 = np.zeros(H*4).astype('f')

        self.layers =[
            TimeEmbedding(embed_W),
            TimeLSTM(LSTM_Wx0, LSTM_Wh0, LSTM_b0)
        ]
        self.lstm_layers = [self.layers[1]]

        self.params, self.grads = [], []

    def forward(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
            self.params += layer.params
        
        h = []
        for i, layer in enumerate(self.lstm_layers):
            h.append( layer.h )

        return h

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
            self.grads += layer.grads
        return dout
    
    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()    

class Decoder(Lm):
    def __init__(self, batch_size, vocab_size, wordvec_size, hidden_size):
        super().__init__(vocab_size, wordvec_size, hidden_size)
        self.cache = [batch_size, vocab_size, wordvec_size]
        #self.params, self.grads = [], []
        #self.params.extend(super().params)
        #self.grads.extend(super().grads)
        
    def predict(self, h, length):
        N, V, D = self.cache

        for i, layer in enumerate(self.lstm_layers): # encoder의 결과 h를 decoder에 입력
            layer.set_state(h[i])


        xs = np.full((N, 5), 5, dtype='int')
        ts = np.empty((N, 5, V), dtype='f')
        for i in range(length-1):
            ts[:, i, :] = super().predict(xs[:, i].reshape(45000, 1)).reshape(45000, 13)
            xs[:, i+1] = np.argmax(ts[:, i, :], axis=1)
            
        return ts 

    def forward(self, h, xs):
        for i, layer in enumerate(self.lstm_layers): # encoder의 결과 h를 decoder에 입력
            layer.set_state(h[i])

        N, V = xs.shape    
        xs = np.array(xs)
        #ts = np.full(xs.shape, 5, dtype='int')
        #ts[:, :V-1] = xs[:, 1:V]
        #loss = super().forward(xs, ts)
        return xs

    def backward(self, dout=1):
        dout = super().backward(dout)
        return dout
    
    def reset_state(self):
        super().reset_state()



