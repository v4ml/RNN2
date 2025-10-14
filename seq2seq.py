import sys
sys.path.append('..')
from dataset import sequence

from common.np import *
from layer_new import *
#from common.layers import *
#from common.time_layers import *
from dataset import ptb

from common import config
config.GPU = True

from lm import Lm



class Seq2seq:
    def __init__(self, vocab_size, wordvec_siez, hidden_size):
        self.encoder = Encoder(vocab_size, wordvec_siez, hidden_size)
        self.decoder = Decoder(vocab_size, wordvec_siez, hidden_size)

    def forward(self, xs, ts):
        h = self.encoder.forward(xs)
        for i, layer in enumerate(self.decoder.lstm_layers):
            layer.set_state(h[i])
        loss = self.decoder.forward(np.full((45000, 1), 6), ts)
        return loss

    def backward(self, dout=1):
        self.decoder.backward(dout)
        dh = self.decoder.lstm_layers[0].dh
        #for layer in self.decoder.lstm_layers:
        #    self.encoder.backward( layer.dh )
        self.encoder.backward(dh)
        pass


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

    def forward(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        
        h = []
        for i, layer in enumerate(self.lstm_layers):
            h.append( layer.h )

        return h

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

class Decoder(Lm):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        super().__init__(vocab_size, wordvec_size, hidden_size)

    def forward(self, xs, ts):
        return super().forward(xs, ts)

    def backward(self, dout=1):
        return super().backward(dout)

(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt', seed=1984)
char_to_id, id_to_char = sequence.get_vocab()
seq2seq = Seq2seq(len(char_to_id), 50, 50)
seq2seq.forward(x_train, t_train)
seq2seq.backward()
