import sys
sys.path.append('..')
from common.np import *
from common.layers import *
from dataset import ptb

from common import config
config.GPU = True

from lm import Lm


corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, word_to_id_test, id_to_word_test = ptb.load_data('test')

corpus = corpus[:1000]
vocab_size = len(word_to_id)
vocab_size = int(max(corpus))+1
wordvec_size = 15
hidden_size = 15


model = Lm(vocab_size, wordvec_size, hidden_size)


xs = np.random.randint(0, vocab_size-1, size=(20, 5))
xs[0,0] = 0
xs[1,0] = 1
model.forward(xs)
dout = np.ones((20, 5, wordvec_size), dtype='float64')
model.backward(dout)
