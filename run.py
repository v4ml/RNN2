import sys
sys.path.append('..')
from common.np import *
from common.layers import *
#from common.time_layers import *
from dataset import ptb

from common import config
config.GPU = True

from lm import Lm
from trainer import Trainer
from common.optimizer import SGD
from better_rnnlm import BetterRnnlm
from common.trainer import RnnlmTrainer


corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, word_to_id_test, id_to_word_test = ptb.load_data('test')

#corpus = corpus[:1001]
vocab_size = len(word_to_id) 
#vocab_size = int(max(corpus))+1
batch_size = 20
wordvec_size = 650
hidden_size = 650
time_size = 35
lr = 20.0
max_epoch = 40
max_grad = 0.25
dropout = 0.5


# vocab_size = len(word_to_id) 
# vocab_size = int(max(corpus))+1
# wordvec_size = 650
# hidden_size = 650
# time_size = 35
# batch_size = 20
# max_epoch = 40
# max_grad = 0.25
# lr = 20.0                                                                      



# xs = np.random.randint(0, vocab_size-1, size=(batch_size, time_size))
# xs[0,0] = 0
# xs[1,0] = 1
# ts = xs+1



corpus = np.array(corpus)
xs = corpus[:-1]
#xs = np.array(xs)
# xs = xs.reshape(batch_size, -1)
ts = corpus[1:]
#ts = np.array(ts)
# ts = ts.reshape(batch_size, time_size)
#model = Lm(vocab_size, wordvec_size, hidden_size, batch_size, time_size)
model = BetterRnnlm(vocab_size, wordvec_size, hidden_size, dropout)
optimizer = SGD(lr)
#trainer = Trainer(model, optimizer, xs, ts, batch_size, time_size, max_epoch)
trainer = RnnlmTrainer(model, optimizer)
trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad)

# model.forward(xs, ts)
# dxs = np.ones((20, 5, wordvec_size), dtype='f')
# dhs = np.ones((batch_size, time_size, hidden_size), dtype='f')
# dvs = np.ones((batch_size, time_size, vocab_size), dtype='f')
# dvs = 1
# model.backward(dvs)
