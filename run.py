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
corpus_val, _, _ = ptb.load_data('val')
corpus_test, _, _ = ptb.load_data('test')
corpus_test, word_to_id_test, id_to_word_test = ptb.load_data('test')

corpus = np.array(corpus)
vocab_size = len(word_to_id) 
#corpus = corpus[0:1001]
#vocab_size = int(max(corpus))+1
corpus_val = np.array(corpus_val)
corpus_test = np.array(corpus_test)

xs = corpus[:-1]
#xs = np.array(xs)
# xs = xs.reshape(batch_size, -1)
ts = corpus[1:]
#ts = np.array(ts)
# ts = ts.reshape(batch_size, time_size)

#corpus = corpus[:1001]

batch_size = 20
wordvec_size = 650
hidden_size = 650
time_size = 35
lr = 20.0
max_epoch = 40
max_grad = 0.25
dropout = 0.5
eval_interval = 20

#batch_size = 10
#wordvec_size = 15
#hidden_size = 15
#time_size = 5
#lr = 20.0
#max_epoch = 10
#max_grad = 0.25
#dropout = 0.5
#eval_interval = 5


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




model = Lm(vocab_size, wordvec_size, hidden_size, dropout)
model.load_params('./BetterRnnlm.pkl')
#model = BetterRnnlm(vocab_size, wordvec_size, hidden_size, dropout)
optimizer = SGD(lr)
trainer = Trainer(model, optimizer, xs, ts, batch_size, time_size, max_epoch)
#trainer = RnnlmTrainer(model, optimizer)
#trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad)

#ppl = trainer.eval_perplexity(model, corpus_val)
#print('검증 퍼플렉서티: ', ppl)

best_ppl = float('inf')
for epoch in range(max_epoch): 
    trainer.fit(xs, ts, max_epoch=1, batch_size=batch_size,
                time_size=time_size, max_grad=max_grad, eval_interval=eval_interval)

    model.reset_state()
    ppl = trainer.eval_perplexity(model, corpus_val)
    print('검증 퍼플렉서티: ', ppl)

    if best_ppl > ppl:
        best_ppl = ppl
        model.save_params()
    else:
        lr /= 4.0
        optimizer.lr = lr

    model.reset_state()
    print('-' * 50)

# model.forward(xs, ts)
# dxs = np.ones((20, 5, wordvec_size), dtype='f')
# dhs = np.ones((batch_size, time_size, hidden_size), dtype='f')
# dvs = np.ones((batch_size, time_size, vocab_size), dtype='f')
# dvs = 1
# model.backward(dvs)
