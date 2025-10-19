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
from common.optimizer import SGD
from common.optimizer import Adam
from common.trainer import RnnlmTrainer
from seq2seqTrainer import Seq2SeqTrainer
import pickle
from seq2seq import Seq2seq


def eval_perplexity(model, xs, ts, batch_size=10, time_size=35):
        print('퍼플렉서티 평가 중 ...')
        # corpus_size = len(corpus)
        # total_loss, loss_cnt = 0, 0
        # max_iters = (corpus_size - 1) // (batch_size * time_size)
        # jump = (corpus_size - 1) // batch_size

        loss = model.forward(xs, ts)

        # for iters in range(max_iters):
        #     xs = np.zeros((batch_size, time_size), dtype=np.int32)
        #     ts = np.zeros((batch_size, time_size), dtype=np.int32)
        #     time_offset = iters * time_size
        #     offsets = [time_offset + (i * jump) for i in range(batch_size)]
        #     for t in range(time_size):
        #         for i, offset in enumerate(offsets):
        #             xs[i, t] = corpus[(offset + t) % corpus_size]
        #             ts[i, t] = corpus[(offset + t + 1) % corpus_size]

        #     try:
        #         loss = model.forward(xs, ts, train_flg=False)
        #     except TypeError:
        #         loss = model.forward(xs, ts)
        #     total_loss += loss

        #     sys.stdout.write('\r%d / %d' % (iters, max_iters))
        #     sys.stdout.flush()

        
        ppl = np.exp(loss)
        print('ppl = ',ppl)
        return ppl            

(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt', seed=1984)
char_to_id, id_to_char = sequence.get_vocab()

in_vocab_size = x_train.shape[1]
out_vocab_size = t_train.shape[1]
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 128
dropout = 0.5
max_epoch = 40
max_grad = 5.0
lr = 20.0
batch_size = x_train.shape[0]
length = t_train.shape[1]

seq2seq = Seq2seq(batch_size, vocab_size, wordvec_size, hidden_size, length, char_to_id, id_to_char)
# seq2seq.forward(x_train, t_train)
# seq2seq.backward()


#model = Lm(out_vocab_size, wordvec_size, hidden_size, dropout)
optimizer = SGD(lr)
trainer = Seq2SeqTrainer(seq2seq, optimizer)

def shuffle_data(x, t):
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    return x[indices.get()], t[indices.get()]

best_ppl = float('inf')
for epoch in range(max_epoch):
    xs, ts = shuffle_data(x_train, t_train)
    trainer.fit(xs, ts, 1)

    seq2seq.reset_state()
    ppl = eval_perplexity(seq2seq, x_test, t_test)
    print('검증 퍼플렉서티: ', ppl)

    if best_ppl > ppl:
        best_ppl = ppl
        seq2seq.save_params("cal.pkl")
    else:
        lr /= 4.0
        optimizer.lr = lr

    seq2seq.reset_state()
    print('-' * 50)
trainer.plot()



# 테스트 데이터로 평가
seq2seq.reset_state()
# print(x_test)
# print( np.argmax(seq2seq.predict(x_test, True)) )
#print('테스트 퍼플렉서티: ', ppl_test)
