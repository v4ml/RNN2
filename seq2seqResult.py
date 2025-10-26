import sys
sys.path.append('..')
from dataset import sequence

from common.np import *
from layer_new import *
from dataset import ptb

from common import config
config.GPU = True


from seq2seq import Seq2seq

(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt', seed=1984)
x_test = np.array(x_test)
t_test = np.array(t_test)
x_train = np.array(x_train)
t_train = np.array(t_train)

x_train = x_train.get()
t_train = t_train.get()
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
seq2seq.load_params('cal.pkl')

ts = seq2seq.generate(x_train, 6, 5)

result = np.argmax(ts, axis=2)
for i in range(5000):
    for k, input in enumerate(x_train[i]):
        print(id_to_char[input], end = '')
    print(' = ', end='')
    for j, output in enumerate(result[i]):
        print(id_to_char[int(output)], end='')
    print(' / ', end='')
    for l, answer in enumerate(t_train[i]):           
        print(id_to_char[answer], end='')
    print('')