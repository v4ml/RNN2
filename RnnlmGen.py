from lm import Lm
from better_rnnlm import BetterRnnlm
from common.np import *
from dataset import ptb
 
#class LmGen(Lm):
class LmGen(BetterRnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        # word_ids = [start_id]

        # while len(word_ids) < sample_size:
        #     word_ids.append( np.argmax(self.predict(start_id)) )

        # return word_ids

        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x)
            p = self.softmax(score.flatten())

            sampled = np.argmax(p)
            #sampled = np.random.choice(len(p), size=1, p=p)[0]
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids
    
    def softmax(self, x):
        if x.ndim == 2:
            x = x - x.max(axis=1, keepdims=True)
            x = np.exp(x)
            x /= x.sum(axis=1, keepdims=True)
        elif x.ndim == 1:
            x = x - np.max(x)
            x = np.exp(x) / np.sum(np.exp(x))

        return x    

batch_size = 20
wordvec_size = 650
hidden_size = 650
time_size = 35
lr = 20.0
max_epoch = 40
max_grad = 0.25
dropout = 0.5
eval_interval = 20

corpus, word_to_id, id_to_word = ptb.load_data('train')


#lm = Lm(vocab_size=10000, wordvec_size=wordvec_size, hidden_size=hidden_size, dropout_ratio=dropout)
lmGen = LmGen(vocab_size=10000, wordvec_size=wordvec_size, hidden_size=hidden_size, dropout_ratio=dropout)
lmGen.load_params('./BetterRnnlm.pkl')

start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$']
skip_ids = [word_to_id[w] for w in skip_words]


word_ids = lmGen.generate(start_id, skip_ids)
txt = ' '.join([id_to_word[i] for i in word_ids] )
txt = txt.replace('<eos>', '.\n')

print(txt)

#for i, elem in enumerate(word_ids):
#   print(id_to_word[elem], end=" ")