import sys
sys.path.append('..')
from common.np import *
from common.layers import *
from common.functions import *
import matplotlib.pyplot as plt
import time

class Trainer:
    def __init__(self, model, optimizer, xs, ts, batch_size, time_size, max_epoch):
        self.model = model
        self.optimizer = optimizer
        self.xs = xs
        self.ts = ts
        self.batch_size = batch_size
        self.time_size = time_size
        self.time_idx = 0
        self.ppl_list = None
        data_size = len(xs)
        jump = data_size // batch_size
        self.offset = [i*jump for i in range(batch_size)]
        self.max_epoch = max_epoch
        self.current_epoch = 0

    def get_batch(self, xs, ys, batch_size, time_size):
        start_idx = np.random.randint(0, 1000)
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_y = np.empty((batch_size, time_size), dtype='i')

        data_size = len(xs)
        for n, offset in enumerate(self.offset):
            for t in range(time_size):
                #print(offset+time_size*self.time_idx, offset+time_size*(self.time_idx+1))
                #print(xs[(offset+time_size*self.time_idx)%data_size : (offset+time_size*(self.time_idx+1))%data_size])
                #print((start_idx+offset+time_size*self.time_idx)%data_size, (start_idx+offset+time_size*(self.time_idx+1))%data_size)
                batch_x[n, t] = xs[(start_idx+offset+time_size*self.time_idx+t)%data_size]
                batch_y[n, t] = ys[(start_idx+offset+time_size*self.time_idx+t)%data_size]
        self.time_idx += 1

        return batch_x, batch_y
        # for t in range(time_size):
        #     for i, offset in enumerate(offset):
        #         batch_x[i, t] = xs[(self.time_idx +offset)%data_size]
        #         batch_y[i, t] = ts[(self.time_idx +offset)%data_size]
        #     self.time_idx += 1


    def fit(self, xs, ts, max_epoch=10, batch_size=32, time_size=20, max_grad=None, eval_interval=20):
        total_loss = 0
        loss_count = 0
        self.time_idx = 0
        start_idx = np.random.randint(0, 1000)
        max_iters = len(self.xs)//batch_size//time_size
        self.ppl_list = []

        start_time = time.time()
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_y = self.get_batch(xs, ts, batch_size, time_size)
                loss = self.model.forward(batch_x, batch_y)
                self.model.backward()
                params, grads = self.remove_duplicate(self.model.params, self.model.grads)

                if max_grad is not None:
                    self.clip_grads(self.model.grads, max_grad)
                self.optimizer.update(self.model.params, self.model.grads)
                total_loss += loss
                loss_count += 1
            
                # 퍼플렉서티 평가
                if (eval_interval is not None) and ((iters) % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print('| 에폭 %d |  반복 %d / %d | 시간 %d[s] | 퍼플렉서티 %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl))
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0         
            self.current_epoch += 1
            
    def fit_org(self, xs, ts, max_epoch=10, batch_size=20, time_size=35,
            max_grad=None, eval_interval=20):
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)
        self.time_idx = 0
        self.ppl_list = []
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                # 기울기를 구해 매개변수 갱신
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = self.remove_duplicate(model.params, model.grads)  # 공유된 가중치를 하나로 모음
                #params, grads = model.params, model.grads
                if max_grad is not None:
                    self.clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # 퍼플렉서티 평가
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print('| 에폭 %d |  반복 %d / %d | 시간 %d[s] | 퍼플렉서티 %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl))
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label='train')
        plt.xlabel('반복 (x' + str(self.eval_interval) + ')')
        plt.ylabel('퍼플렉서티')
        plt.show()
        

    def clip_grads(self, grads, max_norm):
        total_norm = 0
        for grad in grads:
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)

        rate = max_norm / (total_norm + 1e-6)
        if rate < 1:
            for grad in grads:
                grad *= rate


    def remove_duplicate(self, params, grads):
        '''
        매개변수 배열 중 중복되는 가중치를 하나로 모아
        그 가중치에 대응하는 기울기를 더한다.
        '''
        params, grads = params[:], grads[:]  # copy list

        while True:
            find_flg = False
            L = len(params)

            for i in range(0, L - 1):
                for j in range(i + 1, L):
                    # 가중치 공유 시
                    if params[i] is params[j]:
                        grads[i] += grads[j]  # 경사를 더함
                        find_flg = True
                        params.pop(j)
                        grads.pop(j)
                    # 가중치를 전치행렬로 공유하는 경우(weight tying)
                    elif params[i].ndim == 2 and params[j].ndim == 2 and \
                        params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                        grads[i] += grads[j].T
                        find_flg = True
                        params.pop(j)
                        grads.pop(j)

                    if find_flg: break
                if find_flg: break

            if not find_flg: break

        return params, grads
    
    def eval_perplexity(self, model, corpus, batch_size=10, time_size=35):
        print('퍼플렉서티 평가 중 ...')
        corpus_size = len(corpus)
        total_loss, loss_cnt = 0, 0
        max_iters = (corpus_size - 1) // (batch_size * time_size)
        jump = (corpus_size - 1) // batch_size

        for iters in range(max_iters):
            xs = np.zeros((batch_size, time_size), dtype=np.int32)
            ts = np.zeros((batch_size, time_size), dtype=np.int32)
            time_offset = iters * time_size
            offsets = [time_offset + (i * jump) for i in range(batch_size)]
            for t in range(time_size):
                for i, offset in enumerate(offsets):
                    xs[i, t] = corpus[(offset + t) % corpus_size]
                    ts[i, t] = corpus[(offset + t + 1) % corpus_size]

            try:
                loss = model.forward(xs, ts, train_flg=False)
            except TypeError:
                loss = model.forward(xs, ts)
            total_loss += loss

            sys.stdout.write('\r%d / %d' % (iters, max_iters))
            sys.stdout.flush()

        print('')
        ppl = np.exp(total_loss / max_iters)
        return ppl    