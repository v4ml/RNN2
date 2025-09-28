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

    def get_batch(self, xs, ys, start_idx, batch_size, time_size):
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
        for epoch in range(self.max_epoch):
            for iters in range(max_iters):
                batch_x, batch_y = self.get_batch(xs, ts, start_idx, batch_size, time_size)
                loss = self.model.forward(batch_x, batch_y)
                self.model.backward()
                if max_grad is not None:
                    self.clip_grads(self.model.grads, max_grad)
                self.optimizer.update(self.model.params, self.model.grads)
                total_loss += loss
                loss_count += 1
            
                # 퍼플렉서티 평가
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print('| 에폭 %d |  반복 %d / %d | 시간 %d[s] | 퍼플렉서티 %.2f'
                          % (epoch + 1, iters + 1, max_iters, elapsed_time, ppl))
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0         
            

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
            print('=================== clip')
            for grad in grads:
                grad *= rate        