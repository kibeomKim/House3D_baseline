import os
import torch
import torch.multiprocessing as mp
import torch.optim as optim

import argparse

from train import run_sim
from shared_optim import SharedRMSprop, SharedAdam
from models import A3C_LSTM

targets = ['bedroom', 'kitchen', 'bathroom', 'dining_room', 'living_room']

class Params():
    def __init__(self):
        self.n_process = 15
        self.max_episode = 10000
        self.batch_size = 128
        self.gamma = 0.99
        self.gpu_ids = [0, 1, 2, 3]
        self.lr = 0.0001
        self.tau = 1.0
        self.seed = 1
        self.amsgrad = True
        self.num_steps = 20

def main():
    #parser = argparse.ArgumentParser()
    #parser.add_argument('obj')
    #parser.add_argument('--width', type=int, default=400)
    #parser.add_argument('--height', type=int, default=300)
    #args = parser.parse_args()

    params = Params()

    shared_model = A3C_LSTM(len(targets))
    shared_model = shared_model.share_memory()

    shared_optimizer = SharedAdam(shared_model.parameters(), lr=params.lr, amsgrad=params.amsgrad)
    shared_optimizer.share_memory()
    run_sim(0, params, shared_model, shared_optimizer)

    '''
    mp.set_start_method('spawn')
    processes = []
    #p = mp.Process(target=test, args=(args, params, shared_model,))
    #p.start()
    #processes.append(p)

    for rank in range(params.n_process):
        p = mp.Process(target=run_gym, args=(rank, args, params, shared_model, shared_optimizer,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    '''
if __name__ == "__main__":
    main()