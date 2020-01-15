import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.multiprocessing as mp

import argparse
import logging
import time

from models import A3C_LSTM_GA
from test import run_test

class Params():
    def __init__(self):
        self.n_process = 20

        self.gpu_ids_test = [0, 1, 2, 3]
        self.seed = 1
        self.hardness = 0.6
        self.width = 120
        self.height = 90
        self.n_test = 2000
        self.max_steps = 100
        self.semantic_mode = False  #if false, RGB mode on
        self.log_file = 'baseline_dense_RGB-D_0113_test'
        self.weight_dir = './baseline_dense_RGB-D_0113/'

def main():
    params = Params()

    if not os.path.exists('./log'):
        os.mkdir('./log')

    logging.basicConfig(filename='./log/' + params.log_file + '.log', level=logging.INFO)

    mp.set_start_method('spawn')

    test_files = ['model7950657.ckpt', 'model7783645.ckpt', 'model7762120.ckpt', 'model7695404.ckpt', 'model7582072.ckpt',
                  'model7519625.ckpt', 'model7506695.ckpt', 'model7410288.ckpt', 'model7118813.ckpt', 'model7009886.ckpt',
                  'model6912503.ckpt', 'model6760726.ckpt', 'model6635764.ckpt', 'model6622441.ckpt', 'model6577278.ckpt',
                  'model6546240.ckpt', 'model6515106.ckpt', 'model6359619.ckpt', 'model6149225.ckpt', 'model6146211.ckpt',
                  'model6137830.ckpt', 'model6122163.ckpt', 'model6090876.ckpt', 'model6089290.ckpt', 'model6060824.ckpt']

    for ckpt in test_files:
        init_msg = " ".join([
            "\n\n++++++++++++++++++++ Initial Task info +++++++++++++++++++++\n",
            "weight file name = {:s}\n".format(ckpt)
        ])
        print(init_msg)
        logging.info(init_msg)

        seen_succ = mp.Value('i', 0)
        seen_length = mp.Value('i', 0)
        unseen_succ = mp.Value('i', 0)
        unseen_length = mp.Value('i', 0)
        lock = mp.Lock()

        # with lock:  # initialize, is it right?
        #     seen_succ = 0
        #     seen_length = 0
        #     unseen_succ = 0
        #     unseen_length = 0

        load_model = params.weight_dir + ckpt
        # load_model = torch.load(params.weight_dir + ckpt)

        #test(params, shared_model, count, lock, best_acc)

        processes = []

        test_process = 0

        for rank in range(params.n_process):
            p = mp.Process(target=run_test, args=(test_process, params, load_model, lock, seen_succ, seen_length, unseen_succ, unseen_length, ))
            test_process += 1
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        msg = " ".join([
            "++++++++++++++++++++ Total Task Stats +++++++++++++++++++++\n",
            "Seen Avg Length = {:.3f}\n".format(seen_length.value/(20 * params.n_test)),
            "Seen Total Success rate {:3.2f}%".format(100 * seen_succ.value/(20 * params.n_test)),
            "UnSeen Avg Length = {:.3f}\n".format(unseen_length.value/(50 * params.n_test)),
            "UnSeen Total Success rate {:3.2f}%\n\n".format(100 * unseen_succ.value/(50 * params.n_test)),
        ])
        print(msg)
        logging.info(msg)

    print("Done")


# def run_test(type):
#     params = Params()
#
#     #'./weight_one_hot/model3.ckpt' is best
#     # if type == 1:
#     #     load_model = torch.load('./weight_0810/model46301.ckpt', map_location=lambda storage, loc: storage.cuda(params.gpu_ids_test[0]))
#     # else:
#     #     load_model = None
#     test(0, params, load_model, None, None, None, evaluation=False)


if __name__ == "__main__":
    main()