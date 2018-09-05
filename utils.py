import numpy as np
import torch
import json
import logging

houses = ['00065ecbdd7300d35ef4328ffe871505','31966fdc9f9c87862989fae8ae906295', '8b8c1994f3286bfc444a7527ffacde86',
          '1dba3a1039c6ec1a3c141a1cb0ad0757', '492c5839f8a534a673c92912aedc7b63','e3ae3f7b32cf99b29d3c8681ec3be321',
          '5f3f959c7b3e6f091898caa8e828f110', '4383029c98c14177640267bd34ad2f3c', '0884337c703e7c25949d3a237101f060',
          'f10ce4008da194626f38f937fb9c1a03', 'e6f24af5f87558d31db17b86fe269cf2', 'b814705bc93d428507a516b866efda28',
          '26e33980e4b4345587d6278460746ec4', 'b5bd72478fce2a2dbd1beb1baca48abd', '9be4c7bee6c0ba81936ab0e757ab3d61',
          '2364b7dcc432c6d6dcc59dba617b5f4b', 'a7e248efcdb6040c92ac0cdc3b2351a6', '775941abe94306edc1b5820e3a992d75',
          'ff32675f2527275171555259b4a1b3c3', '32e53679b33adfcc5a5660b8c758cc96']
        # it didn't work
        #cf57359cd8603c3d9149445fb4040d90
        # 7995c2a93311717a3a9c48d789563590
        #32e53679b33adfcc5a5660b8c758cc96
        #ff32675f2527275171555259b4a1b3c3
        #775941abe94306edc1b5820e3a992d75
        #a7e248efcdb6040c92ac0cdc3b2351a6
        #2364b7dcc432c6d6dcc59dba617b5f4b

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)

def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object

def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x ** 2).sum(1, keepdim=True))
    return x

def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

def get_house_id(index):
    return houses[index]

def get_house_id_length():
    return len(houses)

def get_word_idx(word):
    word_to_idx = {"bedroom": 0, "kitchen": 1, "bathroom": 2, "dining_room": 3, "living_room": 4}
    return word_to_idx[word]