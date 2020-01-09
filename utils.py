import numpy as np
import torch
import json
import logging

houses = ["00065ecbdd7300d35ef4328ffe871505", "cf57359cd8603c3d9149445fb4040d90", "ff32675f2527275171555259b4a1b3c3",
          "775941abe94306edc1b5820e3a992d75", "7995c2a93311717a3a9c48d789563590", "8b8c1994f3286bfc444a7527ffacde86",
          "31966fdc9f9c87862989fae8ae906295", "32e53679b33adfcc5a5660b8c758cc96", "4383029c98c14177640267bd34ad2f3c",
          "0884337c703e7c25949d3a237101f060", "492c5839f8a534a673c92912aedc7b63", "a7e248efcdb6040c92ac0cdc3b2351a6",
          "2364b7dcc432c6d6dcc59dba617b5f4b", "e3ae3f7b32cf99b29d3c8681ec3be321", "f10ce4008da194626f38f937fb9c1a03",
          "e6f24af5f87558d31db17b86fe269cf2", "1dba3a1039c6ec1a3c141a1cb0ad0757", "b814705bc93d428507a516b866efda28",
          "26e33980e4b4345587d6278460746ec4", "5f3f959c7b3e6f091898caa8e828f110"]

house_test = ["00a42e8f3cb11489501cfeba86d6a297", "09160de615ffdd69d8a9662a46021d29", "0a96348d9c8acf673d3da07b6316e671",
              "0635535a9980bcd4a311464cad45fda5", "02f594bb5d8c5871bde0d8c8db20125b", "b5bd72478fce2a2dbd1beb1baca48abd",
              "02d83f79d7c3311ccc3395bbf2ea4ae4", "0762f81764e4fdd31a3410ebb89f59bc", "08f65a7829871e9399c38a261cdd8be0",
              "04667f43ca426693515b7da9befed6a0", "0a725e073c3a0fd0b0f5a4d126990fda", "04a6f8d6f031d39cd8f52743c08b5fb9",
              "09d26570eeb14c0976a32be3b243e40e", "08e89905e0a41614aa4f85109d362c1d", "0bda523d58df2ce52d0a1d90ba21f95c",
              "0b8755a7f00f2d47e246a6384b6c9b8e", "04af8d2c8883e26e9227e169f9a383f2", "0acf79836db830174c202d3a93a6b14a",
              "01a00a54b07c67729af0c4f5bdb91ccf", "0257400a5ae18e68196835f1e005740c", "03620d135a491a635c198e188e927ba0",
              "06840849cabbdc4bb9f3069242e1f587", "0b6d4fe900eaddd80aecf4bc79248dd9", "0017aeff679f53cd65edf72ef2349ff1",
              "0c05dbdef4ee21dc770e5be2f471dc35", "0642bcc7bd4c964830446b700ed4b5d5", "00cdcd4541a2145d004bbd45ee658f66",
              "09f8b9e6f42002c30d061c3f592d9685", "0acbb6234652a949f52e5b468289226c", "0a578e53e06ac0178bac608e74e51218",
              "066807814a14cf68d58a92792f162a9c", "03353fe273b81f93a11285c759e8a98b", "0c1f9e71298200e948bbee2d67faf578",
              "05e17d97e1be878ef08d963b5344b969", "09e46cb7bc972db216e7e5ba0ee4250f", "0b3ee5eefc0a664a153600f321a7b276",
              "023e21673caa7ac3f2808b498ab4cde8", "047dbcc70a693132ae860b9c73741483", "06f7826572be27f205e701783960416e",
              "072d8bce0ebe90d5c757ca280b633bd5", "09c1b600e4942ef3ba3ca6d940fa7a36", "034e4c22e506f89d668771831080b291",
              "0375e18d4664786745786988af6cfbf7", "09390811c51225aaced1ae50c6e6cecc", "035323ee3330e84be752322e598a24cc",
              "0267d23f3a3888693a11f33506e7f2d4", "092862153228f7466b1f911160e68c41", "09338eb45a9b1b09caeb317bb2b18baa",
              "0c13c5a3cee7ddc40201ab2e765c8dc7", "0a417c6459befd8a9fa4a5428f2de1e9"]

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

def get_eval_house_id(index):
    return house_test[index]

def get_house_id_length():
    return len(houses)

def get_word_idx(word):
    word_to_idx = {"bedroom": 0, "kitchen": 1, "bathroom": 2, "dining_room": 3, "living_room": 4}
    return word_to_idx[word]