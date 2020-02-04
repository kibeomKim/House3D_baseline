import numpy as np
import torch
import json
import logging

# houses = ["00065ecbdd7300d35ef4328ffe871505", "cf57359cd8603c3d9149445fb4040d90", "ff32675f2527275171555259b4a1b3c3",
#           "775941abe94306edc1b5820e3a992d75", "7995c2a93311717a3a9c48d789563590", "8b8c1994f3286bfc444a7527ffacde86",
#           "31966fdc9f9c87862989fae8ae906295", "32e53679b33adfcc5a5660b8c758cc96", "4383029c98c14177640267bd34ad2f3c",
#           "0884337c703e7c25949d3a237101f060", "492c5839f8a534a673c92912aedc7b63", "a7e248efcdb6040c92ac0cdc3b2351a6",
#           "2364b7dcc432c6d6dcc59dba617b5f4b", "e3ae3f7b32cf99b29d3c8681ec3be321", "f10ce4008da194626f38f937fb9c1a03",
#           "e6f24af5f87558d31db17b86fe269cf2", "1dba3a1039c6ec1a3c141a1cb0ad0757", "b814705bc93d428507a516b866efda28",
#           "26e33980e4b4345587d6278460746ec4", "5f3f959c7b3e6f091898caa8e828f110"]
#
# house_test = ["00a42e8f3cb11489501cfeba86d6a297", "09160de615ffdd69d8a9662a46021d29", "0a96348d9c8acf673d3da07b6316e671",
#               "0635535a9980bcd4a311464cad45fda5", "02f594bb5d8c5871bde0d8c8db20125b", "b5bd72478fce2a2dbd1beb1baca48abd",
#               "02d83f79d7c3311ccc3395bbf2ea4ae4", "0762f81764e4fdd31a3410ebb89f59bc", "08f65a7829871e9399c38a261cdd8be0",
#               "04667f43ca426693515b7da9befed6a0", "0a725e073c3a0fd0b0f5a4d126990fda", "04a6f8d6f031d39cd8f52743c08b5fb9",
#               "09d26570eeb14c0976a32be3b243e40e", "08e89905e0a41614aa4f85109d362c1d", "0bda523d58df2ce52d0a1d90ba21f95c",
#               "0b8755a7f00f2d47e246a6384b6c9b8e", "04af8d2c8883e26e9227e169f9a383f2", "0acf79836db830174c202d3a93a6b14a",
#               "01a00a54b07c67729af0c4f5bdb91ccf", "0257400a5ae18e68196835f1e005740c", "03620d135a491a635c198e188e927ba0",
#               "06840849cabbdc4bb9f3069242e1f587", "0b6d4fe900eaddd80aecf4bc79248dd9", "0017aeff679f53cd65edf72ef2349ff1",
#               "0c05dbdef4ee21dc770e5be2f471dc35", "0642bcc7bd4c964830446b700ed4b5d5", "00cdcd4541a2145d004bbd45ee658f66",
#               "09f8b9e6f42002c30d061c3f592d9685", "0acbb6234652a949f52e5b468289226c", "0a578e53e06ac0178bac608e74e51218",
#               "066807814a14cf68d58a92792f162a9c", "03353fe273b81f93a11285c759e8a98b", "0c1f9e71298200e948bbee2d67faf578",
#               "05e17d97e1be878ef08d963b5344b969", "09e46cb7bc972db216e7e5ba0ee4250f", "0b3ee5eefc0a664a153600f321a7b276",
#               "023e21673caa7ac3f2808b498ab4cde8", "047dbcc70a693132ae860b9c73741483", "06f7826572be27f205e701783960416e",
#               "072d8bce0ebe90d5c757ca280b633bd5", "09c1b600e4942ef3ba3ca6d940fa7a36", "034e4c22e506f89d668771831080b291",
#               "0375e18d4664786745786988af6cfbf7", "09390811c51225aaced1ae50c6e6cecc", "035323ee3330e84be752322e598a24cc",
#               "0267d23f3a3888693a11f33506e7f2d4", "092862153228f7466b1f911160e68c41", "09338eb45a9b1b09caeb317bb2b18baa",
#               "0c13c5a3cee7ddc40201ab2e765c8dc7", "0a417c6459befd8a9fa4a5428f2de1e9"]

hard_train = ['0afc1aebcfdeaf2778130397d0ab5247', '096f326a58d25c51089ff62f17b0474b', '0689adf36dbcc7a5146dc77f49b35ab7', '0a390dfb0968eec0bd699c36cf66eb42', '03596b14e0adde7d58e1b844da3d0a4b', '03fe54a2ecb835c38a284eebca01ca91', '093489c7853f42856439e72996cbb535', '048f9cf7d5677ffc712af40e27da0b1c', '0c91d9762d2ebbfc8a9747494e34ffaf', '04f4590d85e296b4c81c5a62f8a99bce', '06cfb97c9d953931040c6e2398e1071c', '8b8c1994f3286bfc444a7527ffacde86', '06c69199f72c9479dafd61eb6d8f418e', '02192418f915ae730c44fe3d37714729', '013365aedf1f2a74e61ef1f23c1764f2', '02feffee246a02bef072edc2c5e40804', '0b8c275d46743f4a948bd34f9b38fd08', '03d4c5b9b91f6e84b109fcc1df6d56eb', '052577edcdb2eb95ae7a221adbd595b7', '0ad80e444d6639d9682af10fc78ee729', '0adb05c478b044265bca443ca6e4ba56', '06a47f2806c29f0d07f1b61937c96b64', '0215271c2af473e57f51ca210eb9b8ed', '05b1c4ffdef776b76c02cdc560dbe23e', '06e1a42f7726ff9aabdf0dc8f9d4339d', '0394080c68c972e8dd56ce591f711aed', '0759f0d8f141ad39f4ef51a1954d87ff', '04d8ca38df7023cef765ff40d33a444d', '05e17d97e1be878ef08d963b5344b969', '0b9285c9f090123caaae85500d48ea8f', '06840849cabbdc4bb9f3069242e1f587', '0bf2b5a34121c69841e8c83d73691ccb', '046f8274faf038cc382b696f71908155', '088c62cb5417ff5db3b4314679d5a53b', '067395d0c2b7367cba199c5bad513491', '035323ee3330e84be752322e598a24cc', '09e9b0cd1f5b6fa3030d4ac9d6f9487d', '08348339da2ada4526b629d2e725fbe6', '0c90efff2ab302c6f31add26cd698bea', '0760f8574d23d5293708098b1d8841eb', '0a64e00205668eed85721e184b0ee2ae', '0ae1427b2db0b46aca0088e95524a529', '00a42e8f3cb11489501cfeba86d6a297', '026c1bca121239a15581f32eb27f2078', '0821ab76519bafe9b2158f176ba3c453', '00b5dd6017b5a5f51e359db09b8d42e0', '01f0807e9cee19a44668362fe089599f', '00052c0562bde7790f8354e6123ae7ff', '05cac5f7fdd5f8138234164e76a97383', '00cdcd4541a2145d004bbd45ee658f66']
hard_test = ['0a725e073c3a0fd0b0f5a4d126990fda', '00065ecbdd7300d35ef4328ffe871505', '0c9a666391cc08db7d6ca1a926183a76', '04d9e6836dd93e9bb29103663dc49098', '09f8b9e6f42002c30d061c3f592d9685', '04197c453e9ce6cc83d70bb04fad45c4', '04af8d2c8883e26e9227e169f9a383f2', '05243b502288a465b4b0e47e88d32f50', '016370d13c9f967d41d133cf93a42b75', '056d6d802946546b2b96123b88a24017', '049f4d9352fcbc72eb99c387e2fc8d45', '022a862db052d1fe2c14e9e5f819ee5e', '006ab253a81b9cd33ce8f94c6865af81', '04dc8d572d6caac1b795f0393f80989b', '0bba0568cf6541a185c7746f86036d1a', '5f3f959c7b3e6f091898caa8e828f110', '0102b1a5299fcd7efefabb58d89cc609', '0a75817da52080fc9d8735f92ec75b1c', '03761e5d9257855349ceb68e32ed9d67', '08e89905e0a41614aa4f85109d362c1d', '05679b634e77fb313821238605840ef2', '047dbcc70a693132ae860b9c73741483', '0796058ceedc30c80a635699fa87617d', '0c7ed9b15204173a493ce77f150a398f', '0696922a8279c10c1f0540ae7e172e83', '07cb50988d4679e53aec95eb00c3ba15', '0a578e53e06ac0178bac608e74e51218', '09179be43fcbecf5a1d811bfd3bc29cb', '0b79aa29e4b1dfdf3dd68345e298e907', '0134748166d15e65324126a5fd02e8e4']
easy_train = ['0c55147a28385f41917919cc41068a25', '04a5c943032eb0b6e2c8e2738ca3e9c4', '0b8755a7f00f2d47e246a6384b6c9b8e', '0aca661702f546f161effb91c4a3be3a', '00cfe094634578865b4384f3adef49e6', '0c31715e3273dae1b18497b63e505412', '02d83f79d7c3311ccc3395bbf2ea4ae4', '0c834166cf31d370628d21a67bedb92d', '0bf0afab8749a4ee89c61f61fbd5c19b', '07d1d46444ca33d50fbcb5dc12d7c103', '0b3ee5eefc0a664a153600f321a7b276', '034e4c22e506f89d668771831080b291', '0956f45246eb18461f217561934cf194', '0b3f9c911b06ddc08ed7ef41ac8d6993', '06f7826572be27f205e701783960416e', '09e46cb7bc972db216e7e5ba0ee4250f', '023add09d70b99f99b6e652b02246e4b', '052ac7d0b5d1e76b971ef0e78fc344da', '040f07adfbcd0048d29ae8ea72b2a9a5', '03261a9a6277e868ffdcba0ac54040ab', '03353fe273b81f93a11285c759e8a98b', '0c191ec1ba4bd40bf0b6d459e3eb2674', '0635535a9980bcd4a311464cad45fda5', '32e53679b33adfcc5a5660b8c758cc96', '0a5090e00368c2a42bfc67a61c6ecad4', '09338eb45a9b1b09caeb317bb2b18baa', '03620d135a491a635c198e188e927ba0', '0a417c6459befd8a9fa4a5428f2de1e9', '00d12ac6157f621c33d3118acee48863', '02130fb14431f31039955d8b02b976a4', '05387e0d468217ee88cd284fded6d79f', '062305b201cd17d2ac824f37ab59d64d', '05fc50487eb1da4939e5a4905b1776ea', '0255592477ea2f93711c76ef346e789f', '02fa7412c634cd525c9747c5e9e834c3', '065488ae866d96a01a8a8281caa0ce0b', '04a6f8d6f031d39cd8f52743c08b5fb9', '0b0af52ec3cfa00d293ed4083062b2e0', '03a1264a60331bf8abb1c9779bf05104', '0a5120ae90dba6afbb0407bd1daae9b8', '00d9be7210856e638fa3b1addf2237d6', '053ba1f84dccd916b6b607f435d991e1', '09d26570eeb14c0976a32be3b243e40e', '09ce8b1dc87d421024fd5e04604e720f', '035feb0125d3cc8ccc6d7281c02fbbfb', '096f7b959262764e8af648ad31017b88', '0ae8ad07c9913d7a16bf3fcb56c12030', '012a8c5a11383d705c1ddb9fee554860', '0acb7063368807464364e8fe03ea78b5', '0642bcc7bd4c964830446b700ed4b5d5']
easy_test = ['0a0639e9bc2d801476658f702c993641', '0147a1cce83b6089e395038bb57673e3', '0c7a76752621b73f85cb6e1c162831e0', '08fd6242edbc91788997ccf5fe2cf16b', '014d4eb144276041957e3b7021671981', '0017aeff679f53cd65edf72ef2349ff1', '0880799c157b4dff08f90db221d7f884', '094fb5b72c212420f3f2abae2317e0bc', '092862153228f7466b1f911160e68c41', '0ac5e76ced23481102476af66e539c75', 'b5bd72478fce2a2dbd1beb1baca48abd', '03aca3988338f63633357ee73e195266', '06798e87b1838acf228a42582b008728', '00ccd3bdb38bc4f1b115ee422cc10660', '0a0b9b45a1db29832dd84e80c1347854', '0601a680273d980b791505cab993096a', '0c05dbdef4ee21dc770e5be2f471dc35', '0bda523d58df2ce52d0a1d90ba21f95c', '069f29cca2e0c17ce73942a8c08ce83d', '0b1346dea27ec954f21c359d1001ce50', '0bab7ecf049983288beb908b9fb80664', '0369cb6417cea8636d6a97f4191840fa', '0375e18d4664786745786988af6cfbf7', '08ac2c76c9397f8944913c27137bbe37', '0c39bf47718aa1fabeb890680f93c359', '066807814a14cf68d58a92792f162a9c', '02f594bb5d8c5871bde0d8c8db20125b', '014cdfc6ba22cf87d8aa5249662766ef', '05fbb325a0809a06568644431c117a5e', '0422eaa085c2fef65e4b116ac9455101']


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

def get_house_id(index, difficulty=True):
    if difficulty:
        return easy_train[index]
    else:
        return hard_train[index]

def get_eval_house_id(index, difficulty=True):
    if difficulty:
        return easy_test[index]
    else:
        return hard_test[index]

def get_house_id_length():
    return len(houses)

def get_word_idx(word):
    word_to_idx = {"bedroom": 0, "kitchen": 1, "bathroom": 2, "dining_room": 3, "living_room": 4}
    return word_to_idx[word]