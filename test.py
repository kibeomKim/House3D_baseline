import torch
from torch.autograd import Variable

from House3D import objrender, Environment, load_config
from House3D.roomnav import RoomNavTask

from models import A3C_LSTM_GA
from agent import run_agent
from utils import get_house_id, get_house_id_length, get_word_idx, setup_logger, get_eval_house_id

import pdb
import os
from setproctitle import setproctitle as ptitle
import time
import numpy as np
import logging

targets = ['bedroom', 'kitchen', 'bathroom', 'dining_room', 'living_room']
actions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def get_instruction_idx(instruction):
    instruction_idx = []
    for word in instruction.split(" "):
        instruction_idx.append(get_word_idx(word))
    instruction_idx = np.array(instruction_idx)

    instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)
    return instruction_idx


def run_test(rank, params, loaded_model, lock, seen_succ, seen_length, unseen_succ, unseen_length):

    logging.basicConfig(filename='./log/'+params.log_file+'.log', level=logging.INFO)
    ptitle('Test Agent: {}'.format(rank))
    gpu_id = params.gpu_ids_test[rank % len(params.gpu_ids_test)]

    api = objrender.RenderAPI(w=params.width, h=params.height, device=gpu_id)
    cfg = load_config('config.json')

    torch.manual_seed(params.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(params.seed + rank)

    load_model = torch.load(loaded_model, map_location=lambda storage, loc: storage.cuda(gpu_id))

    model = A3C_LSTM_GA()
    with torch.cuda.device(gpu_id):
        model = model.cuda()
        model.load_state_dict(load_model)
        model.eval()

    Agent = run_agent(model, gpu_id)

    n_test = 0
    start_time = time.time()

    while True:
        house_id = rank + (n_test * params.n_process)

        if house_id >= 70:
            break
        else:
            if house_id < 20:
                seen = True
                house = get_house_id(house_id)
            else:
                seen = False
                house = get_eval_house_id(house_id - (n_test * params.n_process))

        env = Environment(api, house, cfg)
        task = RoomNavTask(env, hardness=params.hardness, segment_input=params.semantic_mode, max_steps=params.max_steps, discrete_action=True)     #reward_type='indicator'

        eval = []
        for i in range(params.n_test):
            next_observation = task.reset()
            target = task.info['target_room']
            target = get_instruction_idx(target)

            with torch.cuda.device(gpu_id):
                target = Variable(torch.LongTensor(target)).cuda()
                Agent.cx = Variable(torch.zeros(1, 256).cuda())
                Agent.hx = Variable(torch.zeros(1, 256).cuda())
                Agent.target = target
            step, total_rew, good = 0, 0, 0
            done = False

            while not done:
                observation = next_observation
                act = Agent.action_test(observation, target)

                next_observation, rew, done, info = task.step(actions[act[0]])
                total_rew += rew

                if rew == 10:   # success
                    good = 1

                step += 1

                if done:
                    break
            eval.append((step, total_rew, good))

        if len(eval) > 0:
            succ = [e for e in eval if e[2] > 0]
            succ_rate = (len(succ) / len(eval)) * 100

            avg_reward = sum([e[1] for e in eval]) / len(eval)
            avg_length = sum([e[0] for e in eval]) / len(eval)
            if seen:
                msg_seen = "Seen"
                msg_house = house_id
            else:
                msg_seen = "Unseen"
                msg_house = house_id - 20

            msg = " ".join([
                "++++++++++ Task Stats +++++++++++\n",
                "Time {}\n".format(time.strftime("%dd %Hh %Mm %Ss", time.gmtime(time.time() - start_time))),
                "Episode Played: {:d}\n".format(len(eval)),
                "{:s} House id: {:d}\n".format(msg_seen, msg_house),
                "Avg Reward = {:5.3f}\n".format(avg_reward),
                "Avg Length = {:.3f}\n".format(avg_length),
                "Success rate {:3.2f}%".format(succ_rate)
            ])
            print(msg)
            logging.info(msg)
            with lock:
                if seen:
                    seen_succ += len(succ)
                    seen_length += sum([e[0] for e in eval])
                else:
                    unseen_succ += len(succ)
                    unseen_length += sum([e[0] for e in eval])
            n_test += 1
