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
actions = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#actions=[5, 6, 8, 11, 12]

def get_instruction_idx(instruction):
    instruction_idx = []
    for word in instruction.split(" "):
        instruction_idx.append(get_word_idx(word))
    instruction_idx = np.array(instruction_idx)

    instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)
    return instruction_idx


def test(rank, params, shared_model, count, lock, best_acc, evaluation=True):
    if not os.path.exists('./'+params.weight_dir):
        os.mkdir('./'+params.weight_dir)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    logging.basicConfig(filename='./log/'+params.log_file+'.log', level=logging.INFO)
    ptitle('Test Agent: {}'.format(rank))
    gpu_id = params.gpu_ids_test[rank % len(params.gpu_ids_test)]

    api = objrender.RenderAPI(w=params.width, h=params.height, device=gpu_id)
    cfg = load_config('config.json')
    best_rate = 0.0
    save_model_index = 0
    n_update = 0

    torch.manual_seed(params.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(params.seed + rank)

    model = A3C_LSTM_GA()
    with torch.cuda.device(gpu_id):
        model = model.cuda()

    Agent = run_agent(model, gpu_id)

    house_id = params.house_id
    if house_id == -1:
        house_id = rank
    if house_id >= 20:
        house_id = house_id % 20

    #time.sleep(rank*30)

    env = Environment(api, get_eval_house_id(house_id), cfg)
    task = RoomNavTask(env, hardness=params.hardness, segment_input=params.semantic_mode, max_steps=params.max_steps, discrete_action=True)     #reward_type='indicator'

    start_time = time.time()

    if evaluation is True:
        max_episode = params.max_episode
        n_try = params.n_eval
    else:
        max_episode = 1     # for loaded model test
        n_try = params.n_test

    for episode in range(max_episode):
        eval = []
        if evaluation is True:
            with lock:
                n_update = count.value
            with torch.cuda.device(gpu_id):
                Agent.model.load_state_dict(shared_model.state_dict())
        else:
            with torch.cuda.device(gpu_id):
                Agent.model.load_state_dict(shared_model)
        Agent.model.eval()

        for i in range(n_try):
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

            if evaluation is True:  # evaluation mode
                with lock:
                    #if best_acc.value >= best_rate:
                    #    best_rate = best_acc.value
                    if succ_rate >= best_rate:
                        best_rate = succ_rate
                        with torch.cuda.device(gpu_id):
                            torch.save(Agent.model.state_dict(), params.weight_dir + 'model' + str(n_update) + '.ckpt')
                        save_model_index += 1
                    #if best_rate > best_acc.value:
                    #    best_acc.value = best_rate

            avg_reward = sum([e[1] for e in eval]) / len(eval)
            avg_length = sum([e[0] for e in eval]) / len(eval)
            msg = " ".join([
                "++++++++++ Task Stats +++++++++++\n",
                "Time {}\n".format(time.strftime("%dd %Hh %Mm %Ss", time.gmtime(time.time() - start_time))),
                "Episode Played: {:d}\n".format(len(eval)),
                "N_Update = {:d}\n".format(n_update),
                "House id: {:d}\n".format(house_id),
                "Avg Reward = {:5.3f}\n".format(avg_reward),
                "Avg Length = {:.3f}\n".format(avg_length),
                "Best rate {:3.2f}, Success rate {:3.2f}%".format(best_rate, succ_rate)
            ])
            print(msg)
            logging.info(msg)