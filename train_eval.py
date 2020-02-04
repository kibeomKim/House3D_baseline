import torch
from torch.autograd import Variable
import torch.optim as optim

from House3D import objrender, Environment, load_config
from House3D.roomnav import RoomNavTask

from models import A3C_LSTM_GA
from agent import run_agent
from utils import get_house_id, get_word_idx

import os
import pdb
from setproctitle import setproctitle as ptitle
import time
import random
import numpy as np
from collections import deque
import logging

targets = ['bedroom', 'kitchen', 'bathroom', 'dining_room', 'living_room']
actions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#actions=[5, 6, 8, 11, 12]

def get_instruction_idx(instruction):
    instruction_idx = []
    for word in instruction.split(" "):
        instruction_idx.append(get_word_idx(word))
    instruction_idx = np.array(instruction_idx)

    instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)
    return instruction_idx

def run_sim(rank, params, shared_model, shared_optimizer, count, lock):
    if not os.path.exists('./'+params.weight_dir):
        os.mkdir('./'+params.weight_dir)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    logging.basicConfig(filename='./log/'+params.log_file+'.log', level=logging.INFO)

    ptitle('Training Agent: {}'.format(rank))
    gpu_id = params.gpu_ids_train[rank % len(params.gpu_ids_train)]
    api = objrender.RenderAPI(w=params.width, h=params.height, device=gpu_id)
    cfg = load_config('config.json')

    torch.manual_seed(random.randint(0, 1000) + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(random.randint(0, 1000) + rank)

    model = A3C_LSTM_GA()
    with torch.cuda.device(gpu_id):
        model = model.cuda()

    Agent = run_agent(model, gpu_id)
    house_id = params.house_id

    if house_id == -1:
        house_id = rank
    if house_id > 50:
        house_id = house_id % 50

    env = Environment(api, get_house_id(house_id, params.difficulty), cfg)
    task = RoomNavTask(env, hardness=params.hardness, segment_input=params.semantic_mode,
                       max_steps=params.max_steps, discrete_action=True)

    n_train = 0
    best_rate = 0.0
    save_model_index = 0

    while True:
        n_train += 1
        training(task, gpu_id, shared_model, Agent, shared_optimizer, params, lock, count)

        if n_train % 1000 == 0:
            with lock:
                n_update = count.value
            with torch.cuda.device(gpu_id):
                Agent.model.load_state_dict(shared_model.state_dict())

            start_time = time.time()
            best_rate, save_model_index = testing(lock, n_update, gpu_id, Agent, task, best_rate, params, save_model_index, start_time, logging, house_id)

def training(task, gpu_id, shared_model, Agent, optimizer, params, lock, count):

    next_observation = task.reset()
    target = task.info['target_room']
    target = get_instruction_idx(target)

    with torch.cuda.device(gpu_id):
        target = Variable(torch.LongTensor(target)).cuda()
        Agent.model.load_state_dict(shared_model.state_dict())
        Agent.cx = Variable(torch.zeros(1, 256).cuda())
        Agent.hx = Variable(torch.zeros(1, 256).cuda())
        Agent.target = target

    total_reward, num_steps, good = 0, 0, 0
    Agent.done = False
    done = False

    while not done:
        num_steps += 1
        observation = next_observation
        act, entropy, value, log_prob = Agent.action_train(observation, target)
        next_observation, reward, done, info = task.step(actions[act[0]])

        rew = np.clip(reward, -1.0, 10.0)
        if rew != -1.0 and rew != 10.0:     # make sparse reward
            rew = 0.0

        Agent.put_reward(rew, entropy, value, log_prob)
        if done:    # num_steps % params.num_steps == 0 or
            if done:
                Agent.done = done
            with lock:
                count.value += 1
            Agent.training(next_observation, shared_model, optimizer, params)

            break

def testing(lock, n_update, gpu_id, Agent, task, best_rate, params, save_model_index, start_time, logging, house_id):

    eval = []
    Agent.model.eval()

    for i in range(params.n_eval):
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

    return best_rate, save_model_index