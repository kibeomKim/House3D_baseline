import torch
from torch.autograd import Variable
import torch.optim as optim

from House3D import objrender, Environment, load_config
from House3D.roomnav import RoomNavTask

from models import A3C_LSTM_GA
from agent import run_agent
from utils import get_house_id, get_house_id_length, get_word_idx

import pdb
from setproctitle import setproctitle as ptitle
import time
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
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = params.gpu_ids_train[rank % len(params.gpu_ids_train)]
    api = objrender.RenderAPI(w=params.width, h=params.height, device=gpu_id)
    cfg = load_config('config.json')

    if shared_optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(),lr=params.lr, amsgrad=params.amsgrad,
                                      weight_decay=params.weight_decay)
        #optimizer.share_memory()
    else:
        optimizer = shared_optimizer

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
    if house_id > 20:
        house_id = house_id % 20

    env = Environment(api, get_house_id(house_id), cfg)
    task = RoomNavTask(env, hardness=params.hardness, segment_input=params.semantic_mode,
                       max_steps=params.max_steps, discrete_action=True)

    for episode in range(params.max_episode):        
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
            if num_steps % params.num_steps == 0 or done:
                if done:
                    Agent.done = done
                with lock:
                    count.value += 1
                Agent.training(next_observation, shared_model, optimizer, params)

            if done:
                break