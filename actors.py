import torch
from torch.autograd import Variable
import torch.optim as optim

from House3D import objrender, Environment, load_config
from House3D.roomnav import RoomNavTask

from models import A3C_LSTM_GA
from agent import run_agent
from utils import get_house_id, get_word_idx

import pdb
from setproctitle import setproctitle as ptitle
import time
import numpy as np
from collections import deque
import logging

targets = ['bedroom', 'kitchen', 'bathroom', 'dining_room', 'living_room']
action_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def get_instruction_idx(instruction):
    instruction_idx = []
    for word in instruction.split(" "):
        instruction_idx.append(get_word_idx(word))
    instruction_idx = np.array(instruction_idx)

    instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)
    return instruction_idx

def run_sim(rank, params, state_Queue, action_done, actions, reward_Queue, lock):
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = params.gpu_ids_train[rank % len(params.gpu_ids_train)]
    api = objrender.RenderAPI(w=params.width, h=params.height, device=gpu_id)
    cfg = load_config('config.json')

    house_id = params.house_id

    if house_id == -1:
        house_id = rank
    if house_id > 50:
        house_id = house_id % 50

    env = Environment(api, get_house_id(house_id, params.difficulty), cfg)
    task = RoomNavTask(env, hardness=params.hardness, segment_input=params.semantic_mode,
                       max_steps=params.max_steps, discrete_action=True)

    while True:
        next_observation = task.reset()
        target = task.info['target_room']
        target = get_instruction_idx(target)

        # with torch.cuda.device(gpu_id):
        #     target = Variable(torch.LongTensor(target)).cuda()

        total_reward, num_steps, good = 0, 0, 0
        done = False
        test = False

        while not done:
            num_steps += 1
            observation = next_observation
            state = rank, [observation, target]
            state_Queue.put(state)

            state_Queue.join()

            # action_done.get()   # action done
            action = actions[rank]
            if action == 99:
                test = True
                break   # call for test

            next_observation, reward, done, info = task.step(action)

            reward = np.clip(reward, -1.0, 10.0)
            if reward != -1.0 and reward != 10.0:   # make sparse reward
                reward = 0.0
            total_reward += reward
           
            rew = [rank, done, reward]
            # print("send - rank: {:d}, reward: {:3.2f}".format(rank, reward))
            reward_Queue.put(rew)

            reward_Queue.join()

            if done:
                break