import torch
from torch.autograd import Variable

from House3D import objrender, Environment, load_config
from House3D.roomnav import RoomNavTask

from models import A3C_LSTM
from agent import run_agent
from utils import get_house_id, get_house_id_length

import pdb
from setproctitle import setproctitle as ptitle
import time
import numpy as np
from collections import deque

targets = ['bedroom', 'kitchen', 'bathroom', 'dining_room', 'living_room']
actions = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def run_sim(rank, params, shared_model, shared_optimizer):
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = params.gpu_ids[rank % len(params.gpu_ids)]
    api = objrender.RenderAPI(w=400, h=300, device=gpu_id)
    cfg = load_config('config.json')
    house_id = -1
    succ = deque(maxlen=500)

    for episode in range(params.max_episode):
        house_id += 1
        if house_id % get_house_id_length() == 0:
            house_id = 0
        env = Environment(api, get_house_id(house_id), cfg)
        task = RoomNavTask(env, hardness=0.6, discrete_action=True)

        model = A3C_LSTM(len(targets))

        with torch.cuda.device(gpu_id):
            model = model.cuda()
            model.load_state_dict(shared_model.state_dict())
        Agent = run_agent(model, gpu_id)

        #pdb.set_trace()

        next_observation = task.reset()
        target = task.info['target_room']
        target = [1 if targets[i] == target else 0 for i in range(len(targets))]

        with torch.cuda.device(gpu_id):
            target = Variable(torch.FloatTensor(target)).cuda()
            Agent.cx = Variable(torch.zeros(1, 256).cuda())
            Agent.hx = Variable(torch.zeros(1, 256).cuda())
            Agent.target = target

        total_reward, num_steps, good = 0, 0, 0
        done = False

        while not done:
            num_steps += 1
            observation = next_observation
            act = Agent.action_train(observation, target)

            next_observation, reward, done, info = task.step(actions[act[0]])
            total_reward += reward
            Agent.put_reward(reward)
            if num_steps % params.num_steps == 0 or done:
                if done:
                    Agent.done = done
                    Agent.state_done = task.reset()
                    good = 1
                succ.append(good)
                Agent.training(rank, observation, act, reward, next_observation, shared_model, shared_optimizer, params)


def test(args, params, shared_model):
    print('test')