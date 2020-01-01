import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

import numpy as np
from utils import ensure_shared_grads
from collections import deque

import pdb
#numpy.set_printoptions(threshold=numpy.nan)

def preprocess(obs):
    obs = torch.from_numpy(np.array(obs, dtype='f')) / 255.
    state = obs.permute(2, 0, 1).unsqueeze(0)
    
    return state

class run_agent(object):
    def __init__(self, model, gpu_id):
        self.model = model
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = False
        self.info = None
        self.reward = 0
        self.gpu_id = gpu_id
        self.target = None
        self.n_update = 0

    def action_train(self, observation, instruction_idx):
        self.state = preprocess(observation)
        
        with torch.cuda.device(self.gpu_id):
            obs = Variable(torch.FloatTensor(self.state)).cuda()
        value, logit, self.hx, self.cx = self.model(obs, instruction_idx, self.hx, self.cx)

        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        #self.entropies.append(entropy)

        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, Variable(action))
        action = action.cpu().numpy()

        #self.values.append(value)
        #self.log_probs.append(log_prob)
        return np.squeeze(action, axis=0), entropy, value, log_prob

    def action_test(self, observation, instruction_idx):
        with torch.cuda.device(self.gpu_id):
            with torch.no_grad():
                self.state = preprocess(observation)
                obs = Variable(torch.FloatTensor(self.state)).cuda()
                value, logit, self.hx, self.cx = self.model(obs, instruction_idx, self.hx, self.cx)

                prob = F.softmax(logit, dim=1)
                action = prob.max(1)[1].data.cpu().numpy()

        self.eps_len += 1
        return action

    def clear_actions(self):
        self.values.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()
        return self

    def put_reward(self, reward, entropy, value, log_prob):
        self.rewards.append(reward)

        self.entropies.append(entropy)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def training(self, next_observation, shared_model, shared_optimizer, params):
        self.model.train()
        self.n_update += 1
        self.cx = Variable(self.cx.data)
        self.hx = Variable(self.hx.data)

        R = torch.zeros(1, 1)
        if not self.done:
            self.state = preprocess(next_observation)
            with torch.cuda.device(self.gpu_id):
                obs = Variable(torch.FloatTensor(self.state)).cuda()
            value, _, _, _  = self.model(obs, self.target, self.hx, self.cx)
            R = value.data

        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                R = R.cuda()
        R = Variable(R)
        self.values.append(R)

        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                gae = gae.cuda()

        for i in reversed(range(len(self.rewards))):
            R = params.gamma * R + self.rewards[i]
            advantage = R - self.values[i]
            value_loss = value_loss + advantage.pow(2)  # 0.5 *

            # Generalized Advantage Estimataion
            delta_t = params.gamma * self.values[i + 1].data - self.values[i].data + self.rewards[i]

            gae = gae * params.gamma * params.tau + delta_t

            policy_loss = policy_loss - self.log_probs[i] * Variable(gae) - params.entropy_coef * self.entropies[i]

        self.model.zero_grad()
        (policy_loss + params.value_loss_coef * value_loss).backward()
        clip_grad_norm_(self.model.parameters(), 10.0)
        ensure_shared_grads(self.model, shared_model, gpu=self.gpu_id >= 0)
        shared_optimizer.step()
        with torch.cuda.device(self.gpu_id):
            self.model.load_state_dict(shared_model.state_dict())   #model update
        self.clear_actions()