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
    #indices_rgb = torch.tensor([0, 1, 2])
    #obs_rgb = torch.index_select(obs, 1, indices_rgb) / 255.
    #obs_rgb.numpy()
    #indices_depth = torch.tensor([3])
    #obs_depth = torch.index_select(obs, 1, indices_depth) / 50.

    #state = torch.cat([obs_rgb, obs_depth], 1)
    return state

class run_agent(object):
    def __init__(self, model, gpu_id):
        self.model = model
        #self.env = env
        #self.state = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.values = []    #deque(maxlen=65)
        self.log_probs = [] #deque(maxlen=64)
        self.rewards = []   #deque(maxlen=64)
        self.entropies = [] #deque(maxlen=64)
        self.done = False
        self.info = None
        self.reward = 0
        self.gpu_id = gpu_id
        self.target = None
        self.n_update = 0

    def action_train(self, observation, instruction_idx):
        #self.model.train()
        #self.model.eval()
        #pdb.set_trace()
        self.state = preprocess(observation)
        
        with torch.cuda.device(self.gpu_id):
            obs = Variable(torch.FloatTensor(self.state)).cuda()
        value, logit, self.hx, self.cx = self.model(obs, instruction_idx, self.hx, self.cx)

        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)

        action = prob.multinomial(1).data
        #action = prob.argmax(dim=1, keepdim=True)
        log_prob = log_prob.gather(1, Variable(action))
        action = action.cpu().numpy()

        #self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        #self.rewards.append(self.reward)
        return np.squeeze(action, axis=0)

    def action_test(self, observation, instruction_idx):
        #self.model.eval()
        with torch.cuda.device(self.gpu_id):
            with torch.no_grad():
                self.state = preprocess(observation)
                obs = Variable(torch.FloatTensor(self.state)).cuda()
                value, logit, self.hx, self.cx = self.model(obs, instruction_idx, self.hx, self.cx)

                prob = F.softmax(logit, dim=1)
                action = prob.max(1)[1].data.cpu().numpy()
                #action = prob.argmax(dim=1, keepdim=True)
                #action = prob.multinomial(1).data
        #action = action.cpu().numpy()
        self.eps_len += 1
        return action #np.squeeze(action, axis=0)

    def clear_actions(self):
        self.values.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()
        return self

    def put_reward(self, reward):
        self.rewards.append(reward)

    def training(self, rank, observation, action, reward, next_observation, shared_model, shared_optimizer, params):
        #pdb.set_trace()
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
            value_loss = value_loss + 0.5 * advantage.pow(2)  #

            # Generalized Advantage Estimataion
            #pdb.set_trace()
            delta_t = params.gamma * self.values[i + 1].data - self.values[i].data + self.rewards[i]

            gae = gae * params.gamma * params.tau + delta_t

            policy_loss = policy_loss - self.log_probs[i] * Variable(gae) - params.entropy_coef * self.entropies[i]

        #if self.n_update % 100 == 0:
        #    pdb.set_trace()
        self.model.zero_grad()
        (policy_loss + params.value_loss_coef * value_loss).backward()   # retain_graph=True,
        clip_grad_norm_(self.model.parameters(), 1.0)  #1.0
        ensure_shared_grads(self.model, shared_model, gpu=self.gpu_id >= 0)
        shared_optimizer.step()
        with torch.cuda.device(self.gpu_id):
            self.model.load_state_dict(shared_model.state_dict())   #model update
        self.clear_actions()