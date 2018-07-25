import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from utils import ensure_shared_grads

import pdb


class run_agent(object):
    def __init__(self, model, gpu_id):
        self.model = model
        #self.env = env
        #self.state = state
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
        self.gpu_id = gpu_id        #if do not have gpus, set -1
        self.state_done = None
        self.target = None

    def action_train(self, observation, target):
        #pdb.set_trace()
        #self.state = preprocessing(observation, self.gpu_id)
        self.state = observation
        with torch.cuda.device(self.gpu_id):
            obs = Variable(torch.FloatTensor(self.state)).cuda()
        value, logit, self.hx, self.cx = self.model(obs.permute(2, 0, 1)[None], target, self.hx, self.cx)

        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)

        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, Variable(action))
        action = action.cpu().numpy()

        #self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        #self.rewards.append(self.reward)
        return np.squeeze(action, axis=0)

    def action_test(self, observation, target):

        with torch.no_grad():
            self.state = observation
            with torch.cuda.device(self.gpu_id):
                obs = Variable(torch.FloatTensor(self.state)).cuda()
            value, logit, self.hx, self.cx = self.model(obs.permute(2, 0, 1)[None], target, self.hx, self.cx)

        prob = F.softmax(logit, dim=1)

        action = prob.multinomial(1).data
        action = action.cpu().numpy()
        self.eps_len += 1
        return np.squeeze(action, axis=0)

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self

    def put_reward(self, reward):
        self.rewards.append(reward)

    def training(self, rank, observation, action, reward, next_observation, shared_model, shared_optimizer, params):
        #self.rewards.append(reward)

        #ptitle('Training Agent: {}'.format(rank))
        torch.manual_seed(params.seed + rank)
        if self.gpu_id >= 0:
            torch.cuda.manual_seed(params.seed + rank)

        self.cx = Variable(self.cx.data)
        self.hx = Variable(self.hx.data)

        if self.done:
            self.state = self.state_done

        R = torch.zeros(1, 1)
        if not self.done:
            self.state = observation
            with torch.cuda.device(self.gpu_id):
                obs = Variable(torch.FloatTensor(self.state)).cuda()
            value, _, _, _  = self.model(obs.permute(2, 0, 1)[None], self.target, self.hx, self.cx)
            R = value.data

        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                R = R.cuda()

        self.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                gae = gae.cuda()

        R = Variable(R)
        for i in reversed(range(len(self.rewards))):
            R = params.gamma * R + self.rewards[i]
            advantage = R - self.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            #pdb.set_trace()
            # Generalized Advantage Estimataion
            delta_t = params.gamma * self.values[i + 1].data - self.values[i].data + self.rewards[i]

            gae = gae * params.gamma * params.tau + delta_t

            policy_loss = policy_loss - self.log_probs[i] * Variable(gae) - 0.01 * self.entropies[i]

        self.model.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        ensure_shared_grads(self.model, shared_model, gpu=self.gpu_id >= 0)
        shared_optimizer.step()
        self.clear_actions()