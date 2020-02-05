import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

import numpy as np
from utils import ensure_shared_grads
from collections import deque
import copy

from models import A3C_LSTM_GA

import pdb
#numpy.set_printoptions(threshold=numpy.nan)

def preprocess(obs, instruction, gpu_id):
    obs = torch.from_numpy(np.array(obs, dtype=np.float32)) / 255.
    state = obs.permute(0, 3, 1, 2)

    instruction = torch.stack(instruction)

    with torch.cuda.device(gpu_id):
        state = Variable(torch.FloatTensor(state)).cuda()
        instruction = Variable(torch.LongTensor(instruction)).cuda()

    return state, instruction

class run_agent(object):
    def __init__(self, params, gpu_id=0):
        self.params = params
        self.device = "cuda:"+ str(gpu_id)
        self.model = A3C_LSTM_GA().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=params.lr, amsgrad=params.amsgrad,
                                    weight_decay=params.weight_decay)
        self.hx = torch.zeros(self.params.n_process, 256).to(self.device)
        self.cx = torch.zeros(self.params.n_process, 256).to(self.device)
        self.eps_len = 0
        self.values = []
        self.log_probs = []
        self.rewards = []
        # self.entropies = []
        self.done = False
        self.info = None
        self.reward = 0
        self.gpu_id = gpu_id
        self.target = None
        self.n_update = 0

        self.num_steps = params.num_steps
        self.step = 0

        self.rewards = torch.zeros(self.num_steps, params.n_process, 1).to(self.device)
        self.value_preds = torch.zeros(self.num_steps + 1, params.n_process, 1).to(self.device)
        self.returns = torch.zeros(self.num_steps + 1, params.n_process, 1).to(self.device)
        self.action_log_probs = torch.zeros(self.num_steps, params.n_process, 1).to(self.device)
        self.masks = torch.ones(self.num_steps + 1, params.n_process, 1).to(self.device)

        self.entropies = torch.zeros(self.num_steps, params.n_process, 1).to(self.device)


    def action_train(self, observation, instruction_idx):
        self.state, self.target = preprocess(observation, instruction_idx, gpu_id=self.gpu_id)

        value, logit, self.hx, self.cx = self.model(self.state, self.target, self.hx, self.cx)

        prob = F.softmax(logit, dim=-1)
        log_prob = F.log_softmax(logit, dim=-1)
        entropy = -(log_prob * prob).sum(-1)

        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, Variable(action))
        action = action.cpu().numpy()

        # self.entropies.append(entropy)
        # self.values.append(value)
        # self.log_probs.append(log_prob)

        return np.squeeze(action), value, log_prob, entropy.unsqueeze(dim=1)

    def action_test(self, observation, instruction_idx):
        with torch.cuda.device(self.gpu_id):
            with torch.no_grad():
                self.state, self.target = preprocess(observation, instruction_idx, gpu_id=self.gpu_id)
                value, logit, self.hx, self.cx = self.model(self.state, self.target, self.hx, self.cx)

                prob = F.softmax(logit, dim=1)
                action = prob.max(1)[1].data.cpu().numpy()

        self.eps_len += 1
        return action

    def clear_actions(self):
        # del self.values[:-1]
        # del self.log_probs[:-1]
        # del self.entropies[:-1]
        # del self.rewards[:-1]
        # self.values.clear()
        # self.log_probs.clear()
        # self.entropies.clear()
        # self.rewards.clear()
        self.rewards = torch.zeros(self.num_steps, self.params.n_process, 1).to(self.device)
        self.value_preds = torch.zeros(self.num_steps + 1, self.params.n_process, 1).to(self.device)
        self.returns = torch.zeros(self.num_steps + 1, self.params.n_process, 1).to(self.device)
        self.action_log_probs = torch.zeros(self.num_steps, self.params.n_process, 1).to(self.device)
        # self.masks = torch.ones(self.num_steps + 1, self.params.n_process, 1).to(self.device)
        self.masks[0].copy_(self.masks[-1])

        self.entropies = torch.zeros(self.num_steps, self.params.n_process, 1).to(self.device)

    def put_reward(self, reward):
        self.rewards.append(reward)

    def initialize(self):
        with torch.cuda.device(self.gpu_id):
            self.hx = Variable(torch.zeros(self.params.n_process, 256).cuda())
            self.cx = Variable(torch.zeros(self.params.n_process, 256).cuda())
        self.clear_actions()

    def task_init(self, n_process):
        with torch.cuda.device(self.gpu_id):
            self.hx[n_process] = Variable(torch.zeros(1, 256).cuda())
            self.cx[n_process] = Variable(torch.zeros(1, 256).cuda())

    def insert(self, action_log_probs, value_preds, rewards, masks, entropy):
        self.action_log_probs[self.step] = action_log_probs
        self.value_preds[self.step] = value_preds
        self.rewards[self.step] = rewards
        self.masks[self.step + 1] = masks
        self.entropies[self.step] = entropy

        self.step = (self.step + 1) % self.num_steps

    def insert_copy(self, action_log_probs, value_preds, rewards, masks):
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):

        self.masks[0].copy_(self.masks[-1])

        # self.rewards = torch.zeros(self.num_steps, self.params.n_process, 1).to(self.device)
        # self.value_preds = torch.zeros(self.num_steps + 1, self.params.n_process, 1).to(self.device)
        # self.returns = torch.zeros(self.num_steps + 1, self.params.n_process, 1).to(self.device)
        # self.action_log_probs = torch.zeros(self.num_steps, self.params.n_process, 1).to(self.device)


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


    def update(self, params, next_obs, list_dones, entropy):
        self.model.train()
        self.n_update += 1

        observation, instruction_idx = next_obs
        with torch.no_grad():
            state, target = preprocess(observation, instruction_idx, gpu_id=self.gpu_id)
            next_value, _, _, _ = self.model(state, target, self.hx, self.cx)

        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + params.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
            gae = delta + params.gamma * params.tau * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]

        advantages = self.returns[:-1] - self.value_preds[:-1]
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * self.action_log_probs).mean()

        self.optimizer.zero_grad()

        # (policy_loss.mean() + params.value_loss_coef * value_loss.mean()).backward()
        (value_loss * params.value_loss_coef + action_loss - entropy * params.entropy_coef).backward()
        clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        print(action_loss)
        print(value_loss)

        # self.clear_actions()
        self.after_update()

    def update_sync(self, params, next_obs):
        self.model.train()
        self.n_update += 1
        self.cx = Variable(self.cx.data)
        self.hx = Variable(self.hx.data)

        observation, instruction_idx = next_obs
        with torch.no_grad():
            state, target = preprocess(observation, instruction_idx, gpu_id=self.gpu_id)
            next_value, _, _, _ = self.model(state, target, self.hx, self.cx)
        R = next_value

        self.value_preds[-1] = next_value

        policy_loss = 0
        value_loss = 0
        gae = 0

        for i in reversed(range(self.rewards.size(0))):
            R = params.gamma * R + self.rewards[i]
            advantage = R - self.value_preds[i]
            value_loss = value_loss + advantage.pow(2)  # 0.5 *

            # Generalized Advantage Estimataion
            delta_t = params.gamma * self.value_preds[i + 1].data * self.masks[i + 1] - self.value_preds[i].data + self.rewards[i]

            gae = gae * params.gamma * params.tau * self.masks[i + 1] + delta_t

            policy_loss = policy_loss - self.action_log_probs[i] * Variable(gae) - params.entropy_coef * self.entropies[i]

        self.model.zero_grad()
        # print(policy_loss.mean())
        # print(value_loss.mean())
        (policy_loss.mean() + params.value_loss_coef * value_loss.mean()).backward()
        clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        self.clear_actions()