import torch
from torch.autograd import Variable

import random
import numpy as np
import time
import os
import logging

from agent import run_agent

action_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def getState(n_process, state_Queue, list_dones):
    list_obs = [None] * n_process  # initialize
    list_target = [None] * n_process

    for i in range(n_process):
        rank, [obs, target] = state_Queue.get()

        list_obs[rank] = obs
        list_target[rank] = target

        if list_dones[i] is True:
            # Agent.task_init(i)
            list_dones[i] = False

    return list_obs, list_target, list_dones

def getReward(n_process, reward_Queue, list_dones):
    list_rewards = [None] * n_process

    for i in range(n_process):
        [rank, done, reward] = reward_Queue.get()

        if list_rewards[rank] is None:
            list_rewards[rank] = [reward]
        #     print("recv - rank: {:d}, reward: {:3.2f}".format(rank, reward))
        # else:
        #     print("recv - rank: {:d}, double recv".format(rank))

        if done:
            list_dones[rank] = True

    for i in range(n_process):
        reward_Queue.task_done()

    masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in list_dones])

    return torch.from_numpy(np.array(list_rewards, dtype=np.float32)), list_dones, masks

def putActions(n_process, inference, actions, action_done):
    for i in range(n_process):
        actions[i] = action_list[inference[i]]
    # actions = Agent.action_train(list_obs, list_target)

    for i in range(n_process):
        action_done.put([0])

def callReset(n_process, actions, action_done):
    for i in range(n_process):
        actions[i] = 99  # reset for training
        # reward_Queue.task_done()
    test_done = [0] * n_process  # initialize
    test_succ = [0] * n_process

    for i in range(n_process):
        action_done.put([0])

    return test_done, test_succ

def DoneOrNot(n_process, test_done, n_eval):
    for i in range(n_process):
        endTest = True
        if test_done[i] < n_eval:
            endTest = False
            break
    return endTest

def EnvReset(n_process, state_Queue, action_done, list_dones, actions):
    list_obs, list_target, list_dones = getState(n_process, state_Queue, list_dones)
    test_done, test_succ = callReset(n_process, actions, action_done)
    list_rewards = [None] * n_process
    return test_done, test_succ, list_rewards

def writeResult(n_process, test_done, test_succ, best_rate, Agent, params, n_update, start_time, logging):
    total = 0
    n_succ = 0
    for i in range(n_process):
        total += test_done[i]
        n_succ += test_succ[i]
    succ_rate = (float(n_succ) / float(total)) * 100
    if best_rate < succ_rate:
        best_rate = succ_rate
        torch.save(Agent.model.state_dict(), params.weight_dir + 'model' + str(n_update) + '.ckpt')
    msg = " ".join([
        "++++++++++ Task Stats +++++++++++\n",
        "Time {}\n".format(
            time.strftime("%dd %Hh %Mm %Ss", time.gmtime(time.time() - start_time))),
        "Episode Played: {:d}\n".format(test_done[0]),
        "N_Update = {:d}\n".format(n_update),
        "Best rate {:3.2f}, Success rate {:3.2f}%".format(best_rate, succ_rate)
    ])
    print(msg)
    logging.info(msg)

    return best_rate


def learning(params, state_Queue, action_done, actions, reward_Queue):
    if not os.path.exists('./'+params.weight_dir):
        os.mkdir('./'+params.weight_dir)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    logging.basicConfig(filename='./log/'+params.log_file+'.log', level=logging.INFO)

    n_process = params.n_process

    n_inference = 0
    n_update = 0

    Agent = run_agent(params)
    # Agent.initialize()

    list_dones = [False] * n_process

    n_result = 0
    test_done = [0] * n_process
    test_succ = [0] * n_process

    best_rate = 0

    start_time = time.time()
    while True:
        list_obs, list_target, list_dones = getState(n_process, state_Queue, list_dones)
        # do inference and make action
        # actions.value = [random.randrange(len(action_list))] * n_process
        inference, value, log_prob, entropy = Agent.action_train(list_obs, list_target)
        n_inference += 1

        putActions(n_process, inference, actions, action_done)

        torch_rewards, list_dones, masks = getReward(n_process, reward_Queue, list_dones)

        Agent.insert(log_prob, value, torch_rewards, masks, entropy)

        if n_inference % 30 == 0 and n_inference != 0:
            next_obs = list_obs, list_target
            # Agent.update(params, next_obs, list_dones, entropy)
            Agent.update_sync(params, next_obs)
            n_update += 1

            if n_update % 200 == 0:  # test
                test_done, test_succ, list_rewards = EnvReset(n_process, state_Queue, action_done, list_dones, actions)

                endTest = False
                while True:
                    list_obs, list_target, list_dones = getState(n_process, state_Queue, list_dones)

                    if endTest is True:
                        best_rate = writeResult(n_process, test_done, test_succ, best_rate, Agent, params, n_update, start_time, logging)
                        test_done, test_succ = callReset(n_process, actions, action_done)
                        break

                    # do inference and make action
                    inference = Agent.action_test(list_obs, list_target)
                    putActions(n_process, inference, actions, action_done)

                    for i in range(n_process):
                        rank, done, reward = reward_Queue.get()
                        list_rewards[rank] = [reward]
                        list_dones[rank] = True

                        if done:
                            if test_done[rank] < params.n_eval:
                                test_done[rank] += 1
                                if reward == 10:    # success
                                    test_succ[rank] += 1

                    for i in range(n_process):
                        reward_Queue.task_done()

                    endTest = DoneOrNot(n_process, test_done, params.n_eval)
