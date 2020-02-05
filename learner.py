import torch
from torch.autograd import Variable

import random
import numpy as np
import time
import os
import logging

from agent import run_agent

action_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

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

    list_obs = [None] * n_process  # initialize
    list_target = [None] * n_process
    list_rewards = [None] * n_process
    list_dones = [False] * n_process

    test_result = [[None] * n_process] * params.n_eval
    n_result = 0
    test_done = [0] * n_process
    test_succ = [0] * n_process

    best_rate = 0

    start_time = time.time()
    while True:
        for i in range(n_process):
            rank, [obs, target] = state_Queue.get()
            list_obs[rank] = obs
            list_target[rank] = target

            if list_dones[i] is True:
                # Agent.task_init(i)
                list_dones[i] = False

        # do inference and make action
        # actions.value = [random.randrange(len(action_list))] * n_process
        inference, value, log_prob, entropy = Agent.action_train(list_obs, list_target)
        n_inference += 1

        for i in range(n_process):
            actions[i] = action_list[inference[i]]
        # actions = Agent.action_train(list_obs, list_target)

        for i in range(n_process):
            action_done.put([0])

        for i in range(n_process):
            rank, done, reward = reward_Queue.get()
            list_rewards[rank] = [reward]

            if done:
                list_dones[rank] = True

        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in list_dones])
        # Agent.put_reward(list_rewards)
        Agent.insert(log_prob, value, torch.from_numpy(np.array(list_rewards, dtype=np.float32)), masks, entropy)

        if n_inference % 30 == 0 and n_inference != 0:
            next_obs = list_obs, list_target
            # Agent.update(params, next_obs, list_dones, entropy)
            Agent.update_sync(params, next_obs)
            n_update += 1

            if n_update % 200 == 0:  # test
                n_eval = False
                startTest = False
                endTest = False
                while True:
                    for i in range(n_process):
                        rank, [obs, target] = state_Queue.get()
                        list_obs[rank] = obs
                        list_target[rank] = target

                        if list_dones[i] is True:
                            # Agent.task_init(i)
                            list_dones[i] = False

                    # do inference and make action
                    # actions.value = [random.randrange(len(action_list))] * n_process
                    if startTest is False:
                        for i in range(n_process):
                            actions[i] = 99 # reset for evaluation
                            startTest = True
                    elif endTest is True:
                        for i in range(n_process):
                            actions[i] = 99 # reset for training

                            test_done = [0] * n_process     # initialize
                            test_succ = [0] * n_process
                            n_eval = False
                    else:
                        inference = Agent.action_test(list_obs, list_target)
                        for i in range(n_process):
                            actions[i] = action_list[inference[i]]
                    # actions = Agent.action_train(list_obs, list_target)

                    for i in range(n_process):
                        action_done.put([0])

                    if endTest is True:
                        break

                    if n_eval is False:
                        n_eval = True
                    else:
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
                            endTest = True
                            if test_done[i] < params.n_eval:
                                endTest = False
                                break

                        if endTest:
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