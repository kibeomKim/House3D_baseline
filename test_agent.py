from House3D import Environment, objrender, load_config
from House3D.roomnav import RoomNavTask
from collections import deque
import numpy as np

EPISODE = 1000

houses = ['00065ecbdd7300d35ef4328ffe871505',
          'cf57359cd8603c3d9149445fb4040d90', '31966fdc9f9c87862989fae8ae906295',
          '7995c2a93311717a3a9c48d789563590', '8b8c1994f3286bfc444a7527ffacde86',
          '32e53679b33adfcc5a5660b8c758cc96',
          '492c5839f8a534a673c92912aedc7b63',
          'e3ae3f7b32cf99b29d3c8681ec3be321',
          '1dba3a1039c6ec1a3c141a1cb0ad0757',
          '5f3f959c7b3e6f091898caa8e828f110']

def test(args, params, shared_model):
    ptitle('Test Agent')
    api = objrender.RenderAPI(w=400, h=300, device=gpu_id)
    cfg = load_config('config.json')
    env = Environment(api, np.random.choice(houses, 1)[0], cfg)
    task = RoomNavTask(env, hardness=0.6, discrete_action=True)

    succ = deque(maxlen=500)
    for i in range(EPISODE):
        step, total_rew, good = 0, 0, 0
        task.reset()

        while True:
            act = task._action_space.sample()

            obs, rew, done, info = task.step(act)

            total_rew += rew

            step += 1

            if done or step > 150:
                if done:
                    good = 1
                succ.append(good)

                print("\n+++++++++++++ status ++++++++++++")
                print("Episode %04d, Reward: %2.3f" % (i + 1, total_rew))
                print("Target %s" % task.info['target_room'])
                print("Success rate {:3.2f}%".format(np.sum(succ) / len(succ) * 100))
                break