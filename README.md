# House3D_RoomNav_baseline

This is the baseline model of the RoomNav task using House3D that I implemented.

It implements A3C with gated-LSTM policy for discrete actions.

In the paper, they used 120 or 200 processes but I only used 20 processes.
https://arxiv.org/abs/1801.02209



#### requirements
- python 3.6+
- pytorch 0.4.1

```
pip install -r requirements.txt 
```

you should input your path to **config.json**

and your gpu ids depending on your environment in Class Params() in main.py . (In my case, I used gpu 0 for tests and gpu 1, 2, 3 for training)



#### Training

```
python main.py 
```

#### Evaluation

![](https://raw.githubusercontent.com/kibeomKim/House3D_baseline/master/img/train_success_rate.png?token=ATkBLxbiNzijzfOOlUTaLUEXosxiHGa1ks5btsfuwA%3D%3D)

Each line represents each env used in the learning.

#### Todo

- [ ] add batch structure 64(batch)*30(continuous actions)
- [ ] add decay the learning rate by a factor of 1.5 when the difference of KL-divergence becomes larger than 0.01



#### Project Reference

https://github.com/dgriff777/rl_a3c_pytorch
