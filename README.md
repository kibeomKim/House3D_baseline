# House3D_RoomNav_baseline

This is the baseline model of the RoomNav task using House3D.

It implements A3C with gated-LSTM policy for discrete actions. (DDPG with gated-CNN policy did not yet impleneted.)

In the paper, they used 120 or 200 processes but I only used 20 processes.

https://arxiv.org/abs/1801.02209



#### requirements

```
pip install -r requirements.txt 
```

you should input your path to **config.json**

and your gpu ids depending on your environment in Class Params() in main.py . (In my case, I used gpu 0 for tests and gpu 1, 2, 3 for training)



#### Training

```
python main.py 
```

