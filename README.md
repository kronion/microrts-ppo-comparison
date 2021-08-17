# MicroRTS PPO Performance Comparsion

This repo attempts to reproduce the results of the unmasked PPO model found in
[the source code](https://github.com/vwxyzjn/invalid-action-masking)
for the paper [*A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*](https://arxiv.org/abs/2006.14171).
The original implementation is compared against a new model using Stable Baselines 3's PPO implementation.

## Installation

You should already have Python 3 and CUDA installed. Then simply install the necessary Python packages:

```
pip install -r requirements.txt
```

## Running

### Original implementation

```
python original/new_train_ppo_4x4.py
```

Note that this script is functionally identical to
[the original found here](https://github.com/vwxyzjn/invalid-action-masking/blob/54bfb37b939e8f9e77dcf96f79b7df4953e012f2/ppo.py).

### Stable Baselines 3 (SB3) implementation

```
python sb3/train_ppo_4x4.py zoo/4x4
```

This script makes use of SB3's [core PPO algorithm](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).

## Results

At present, the SB3 version struggles to achieve episode rewards greater than 2 after 100k timesteps,
while the original version consistently achieves a reward of 40 (the maximum for the envionment).
