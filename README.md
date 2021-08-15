# MicroRTS PPO Performance Comparsion

This repo attempts to reproduce the results of the unmasked PPO model found in [the source code](https://github.com/vwxyzjn/invalid-action-masking) for the paper [*A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*](https://arxiv.org/abs/2006.14171). The original implementation is compared against a new model using Stable Baselines 3's PPO implementation.

## Installation

You should already have Python 3 and CUDA installed. Then simply install the necessary Python packages:

```
pip install -r requirements.txt
```

## Running

### Original implementation

```
python original/train_ppo_4x4.py
```

Note that this script is functionally identical to [the original found here](https://github.com/vwxyzjn/invalid-action-masking/blob/c0d47cca3c2d8522ce97412b76ca4e4e36c5d95e/invalid_action_masking/ppo_no_mask_4x4.py).

### Stable Baselines 3 (SB3) implementation

```
python sb3/train_ppo_4x4.py zoo/4x4
```

This script makes use of SB3's [core PPO algorithm](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).
