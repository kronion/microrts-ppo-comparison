import datetime
import random
from pathlib import Path

import click
import gym
import gym_microrts
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common import logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.ppo import CnnPolicy, MlpPolicy, PPO

from extractors import MicroRTSExtractor


EVAL_FREQ = 5000
EVAL_EPISODES = 10

SEED = 1


# Parameters specified by this paper: https://arxiv.org/pdf/2006.14171.pdf
gamma = 0.99
gae_lambda = 0.97
clip_coef = 0.2
entropy_reg_coef = 0.01
max_grad_norm = 0.5
learning_rate = 0.0003


# Maintain a similar CLI to the original paper's implementation
@click.command()
@click.argument("output_folder", type=click.Path())
@click.option("--load", "-l", "load_path")
@click.option("--seed", type=int, default=1, help="seed of the experiment")
@click.option(
    "--total-timesteps",
    type=int,
    default=100000,
    help="total timesteps of the experiments",
)
@click.option(
    "--torch-deterministic/--no-torch-deterministic",
    default=True,
    help="if toggled, `torch.backends.cudnn.deterministic=False`",
)
def train(output_folder, load_path, seed, total_timesteps, torch_deterministic):
    base_output = Path(output_folder)
    full_output = base_output / datetime.datetime.now().isoformat(timespec="seconds")
    logger.configure(folder=str(full_output))

    env = gym.make("MicrortsMining4x4F9-v0")

    # We want deterministic operations whenever possible, but unfortunately we
    # still depend on some non-deterministic operations like
    # scatter_add_cuda_kernel. For now we settle for deterministic convolution.
    # th.use_deterministic_algorithms(torch_deterministic)
    th.backends.cudnn.deterministic = torch_deterministic

    random.seed(SEED)
    np.random.seed(SEED)
    th.manual_seed(SEED)
    env.seed(SEED)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)

    # Normalize env with VecNormalize
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_reward=False)

    if load_path:
        model = PPO.load(load_path, env)
    else:
        model = PPO(
            MlpPolicy,
            env,
            verbose=1,
            batch_size=256,
            n_epochs=80,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_coef,
            clip_range_vf=clip_coef,
            ent_coef=entropy_reg_coef,
            max_grad_norm=max_grad_norm,
            learning_rate=learning_rate,
            policy_kwargs={
                "net_arch": [128],
                "activation_fn": nn.ReLU,
                "features_extractor_class": MicroRTSExtractor,
            },
        )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=str(full_output),
        log_path=str(full_output),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=False,
        render=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(str(full_output / "final_model"))
    env.close()


if __name__ == "__main__":
    train()
