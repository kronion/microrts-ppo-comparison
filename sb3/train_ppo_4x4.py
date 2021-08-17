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
import wandb
from wandb.integration.sb3 import WandbCallback
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


def make_env(gym_id, seed, idx):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

# Maintain a similar CLI to the original paper's implementation
@click.command()
@click.argument("output_folder", type=click.Path())
@click.option("--load", "-l", "load_path")
@click.option("--seed", type=int, default=1, help="seed of the experiment")
@click.option(
    "--total-timesteps",
    type=int,
    default=1000000,
    help="total timesteps of the experiments",
)
@click.option(
    "--torch-deterministic/--no-torch-deterministic",
    default=True,
    help="if toggled, `torch.backends.cudnn.deterministic=False`",
)
def train(output_folder, load_path, seed, total_timesteps, torch_deterministic):
    run = wandb.init(
        project="sb3",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    base_output = Path(output_folder)
    full_output = base_output / datetime.datetime.now().isoformat(timespec="seconds")
    logger.configure(folder=str(full_output))

    
    # env = gym.make()

    # We want deterministic operations whenever possible, but unfortunately we
    # still depend on some non-deterministic operations like
    # scatter_add_cuda_kernel. For now we settle for deterministic convolution.
    # th.use_deterministic_algorithms(torch_deterministic)
    th.backends.cudnn.deterministic = torch_deterministic

    random.seed(SEED)
    np.random.seed(SEED)
    th.manual_seed(SEED)
    # env.seed(SEED)
    # env.action_space.seed(SEED)
    # env.observation_space.seed(SEED)

    # # Normalize env with VecNormalize
    # env = Monitor(env)
    # env = DummyVecEnv([lambda: env] * 8)
    env = DummyVecEnv([make_env("MicrortsMining4x4F9-v0", SEED+i, i) for i in range(8)])

    if load_path:
        model = PPO.load(load_path, env)
    else:
        model = PPO(
            MlpPolicy,
            env,
            n_steps=128,
            n_epochs=4,
            learning_rate=lambda progression: 2.5e-4 * progression,
            ent_coef=0.01,
            clip_range=0.1,
            batch_size=256,
            verbose=1,
            policy_kwargs={
                "net_arch": [128],
                "activation_fn": nn.ReLU,
                "features_extractor_class": MicroRTSExtractor,
            },
            tensorboard_log=f"runs/{run.id}"
        )
    print(env.num_envs)
    print(model.policy)
    # raise
    model.learn(
        total_timesteps=total_timesteps,
        callback=WandbCallback(
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    model.save(str(full_output / "final_model"))
    env.close()


if __name__ == "__main__":
    train()
