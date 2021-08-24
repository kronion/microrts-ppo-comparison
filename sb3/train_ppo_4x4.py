import datetime
import random
from pathlib import Path

import click
import gym
import gym_microrts
import numpy as np
import torch as th
import torch.nn as nn
import wandb
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.ppo import PPO
from wandb.integration.sb3 import WandbCallback

from extractors import MicroRTSExtractor
from policies import CustomActorCriticPolicy


class Defaults:
    TOTAL_TIMESTEPS = 100000
    EVAL_FREQ = 5000
    EVAL_EPISODES = 10
    SEED = 1

    # Parameters specified by this paper: https://arxiv.org/pdf/2006.14171.pdf
    ENTROPY_COEF = 0.01


gamma = 0.99
gae_lambda = 0.97
clip_coef = 0.2
max_grad_norm = 0.5
learning_rate = 0.0003


# Maintain a similar CLI to the original paper's implementation
@click.command()
@click.argument("output_folder", type=click.Path())
@click.option("--load", "-l", "load_path")
@click.option("--seed", type=int, default=Defaults.SEED, help="seed of the experiment")
@click.option(
    "--total-timesteps",
    type=int,
    default=Defaults.TOTAL_TIMESTEPS,
    help="total timesteps of the experiments",
)
@click.option(
    "--eval-freq",
    type=int,
    default=Defaults.EVAL_FREQ,
    help="number of timesteps between model evaluations",
)
@click.option(
    "--eval-episodes",
    type=int,
    default=Defaults.EVAL_EPISODES,
    help="number of games to play during each model evaluation step",
)
@click.option(
    "--torch-deterministic/--no-torch-deterministic",
    default=True,
    help="if toggled, `torch.backends.cudnn.deterministic=False`",
)
@click.option(
    "--entropy-coef",
    type=float,
    default=Defaults.ENTROPY_COEF,
    help="Coefficient for entropy component of loss function",
)
def train(
        output_folder,
        load_path,
        seed,
        total_timesteps,
        eval_freq,
        eval_episodes,
        torch_deterministic,
        entropy_coef,
):
    run = wandb.init(
        project="sb3",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        anonymous="true",  # wandb documentation is wrong...
    )

    base_output = Path(output_folder)
    full_output = base_output / datetime.datetime.now().isoformat(timespec="seconds")
    logger.configure(folder=str(full_output))

    env = gym.make("MicrortsMining4x4F9-v0")

    # We want deterministic operations whenever possible, but unfortunately we
    # still depend on some non-deterministic operations like
    # scatter_add_cuda_kernel. For now we settle for deterministic convolution.
    # th.use_deterministic_algorithms(torch_deterministic)
    th.backends.cudnn.deterministic = torch_deterministic

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # Normalize env with VecNormalize
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_reward=False)

    if load_path:
        model = PPO.load(load_path, env)
    else:
        model = PPO(
            CustomActorCriticPolicy,
            env,
            verbose=1,
            batch_size=256,
            n_epochs=4,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_coef,
            clip_range_vf=clip_coef,
            ent_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            learning_rate=learning_rate,
            policy_kwargs={
                "net_arch": [dict(pi=[128], vf=[128])],
                "activation_fn": nn.ReLU,
                "policy_features_extractor_class": MicroRTSExtractor,
                "value_features_extractor_class": MicroRTSExtractor,
            },
            tensorboard_log=str(full_output / f"runs/{run.id}"),
        )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=str(full_output),
        log_path=str(full_output),
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=False,
        render=True,
    )

    wandb_callback = WandbCallback(model_save_path=str(full_output / f"models/{run.id}"))

    # model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.learn(total_timesteps=total_timesteps, callback=wandb_callback)
    model.save(str(full_output / "final_model"))
    env.close()


if __name__ == "__main__":
    train()
