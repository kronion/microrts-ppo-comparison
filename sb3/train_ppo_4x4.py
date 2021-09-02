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
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from extractors import MicroRTSExtractor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env


class Defaults:
    TOTAL_TIMESTEPS = 10000000
    EVAL_FREQ = 10000
    EVAL_EPISODES = 10
    SEED = 42

    # Parameters specified by this paper: https://arxiv.org/pdf/2006.14171.pdf
    ENTROPY_COEF = 0.01


gamma = 0.99
gae_lambda = 0.95
clip_coef = 0.1
max_grad_norm = 0.5
learning_rate = 2.5e-4

def mask_fn(env: gym.Env) -> np.ndarray:
    # Disable mask:
    # return np.ones_like(env.action_mask)
    return env.action_mask

def get_wrapper(env: gym.Env): -> gym.Env
    return ActionMasker(env, mask_fn)

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
        project="invalidActions",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        anonymous="true",  # wandb documentation is wrong...
    )

    base_output = Path(output_folder)
    full_output = base_output / datetime.datetime.now().isoformat(timespec="seconds")
    logger.configure(folder=str(full_output))

    env_id = "MicrortsMining10x10F9-v0"
    n_envs = 8
    env = make_vec_env(env_id, n_envs=n_envs, wrapper_class=get_wrapper)
    env = VecNormalize(env, norm_reward=False)

    eval_env = make_vec_env(env_id, n_envs=10, wrapper_class=get_wrapper)
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False)

    eval_callback = MaskableEvalCallback(eval_env, eval_freq=eval_freq, n_eval_episodes=eval_episodes)

    if True:
        lr = lambda progress_remaining: progress_remaining * learning_rate

    if False:
        clip_range_vf = None
    else:
        clip_range_vf = clip_coef

    if load_path:
        model = PPO.load(load_path, env)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            batch_size=256,
            n_epochs=4,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_coef,
            clip_range_vf=clip_range_vf,
            ent_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            learning_rate=lr,
            seed=seed,
            policy_kwargs={
                "net_arch": [128],
                "activation_fn": nn.ReLU,
                "features_extractor_class": MicroRTSExtractor,
                "ortho_init": True,
            },
            tensorboard_log=str(full_output / f"runs/{run.id}"),
        )

    wandb_callback = WandbCallback(model_save_path=str(full_output / f"models/{run.id}"))

    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, wandb_callback])
    model.save(str(full_output / "final_model"))
    env.close()


if __name__ == "__main__":
    train()
