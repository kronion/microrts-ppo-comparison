import datetime
import random
from pathlib import Path

import click
import gym
import gym_microrts
import numpy as np
import torch as th
import torch.nn as nn
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common import logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.ppo import PPO
import wandb
from wandb.integration.sb3 import WandbCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from extractors import MicroRTSExtractor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker


class Defaults:
    TOTAL_TIMESTEPS = 300000
    EVAL_FREQ = 10000
    EVAL_EPISODES = 10
    SEED = 42

    # Parameters specified by this paper: https://arxiv.org/pdf/2006.14171.pdf
    ENTROPY_COEF = 0.01


gamma = 0.99
gae_lambda = 0.97
clip_coef = 0.2
max_grad_norm = 0.5
learning_rate = 0.0003


def mask_fn(env: gym.Env) -> np.ndarray:
    # Uncomment to make masking a no-op
    # return np.ones_like(env.action_mask)
    return env.action_mask


# def make_env(gym_id, seed, idx):
#     def thunk():
#         env = gym.make(gym_id)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         env.seed(seed)
#         env.action_space.seed(seed)
#         env.observation_space.seed(seed)
#         return env
#     return thunk


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
@click.option(
    "--mask/--no-mask",
    default=False,
    help="if toggled, enable invalid action masking",
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
    mask,
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

    # # Normalize env with VecNormalize
    env = Monitor(env)
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_reward=False)

    if mask:
        Alg = MaskablePPO
    else:
        Alg = PPO

    eval_env = gym.make("MicrortsMining4x4F9-v0")
    eval_env = Monitor(eval_env)
    eval_env = ActionMasker(eval_env, mask_fn)  # Wrap to enable masking
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False)

    eval_callback = MaskableEvalCallback(eval_env, eval_freq=eval_freq, n_eval_episodes=eval_episodes)

    if load_path:
        model = Alg.load(load_path, env)
    else:
        model = Alg(
            "MlpPolicy",
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

    if mask:
        model.learn(total_timesteps=total_timesteps, callback=[eval_callback, wandb_callback], use_masking=mask)
    else:
        model.learn(total_timesteps=total_timesteps, callback=[eval_callback, wandb_callback])

    model.save(str(full_output / "final_model"))
    env.close()


if __name__ == "__main__":
    train()
