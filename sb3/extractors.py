import gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer


class MicroRTSExtractor(BaseFeaturesExtractor):
    """
    Reorders input features before performing convolution.

    Based on the following:
    https://github.com/vwxyzjn/invalid-action-masking/blob/c0d47cca3c2d8522ce97412b76ca4e4e36c5d95e/invalid_action_masking/ppo_4x4.py#L232
    """

    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=1)

        n_input_channels = observation_space.shape[-1]
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=2)),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            obs = observation_space.sample()[None]
            x = th.Tensor(np.moveaxis(obs, -1, 1)).float()
            features_dim = self.cnn(x).shape[1]

        self._features_dim = features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn(observations.permute(0, 3, 1, 2))


def make_extractor_class(map_size: int) -> BaseFeaturesExtractor:
    VALID_SIZES = ["4", "10"]

    if map_size not in VALID_SIZES:
        raise ValueError(f"Invalid map size {map_size}. Options: f{VALID_SIZES}")

    class MicroRTSExtractor(BaseFeaturesExtractor):
        """
        Reorders input features before performing convolution.

        Based on the following:
        https://github.com/vwxyzjn/invalid-action-masking/blob/c0d47cca3c2d8522ce97412b76ca4e4e36c5d95e/invalid_action_masking/ppo_4x4.py#L232
        """

        def __init__(self, observation_space: gym.spaces.Box):
            super().__init__(observation_space, features_dim=1)

            n_input_channels = observation_space.shape[-1]
            if map_size == "10":
                self.cnn = nn.Sequential(
                    layer_init(nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=2)),
                    nn.ReLU(),
                    layer_init(nn.Conv2d(16, 32, kernel_size=2)),
                    nn.Flatten(),
                )
            elif map_size == "4":
                self.cnn = nn.Sequential(
                    layer_init(nn.Conv2d(n_input_channels, 16, kernel_size=2)),
                    nn.ReLU(),
                    nn.Flatten(),
                )
            else:
                raise ValueError(f"Invalid map size {map_size}. Options: f{VALID_SIZES}")

            # Compute shape by doing one forward pass
            with th.no_grad():
                obs = observation_space.sample()[None]
                x = th.Tensor(np.moveaxis(obs, -1, 1)).float()
                features_dim = self.cnn(x).shape[1]

            self._features_dim = features_dim

        def forward(self, observations: th.Tensor) -> th.Tensor:
            return self.cnn(observations.permute(0, 3, 1, 2))

    return MicroRTSExtractor
