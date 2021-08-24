from functools import partial
from typing import Any, Dict, Optional, Tuple, Type

import gym
import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, MlpExtractor


class CustomMlpExtractor(MlpExtractor):
    def forward(
        self,
        policy_features: th.Tensor,
        value_features: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return (
            self.policy_net(self.shared_net(policy_features)),
            self.value_net(self.shared_net(value_features))
        )


class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    :param policy_features_extractor_class: Features extractor to use for policy network.
    :param policy_features_extractor_kwargs: Keyword arguments to pass to the policy features extractor.
    :param value_features_extractor_class: Features extractor to use for value network.
    :param value_features_extractor_kwargs: Keyword arguments to pass to the value features extractor.
    """
    def __init__(
        self,
        *args,
        policy_features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        policy_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        value_features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        value_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if policy_features_extractor_kwargs is None:
            policy_features_extractor_kwargs = {}

        if value_features_extractor_kwargs is None:
            value_features_extractor_kwargs = {}

        self.policy_features_extractor_class = policy_features_extractor_class
        self.policy_features_extractor_kwargs = policy_features_extractor_kwargs

        self.value_features_extractor_class = value_features_extractor_class
        self.value_features_extractor_kwargs = value_features_extractor_kwargs

        super().__init__(*args, **kwargs)

    def extract_features(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.policy_features_extractor(preprocessed_obs), self.value_features_extractor(preprocessed_obs)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        del data["features_extractor_class"]
        del data["features_extractor_kwargs"]

        data.update(
            dict(
                policy_features_extractor_class=self.policy_features_extractor_class,
                policy_features_extractor_kwargs=self.policy_features_extractor_kwargs,
                value_features_extractor_class=self.value_features_extractor_class,
                value_features_extractor_kwargs=self.value_features_extractor_kwargs,
            )
        )
        return data

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = CustomMlpExtractor(
            self.policy_features_extractor.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, *args, **kwargs) -> None:
        self.policy_features_extractor = self.policy_features_extractor_class(self.observation_space, **self.policy_features_extractor_kwargs)
        self.value_features_extractor = self.value_features_extractor_class(self.observation_space, **self.value_features_extractor_kwargs)

        assert self.policy_features_extractor.features_dim == self.value_features_extractor.features_dim
        super()._build(*args, **kwargs)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.policy_features_extractor: np.sqrt(2),
                self.value_features_extractor: np.sqrt(2),
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

    def _get_latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # Preprocess the observation if needed
        policy_features, value_features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(policy_features, value_features)

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            raise RuntimeError("Not implemented")
        return latent_pi, latent_vf, latent_sde
