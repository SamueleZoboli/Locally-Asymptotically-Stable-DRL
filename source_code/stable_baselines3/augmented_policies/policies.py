from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.policies import ActorCriticPolicy, ContinuousCritic, get_action_dim
from stable_baselines3.common.policies import SACActor as SacActor
from stable_baselines3.common.policies import SACPolicy 
from stable_baselines3.common.policies import TD3Actor as Td3Actor
from stable_baselines3.common.policies import TD3Policy
from stable_baselines3.augmented_policies.local_controller import LocalController


class AugmentedActorCriticPolicy(ActorCriticPolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param local_sol: Local linear controller
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            local_sol: LocalController,
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            squash_output: bool = True, # TODO: check squash_output only for the NN
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(AugmentedActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

        # discounted LQ controller
        self.local=local_sol
        self.register_buffer("k_matrix", self.local.K)
        self.register_buffer("p_matrix", self.local.P)
        self.register_buffer("low", th.from_numpy(self.action_space.low))
        self.register_buffer("high", th.from_numpy(self.action_space.high))
        self.register_buffer("c", self.local.doa)
        self.register_buffer("u_bar", self.local.u_bar)
        self.register_buffer("x_star", self.local.x_star)
        self.register_buffer("h_matrix", self.local.H)
        self.alpha = 0.3

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()
        data.update(
            dict(
                k_matrix=self.k_matrix,
                p_matrix=self.p_matrix,
                doa=self.c,
                u_bar=self.u_bar,
                x_star=self.x_star,
                h_matrix=self.h_matrix,
            )
        )
        return data

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation to feed to the controller
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        pi_x = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(pi_x)  # extracting logprobs thanks to change of input
        pi_x = self.low + (0.5 * (pi_x + 1.0) * (self.high - self.low))  # interpolation
        u_l=self.local.act(obs).type(pi_x.dtype)
        h1_x=self.local.smooth_sat(obs,self.alpha,2).type(pi_x.dtype)
        u_theta = h1_x * (pi_x-u_l) # h1 is scalar hence will be broadcasted
        actions = u_l + u_theta
        # Evaluate the values for the given observations
        v_phi = self.value_net(latent_vf)
        v_ul = self.local.evaluate(obs).type(v_phi.dtype)
        h2_x = self.local.smooth_sat(obs, self.alpha, 3).type(v_phi.dtype)
        values = -v_ul + h2_x * (v_phi+v_ul) # h2 is scalar hence will be broadcasted
        a_sample = th.tensor(self.action_space.sample()).to(self.device)
        return actions.type_as(a_sample), values, log_prob

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation: Input vector to the controller
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        latent_pi, _, latent_sde = self._get_latent(observation)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        pi_x = distribution.get_actions(deterministic=deterministic)
        pi_x = self.low + (0.5 * (pi_x + 1.0) * (self.high - self.low))  # interpolation
        u_l = self.local.act(observation).type(pi_x.dtype)
        h1_x = self.local.smooth_sat(observation, self.alpha, 2).type(pi_x.dtype)
        u_theta = h1_x * (pi_x - u_l)
        actions = u_l + u_theta
        a_sample = th.tensor(self.action_space.sample()).to(self.device)
        return actions.type_as(a_sample)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        u_l = self.local.act(obs).type(actions.dtype)
        h1_x = self.local.smooth_sat(obs, self.alpha, 2).type(actions.dtype)
        u_theta = (actions - u_l)
        pi_x = u_theta / h1_x + u_l
        pi_x = 2.0 * ((pi_x - self.low) / (self.high - self.low)) - 1.0 # back to squashed
        log_prob = distribution.log_prob(pi_x)  # logprob of the policy can be derived by the one of pi
        v_phi = self.value_net(latent_vf)
        v_ul = self.local.evaluate(obs).type(v_phi.dtype)
        h2_x = self.local.smooth_sat(obs, self.alpha, 3).type(v_phi.dtype)
        values = -v_ul + h2_x * (v_phi + v_ul)
        return values, log_prob, distribution.entropy()


class AugmentedSacActor(SacActor):
    """
    Augmented actor network (policy) for SAC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param local_sol: Local linear controller
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            net_arch: List[int],
            features_extractor: nn.Module,
            features_dim: int,
            local_sol: LocalController,
            activation_fn: Type[nn.Module] = nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -3,
            full_std: bool = True,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            normalize_images: bool = True,
    ):
        super(AugmentedSacActor, self).__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            clip_mean,
            normalize_images
        )

        # Local LQ controller
        self.local = local_sol
        self.register_buffer("k_matrix", self.local.K)
        self.register_buffer("p_matrix", self.local.P)
        self.register_buffer("low", th.from_numpy(self.action_space.low))
        self.register_buffer("high", th.from_numpy(self.action_space.high))
        self.register_buffer("c", self.local.doa)
        self.register_buffer("u_bar", self.local.u_bar)
        self.register_buffer("x_star", self.local.x_star)
        self.register_buffer("h_matrix", self.local.H)
        self.alpha = 0.3

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(
            dict(
                k_matrix=self.k_matrix,
                p_matrix=self.p_matrix,
                doa=self.c,
                u_bar=self.u_bar,
                x_star=self.x_star,
                h_matrix=self.h_matrix,
            )
        )
        return data

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is not squashed but pi_x is
        pi_x = self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)
        if not np.isfinite(pi_x).all():
            print(mean_actions, log_std, kwargs)
        pi_x = self.low + (0.5 * (pi_x + 1.0) * (self.high - self.low)) #interpolation
        if not np.isfinite(pi_x).all():
            print(obs)
        u_l = self.local.act(obs).type(pi_x.dtype)
        h1_x = self.local.smooth_sat(obs, self.alpha, 2).type(pi_x.dtype)
        u_theta = h1_x * (pi_x - u_l)
        actions = u_l + u_theta
        a_sample = th.tensor(self.action_space.sample()).to(self.device)

        return actions.type_as(a_sample)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        pi_x, log_prob = self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)
        pi_x = self.low + (0.5 * (pi_x + 1.0) * (self.high - self.low))
        u_l = self.local.act(obs).type(pi_x.dtype)
        h1_x = self.local.smooth_sat(obs, self.alpha, 2).type(pi_x.dtype)
        u_theta = h1_x * (pi_x - u_l)
        actions = u_l + u_theta
        a_sample = th.tensor(self.action_space.sample()).to(self.device)
        return actions.type_as(a_sample), log_prob


class AugmentedContinuousCritic(ContinuousCritic):
    """
        Augmented Critic network(s) for DDPG/SAC/TD3.
        It represents the action-state value function (Q-value function).
        Compared to A2C/PPO critics, this one represents the Q-value
        and takes the continuous action as input. It is concatenated with the state
        and then fed to the network which outputs a single value: Q^phi(x, u), then
        the linear framework contribution Q^uL(x,u) is added.
        For more recent algorithms like SAC/TD3, multiple networks
        are created to give different estimates.

        By default, it creates two critic networks used to reduce overestimation
        thanks to clipped Q-learning (cf TD3 paper).

        :param observation_space: Obervation space
        :param action_space: Action space
        :param net_arch: Network architecture
        :param features_extractor: Network to extract features
            (a CNN when using images, a nn.Flatten() layer otherwise)
        :param features_dim: Number of features
        :param local_sol: Local linear controller
        :param activation_fn: Activation function
        :param normalize_images: Whether to normalize images or not,
             dividing by 255.0 (True by default)
        :param n_critics: Number of critic networks to create.
        :param share_features_extractor: Whether the features extractor is shared or not
            between the actor and the critic (this saves computation time)
        """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            net_arch: List[int],
            features_extractor: nn.Module,
            features_dim: int,
            local_sol: LocalController,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
            n_critics: int = 2,
            share_features_extractor: bool = True,
    ):
        super(AugmentedContinuousCritic, self).__init__(
            observation_space,
            action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor
        )

        # Local LQ controller
        self.local = local_sol
        self.register_buffer("k_matrix", self.local.K)
        self.register_buffer("p_matrix", self.local.P)
        self.register_buffer("low", th.from_numpy(self.action_space.low))
        self.register_buffer("high", th.from_numpy(self.action_space.high))
        self.register_buffer("c", self.local.doa)
        self.register_buffer("u_bar", self.local.u_bar)
        self.register_buffer("x_star", self.local.x_star)
        self.register_buffer("h_matrix", self.local.H)
        self.alpha = 0.3


    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()
        data.update(
            dict(
                k_matrix=self.k_matrix,
                p_matrix=self.p_matrix,
                doa=self.c,
                u_bar=self.u_bar,
                x_star=self.x_star,
                h_matrix=self.h_matrix,
            )
        )
        return data

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Computes the linear contribution q_ul and blends it with the learned one
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, actions], dim=1)
        values = []
        for q_net in self.q_networks:
            q_phi = q_net(qvalue_input)
            q_ul = self.local.evaluate(obs,actions).type(q_phi.dtype)
            h2_x = self.local.smooth_sat(obs, self.alpha, 3).type(q_phi.dtype)
            value = -q_ul + h2_x * (q_phi + q_ul) # h2 is broadcasted since it is a scalar
            values.append(value)
        return tuple(values)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, actions], dim=1)
        q_phi = self.q_networks[0](qvalue_input)
        q_ul = self.local.evaluate(obs, actions).type(q_phi.dtype)
        h2_x = self.local.smooth_sat(obs, self.alpha, 3).type(q_phi.dtype)
        value = -q_ul + h2_x * (q_phi + q_ul) # h2 is broadcasted since it is a scalar
        return value


class AugmentedSACPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for Augmented SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param local_sol: Local linear controller
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable,
            local_sol: LocalController,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -3,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = True,
    ):
        self.local = local_sol

        super(AugmentedSACPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
            squash_output= False # TODO:check how to deal with squashoutput
        )

        # discounted LQ controller
        self.register_buffer("k_matrix", self.local.K)
        self.register_buffer("p_matrix", self.local.P)
        self.register_buffer("low", th.from_numpy(self.action_space.low))
        self.register_buffer("high", th.from_numpy(self.action_space.high))
        self.register_buffer("c", self.local.doa)
        self.register_buffer("u_bar", self.local.u_bar)
        self.register_buffer("x_star", self.local.x_star)
        self.register_buffer("h_matrix", self.local.H)
        self.alpha = 0.3

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(
            dict(
                k_matrix=self.k_matrix,
                p_matrix=self.p_matrix,
                doa=self.c,
                u_bar=self.u_bar,
                x_star=self.x_star,
                h_matrix=self.h_matrix,
            )
        )
        return data

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> AugmentedSacActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor = AugmentedSacActor(local_sol=self.local,**actor_kwargs).to(self.device)
        return actor

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> AugmentedContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic = AugmentedContinuousCritic(local_sol=self.local,**critic_kwargs).to(self.device)
        return critic

    def unscale_action(self, scaled_pi: np.ndarray, obs: np.ndarray) -> np.ndarray:
        # action to be used considering the new pi
        low, high = self.action_space.low, self.action_space.high
        pi_x = low + (0.5 * (scaled_pi + 1.0) * (high - low))
        u_l = self.local.act(th.from_numpy(obs).to(self.device)).cpu().numpy().astype(pi_x.dtype)
        h1_x = self.local.smooth_sat(th.from_numpy(obs).to(self.device), self.actor.alpha, 2).cpu().numpy().astype(pi_x.dtype)
        u_theta = h1_x * (pi_x - u_l)
        action = u_l + u_theta
        return action.astype(self.action_space.dtype)

    def scale_action(self, action: np.ndarray, obs: np.ndarray) -> np.ndarray:
        # extract the output of the network (which is in [-1,1])
        u_l = self.local.act(th.from_numpy(obs).to(self.device)).cpu().numpy().astype(action.dtype)
        h1_x = self.local.smooth_sat(th.from_numpy(obs).to(self.device), self.actor.alpha, 2).cpu().numpy().astype(action.dtype)
        u_theta = (action - u_l)
        pi_x = u_theta / h1_x + u_l
        low, high = self.action_space.low, self.action_space.high
        pi_x = 2.0 * ((pi_x - low) / (high - low)) - 1.0
        return pi_x.astype(self.action_space.dtype)


class AugmentedTd3Actor(Td3Actor):
    """
    Augmented actor network (policy) for TD3.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param local_sol: Local linear controller
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            local_sol: LocalController,
            net_arch: List[int],
            features_extractor: nn.Module,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
    ):
        super(AugmentedTd3Actor, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
            squash_output=True # TODO: squash?
        )

        # Local LQ controller
        self.local = local_sol
        self.register_buffer("k_matrix", self.local.K)
        self.register_buffer("p_matrix", self.local.P)
        self.register_buffer("low", th.from_numpy(self.action_space.low))
        self.register_buffer("high", th.from_numpy(self.action_space.high))
        self.register_buffer("c", self.local.doa)
        self.register_buffer("u_bar", self.local.u_bar)
        self.register_buffer("x_star", self.local.x_star)
        self.register_buffer("h_matrix", self.local.H)
        self.alpha = 0.3

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()
        data.update(
            dict(
                k_matrix=self.k_matrix,
                p_matrix=self.p_matrix,
                doa=self.c,
                u_bar=self.u_bar,
                x_star=self.x_star,
                h_matrix=self.h_matrix,
            )
        )
        return data

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        # The action is composed by a squashed-interpolated learned output pi_x
        # and a linear component u_l
        features = self.extract_features(obs)
        pi_x = self.mu(features)
        pi_x = self.low + (0.5 * (pi_x + 1.0) * (self.high - self.low)) # scale pi to action space
        u_l = self.local.act(obs).type(pi_x.dtype)
        h1_x = self.local.smooth_sat(obs, self.alpha, 2).type(pi_x.dtype)
        u_theta = h1_x * (pi_x - u_l) # h1 is broadcasted since it is a scalar
        actions = u_l + u_theta
        a_sample = th.tensor(self.action_space.sample()).to(self.device)
        return actions.type_as(a_sample)

    def pi_from_action(self, action: th.Tensor, obs: th.Tensor) -> th.Tensor:
        # extract the output of the network (which is in [-1,1])
        u_l = self.local.act(obs).type(action.dtype)
        h1_x = self.local.smooth_sat(obs, self.alpha, 2).type(action.dtype)
        u_theta = (action - u_l)
        pi_x = u_theta / h1_x + u_l
        pi_x = 2.0 * ((pi_x - self.low) / (self.high - self.low)) - 1.0
        a_sample = th.tensor(self.action_space.sample()).to(self.device)
        return pi_x.type_as(a_sample)

    def action_from_pi(self, pi: th.Tensor, obs: th.Tensor) -> th.Tensor:
        # action to be used considering the new pi
        pi_x = self.low + (0.5 * (pi + 1.0) * (self.high - self.low))
        u_l = self.local.act(obs).type(pi_x.dtype)
        h1_x = self.local.smooth_sat(obs, self.alpha, 2).type(pi_x.dtype)
        u_theta = h1_x * (pi_x - u_l)
        action = u_l + u_theta
        a_sample=th.tensor(self.action_space.sample()).to(self.device)
        return action.type_as(a_sample)


class AugmentedTD3Policy(TD3Policy):
    """
        Augmented policy class (with both augmented actor and critic) for TD3.

        :param observation_space: Observation space
        :param action_space: Action space
        :param lr_schedule: Learning rate schedule (could be constant)
        :param local_sol: Local linear controller
        :param net_arch: The specification of the policy and value networks.
        :param activation_fn: Activation function
        :param features_extractor_class: Features extractor to use.
        :param features_extractor_kwargs: Keyword arguments
            to pass to the features extractor.
        :param normalize_images: Whether to normalize images or not,
             dividing by 255.0 (True by default)
        :param optimizer_class: The optimizer to use,
            ``th.optim.Adam`` by default
        :param optimizer_kwargs: Additional keyword arguments,
            excluding the learning rate, to pass to the optimizer
        :param n_critics: Number of critic networks to create.
        :param share_features_extractor: Whether to share or not the features extractor
            between the actor and the critic (this saves computation time)
        """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable,
            local_sol: LocalController,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = True,
    ):
        self.local = local_sol

        super(AugmentedTD3Policy, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
            squash_output=False
        )

        # discounted LQ controller
        self.register_buffer("k_matrix", self.local.K)
        self.register_buffer("p_matrix", self.local.P)
        self.register_buffer("low", th.from_numpy(self.action_space.low))
        self.register_buffer("high", th.from_numpy(self.action_space.high))
        self.register_buffer("c", self.local.doa)
        self.register_buffer("u_bar", self.local.u_bar)
        self.register_buffer("x_star", self.local.x_star)
        self.register_buffer("h_matrix", self.local.H)
        self.alpha = 0.3

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()
        data.update(
            dict(
                k_matrix=self.k_matrix,
                p_matrix=self.p_matrix,
                doa=self.c,
                u_bar=self.u_bar,
                x_star=self.x_star,
                h_matrix=self.h_matrix,
            )
        )
        return data

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> AugmentedTd3Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return AugmentedTd3Actor(local_sol=self.local,**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> AugmentedContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return AugmentedContinuousCritic(local_sol=self.local,**critic_kwargs).to(self.device)

    def unscale_action(self, scaled_pi: np.ndarray, obs: np.ndarray) -> np.ndarray:
        # action to be used considering the new pi
        low, high = self.action_space.low, self.action_space.high
        pi_x = low + (0.5 * (scaled_pi + 1.0) * (high - low))
        u_l = self.local.act(th.from_numpy(obs).to(self.device)).cpu().numpy().astype(pi_x.dtype)
        h1_x = self.local.smooth_sat(th.from_numpy(obs).to(self.device), self.actor.alpha, 2).cpu().numpy().astype(pi_x.dtype)
        u_theta = h1_x * (pi_x - u_l) # h1 is broadcasted since it is a scalar
        action = u_l + u_theta
        return action.astype(self.action_space.dtype)

    def scale_action(self, action: np.ndarray, obs: th.Tensor) -> np.ndarray:
        # extract the output of the network (which is in [-1,1])
        u_l = self.local.act(th.from_numpy(obs).to(self.device)).cpu().numpy().astype(action.dtype)
        h1_x = self.local.smooth_sat(th.from_numpy(obs).to(self.device), self.actor.alpha, 2).cpu().numpy().astype(action.dtype)
        u_theta = (action - u_l)
        pi_x = u_theta / h1_x + u_l # h1 is broadcasted since it is a scalar
        low, high = self.action_space.low, self.action_space.high
        pi_x = 2.0 * ((pi_x - low) / (high - low)) - 1.0
        return pi_x.astype(self.action_space.dtype)
