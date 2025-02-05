# This file is here just to define MlpPolicy/CnnPolicy
# that work for A2C
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    MultiInputActorCriticPolicy,
    register_policy,
)

from stable_baselines3.augmented_policies.policies import AugmentedActorCriticPolicy


MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy

register_policy("MlpPolicy", ActorCriticPolicy)
register_policy("CnnPolicy", ActorCriticCnnPolicy)
register_policy("MultiInputPolicy", MultiInputPolicy)


AmlpPolicy = AugmentedActorCriticPolicy
register_policy("AmlpPolicy", AugmentedActorCriticPolicy)
