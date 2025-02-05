from stable_baselines3.common.policies import TD3Policy, TD3CnnPolicy, register_policy, TD3MultiInputPolicy
from stable_baselines3.augmented_policies.policies import AugmentedTD3Policy

MlpPolicy = TD3Policy
CnnPolicy = TD3CnnPolicy
MultiInputPolicy = TD3MultiInputPolicy


register_policy("MlpPolicy", MlpPolicy)
register_policy("CnnPolicy", CnnPolicy)
register_policy("MultiInputPolicy", MultiInputPolicy)

AmlpPolicy=AugmentedTD3Policy
register_policy("AmlpPolicy", AmlpPolicy)
