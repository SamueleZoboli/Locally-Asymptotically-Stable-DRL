from stable_baselines3.common.policies import SACPolicy,SACCnnPolicy, SACMultiInputPolicy, register_policy
from stable_baselines3.augmented_policies.policies import AugmentedSACPolicy

MlpPolicy = SACPolicy
CnnPolicy = SACCnnPolicy
MultiInputPolicy = SACMultiInputPolicy

register_policy("MlpPolicy", MlpPolicy)
register_policy("CnnPolicy", CnnPolicy)
register_policy("MultiInputPolicy", MultiInputPolicy)

AmlpPolicy = AugmentedSACPolicy
register_policy("AmlpPolicy", AmlpPolicy)
