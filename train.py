from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
checkpoint_callback = CheckpointCallback(
  save_freq=max(10000//10,1),
  save_path="./checkpoint/",
  name_prefix="ppo_FIR"
)
register(
    id="five_in_roll-v0",
    entry_point="fir_env:FiveInRowEnv"
)
check_env(gym.make("five_in_roll-v0"))
env = DummyVecEnv([lambda :Monitor(gym.make("five_in_roll-v0")) for _ in range(10)])
model = PPO(
    "CnnPolicy", 
    env,
    device="cuda", 
    verbose=1,
    n_steps=512,
    batch_size=512,
    n_epochs=4,
    gamma=0.94,
    learning_rate=5e-4,
    tensorboard_log="logs",
)
model.learn(total_timesteps=1e5,callback=[checkpoint_callback])
model.save("./model/FIR.zip")