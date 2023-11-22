from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
import torch
import os

N_STEP          = 512
BATCH_SIZE      = 512
N_EPOCHS        = 4
GAMMA           = 0.94
LEARNING_RATE   = 5e-4
TOTAL_TIMESTEP  = 1e5
SAVE_FREQ       = 10000
EVN_NUMS        = 10
PLAYERS         = ['black','white']
BLACK_MODEL     = './model/black_model.zip'
WHITE_MODEL     = './model/white_model.zip'


register(
    id="five_in_roll_black-v0",
    entry_point="fir_env_black:FiveInRowBlackEnv"
)
register(
    id="five_in_roll_white-v0",
    entry_point="fir_env_white:FiveInRowWhiteEnv"
)

check_env(gym.make("five_in_roll_white-v0"))
check_env(gym.make("five_in_roll_black-v0"))

def train(train_player=None):
    if train_player=='white':
        checkpoint_callback = CheckpointCallback(
          save_freq=max(SAVE_FREQ//EVN_NUMS,1),
          save_path="./checkpoint_white/",
          name_prefix="ppo_FIR_white"
        )
        env = DummyVecEnv([lambda :Monitor(gym.make("five_in_roll_white-v0")) for _ in range(EVN_NUMS)])
    else:
        checkpoint_callback = CheckpointCallback(
          save_freq=max(SAVE_FREQ//EVN_NUMS,1),
          save_path="./checkpoint_black/",
          name_prefix="ppo_FIR_black"
        )
        env = DummyVecEnv([lambda :Monitor(gym.make("five_in_roll_black-v0")) for _ in range(EVN_NUMS)])

    model = PPO(
        "CnnPolicy", 
        env,
        device="cuda", 
        verbose=1,
        n_steps=N_STEP,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        tensorboard_log="logs",
    )
    if train_player=='white':
        if os.path.exists(WHITE_MODEL):
            model.load(WHITE_MODEL)
        model.learn(total_timesteps=TOTAL_TIMESTEP,callback=[checkpoint_callback])
        model.save(WHITE_MODEL)
    else:
        if os.path.exists(BLACK_MODEL):
            model.load(BLACK_MODEL)
        model.learn(total_timesteps=TOTAL_TIMESTEP,callback=[checkpoint_callback])
        model.save(BLACK_MODEL)

    del model

if __name__ == '__main__':
  for i in PLAYERS*100:
      torch.cuda.empty_cache()
      print('now train is :',i)
      train(i)