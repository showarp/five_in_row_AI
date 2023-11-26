from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback,EvalCallback 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
import torch
import os

N_STEP          = 225
BATCH_SIZE      = 512
N_EPOCHS        = 10
GAMMA           = 0.99
LEARNING_RATE   = 2e-4
TOTAL_TIMESTEP  = 1e5
SAVE_FREQ       = 10000
EVN_NUMS        = 10
PLAYERS         = ['black','white']
BLACK_MODEL     = './model/black_model.zip'
WHITE_MODEL     = './model/white_model.zip'

policy_kwargs={"net_arch" : [256, 512, 256]}

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
        env = DummyVecEnv([lambda :Monitor(gym.make("five_in_roll_white-v0")) for _ in range(EVN_NUMS)])
        checkpoint_callback = CheckpointCallback(
          save_freq=max(SAVE_FREQ//EVN_NUMS,1),
          save_path="./checkpoint_white/",
          name_prefix="ppo_FIR_white"
        )
        eval_callback = EvalCallback(env,
                                      best_model_save_path="./best_white/",
                                      log_path="./best_whtie_logs/",
                                      eval_freq=500,
                                      render=False
                                      ) 
    else:
        env = DummyVecEnv([lambda :Monitor(gym.make("five_in_roll_black-v0")) for _ in range(EVN_NUMS)])
        checkpoint_callback = CheckpointCallback(
          save_freq=max(SAVE_FREQ//EVN_NUMS,1),
          save_path="./checkpoint_black/",
          name_prefix="ppo_FIR_black"
        )
        eval_callback = EvalCallback(env,
                                      best_model_save_path="./best_black/",
                                      log_path="./best_balck_logs/",
                                      eval_freq=500,
                                      render=False
                                      ) 

    model = MaskablePPO(
        "CnnPolicy", 
        env,
        device="cuda", 
        verbose=1,
        n_steps=N_STEP,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        policy_kwargs = policy_kwargs,
        tensorboard_log="logs",
    )
    if train_player=='white':
        if os.path.exists(WHITE_MODEL):
            model.load(WHITE_MODEL)
        model.learn(total_timesteps=TOTAL_TIMESTEP,callback=[checkpoint_callback,eval_callback],progress_bar=True)
        model.save(WHITE_MODEL)
    else:
        if os.path.exists(BLACK_MODEL):
            model.load(BLACK_MODEL)
        model.learn(total_timesteps=TOTAL_TIMESTEP,callback=[checkpoint_callback,eval_callback],progress_bar=True)
        model.save(BLACK_MODEL)

    del model

if __name__ == '__main__':
  for i in PLAYERS*100:
      torch.cuda.empty_cache()
      print('now train is :',i)
      train(i)