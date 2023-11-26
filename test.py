from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from gymnasium.envs.registration import register
import gymnasium as gym
import numpy as np

N_STEP          = 512
BATCH_SIZE      = 512
N_EPOCHS        = 4
GAMMA           = 0.94
LEARNING_RATE   = 5e-4
TEST_PLAYER     = 'ai'#ai/human

if TEST_PLAYER=="ai":
    black_model_path = './model/black_model.zip'
    black_model = MaskablePPO.load(black_model_path)
register(
    id="five_in_roll_black-v0",
    entry_point="fir_env_black:FiveInRowBlackEnv"
)

env = gym.make('five_in_roll_black-v0')
while True:
    obs,info = env.reset()
    env.render()
    done = False
    while not done:
        if TEST_PLAYER=="ai":
            print("\n\nAI VS AI")
            input('回车以继续下一回合......')
            action, _ = black_model.predict(observation=obs)
        else:
            print("\n\nHuman VS AI")
            xy = input("输入落子点(例如 3J)：")
            # x,y = xy[:-1],xy[-1]
            # x,y = int(x),ord(y)-ord("A")
            # action = x*15+y%15
            action = env.action_space.sample()
        obs,reward,done,Terminated,info = env.step(action)
        env.render()
        print(reward)
        if done:break
env.close()

#关于棋子下重复位置的解决问题
