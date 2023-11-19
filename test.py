from stable_baselines3 import PPO
from gymnasium.envs.registration import register
import gymnasium as gym
from fir_env import FiveInRowEnv
import numpy as np

N_STEP          = 512
BATCH_SIZE      = 512
N_EPOCHS        = 4
GAMMA           = 9.94
LEARNING_RATE   = 5e-4

register(
    id="five_in_roll-v0",
    entry_point="fir_env:FiveInRowEnv"
)
env = gym.make("five_in_roll-v0")
model = PPO(    
    "CnnPolicy", 
    env,
    device="cuda")


model = model.load('./checkpoint/ppo_FIR_40000_steps.zip')

for i in range(1):
    state,info = env.reset()
    env.render()
    done = False
    s = 0
    while not done:
        # if s%2==0:
        #     x,y = input().split()
        #     x = int(x)
        #     y = ord(y)-ord("A")
        #     action = np.zeros((2,15))
        #     action[0][x] = 1
        #     action[1][y] = 1
        # else:
            # action, _ = model.predict(observation=state)
        # # s+=1
        # action = env.action_space.sample()
        action, _ = model.predict(observation=state)
        obs,reward,done,Terminated,info = env.step(action)
        env.render()
        print(reward)
        input()
        if done:break
env.close()
print(reward)
