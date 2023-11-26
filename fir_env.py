from typing import Any, Tuple
from check_reward import check_reward
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from fir_game import FirInRowGame
import os
import random

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback,EvalCallback 
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

N_STEP          = 225
BATCH_SIZE      = 512
N_EPOCHS        = 10
GAMMA           = 0.99
LEARNING_RATE   = 2e-4
TOTAL_TIMESTEP  = 1e5
SAVE_FREQ       = 10000
EVN_NUMS        = 1
policy_kwargs={"net_arch" : [256, 512, 256]}

class FiveInRowEnv(gym.Env,FirInRowGame):
    def __init__(self,render_mode=None):
        FirInRowGame.__init__(self)
        self.now_player = True
        self.step_log = [[None,None],[None,None]]
        self.now_down = [None,None]
        self.render_mode=render_mode

        self.observation_space = spaces.Box(0,255,(3,37,37),dtype=np.uint8)
        self.action_space = spaces.Discrete(15*15)


    def random_think(self):
        invalid_board = self.chess_board[self.INVALID_CHANNEL]
        random_space = []
        for i in range(15):
            for j in range(15):
                if invalid_board[i,j]==0:random_space.append([i,j])
        x,y = random.sample(random_space,1)[0]
        action = np.zeros((2,15))
        action[0][x] = 1
        action[1][y] = 1
        if self.now_player:
            self.down_black_chess(x,y)
        else:
            self.down_white_chess(x,y)
    
    def __get_info(self):
        return {
            "now_player":self.now_player,
            "step_log":self.step_log
        }

    def __get_obs(self):
        return np.pad(self.chess_board,[[0,0],[11,11],[11,11]])

    def __get_done(self):
        lastX,lastY = self.step_log[-1]
        if lastX==lastY==None:return False
        return self.check_win(lastX,lastY) in ["Black wins","White wins","Draw"]
    
    def __get_terminated(self):
        return len(self.step_log)>225

    def __get_reward(self,nwp,x,y):
        ct = 1 if nwp else 2
        reward = check_reward(self.chess_board[self.BOARD_CHANNEL, :, :],ct,x,y)
        return reward

    def action_masks(self):
        return (1-self.chess_board[self.INVALID_CHANNEL]).flatten().astype(np.bool_)

    def reset(self,seed=None,options=None) -> Tuple[Any, dict]:
        super().reset(seed=seed)
        self.__init__()
        obs = self.__get_obs()
        info = self.__get_info()
        return obs,info
    
    def render(self):
            os.system('cls' if os.name == 'nt' else 'clear')
            print(FirInRowGame.__str__(self))
            print(f"当前轮到:{'Black' if self.now_player else 'White'}")
            x,y = self.__get_info()["step_log"][-2]
            y = " None" if y==None else chr(ord("A")+y)
            print(f"上一个落子位置:{x}{y}")

    def close(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        return super().close()
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        x,y = action//15,action%15
        is_valid = self.play(x,y)
        done = self.__get_done()
        obs = self.__get_obs()
        # reward = self.__get_reward(not self.get_now_player(),x,y) if is_valid else 0
        if not is_valid:
            reward = -500
        else:
            if done:reward=10000000000
            else:reward = -len(self.step_log)

        Terminated = self.__get_terminated()
        info = self.__get_info()
        return obs,reward,done,Terminated,info


check_env(FiveInRowEnv())
env = FiveInRowEnv(render_mode='human')
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
eval_callback = EvalCallback(FiveInRowEnv(render_mode='human'),
                                best_model_save_path="./best_model/",
                                log_path="./best_logs/",
                                eval_freq=500,
                                deterministic=False,
                                render=True
                                ) 
model.learn(total_timesteps=TOTAL_TIMESTEP,callback=[eval_callback],progress_bar=True)
# model.save('./model.zip')


model.load('./model.zip')
while True:
    obs,info = env.reset()
    env.render()
    done = False
    while not done:
        action, _ = model.predict(observation=obs)
        obs,reward,done,Terminated,info = env.step(action)
        env.render()
        print(reward)
        input()
        if done:break