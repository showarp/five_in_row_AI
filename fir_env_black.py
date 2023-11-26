import copy
from typing import Any, Tuple
from check_reward import check_reward
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from fir_game import FirInRowGame
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
import os
import random

class FiveInRowBlackEnv(gym.Env,FirInRowGame):
    
    def __init__(self,render_mode=None):
        self.WHITE_MODEL_PATH = './model/white_model.zip'

        FirInRowGame.__init__(self)
        self.step_log = [[None,None],[None,None]]
        self.now_down = [None,None]
        self.render_mode=render_mode
        self.observation_space = spaces.Box(0,255,(3,37,37),dtype=np.uint8)
        self.action_space = spaces.Discrete(15*15)
        
        if self.__check_exist_model():
            self.white_model = MaskablePPO.load(self.WHITE_MODEL_PATH)
            print('=================================================')
            print("load WHITE MODEL")
            print('=================================================')

    def think(self):
        invalid_board = self.chess_board[self.INVALID_CHANNEL]
        check_board = copy.deepcopy(self.chess_board[self.BOARD_CHANNEL])
        reward_list = []
        for i in range(15):
            for j in range(15):
                if invalid_board[i,j]==0:
                    check_board[i,j] = 2
                    reward_list.append(((i,j),check_reward(check_board,2,i,j)))
                    check_board[i,j] = 0
        reward_list.sort(key = lambda x:x[1],reverse=True)
        x,y = reward_list[2][0]
        if self.get_now_player():
            self.down_black_chess(x,y)
        else:
            self.down_white_chess(x,y)

    def random_think(self):
        random_space = []
        invalid_board = self.chess_board[self.INVALID_CHANNEL]
        for i in range(15):
            for j in range(15):
                if invalid_board[i,j]==0:random_space.append([i,j])
        x,y = random.sample(random_space,1)[0]
        if self.get_now_player():
            self.down_black_chess(x,y)
        else:
            self.down_white_chess(x,y)
    
    def white_ai_think(self):
        action,_ = self.white_model.predict(self.__get_obs())
        x,y = action//15,action%15
        if self.get_now_player():
            is_valid = self.down_black_chess(x,y)
        else:
            is_valid = self.down_white_chess(x,y)
        return is_valid
    
    def __check_exist_model(self):
        return os.path.exists(self.WHITE_MODEL_PATH)
    
    def __get_info(self):
        return {
            "now_player":self.get_now_player(),
            "step_log":self.step_log
        }

    def __get_obs(self):
        return np.pad(self.chess_board,[[0,0],[11,11],[11,11]])

    def __get_done(self):
        if len(self.step_log)>225:return True
        lastX,lastY = self.step_log[-1]
        if lastX==lastY==None:return False
        return self.check_win(lastX,lastY) in ["Black wins","White wins","Draw"]
    
    def __get_reward(self,nwp,x,y):
        ct = 1 if nwp else 2
        reward = check_reward(self.chess_board[self.BOARD_CHANNEL, :, :],ct,x,y)
        return reward
    
    def action_masks(self):
        return (1-self.chess_board[self.INVALID_CHANNEL]).flatten()

    def reset(self,seed=None,options=None) -> Tuple[Any, dict]:
        super().reset(seed=seed)
        self.__init__()
        obs = self.__get_obs()
        info = self.__get_info()
        return obs,info
    
    def render(self):
            os.system('cls' if os.name == 'nt' else 'clear')
            print(FirInRowGame.__str__(self))
            print(f"当前轮到:{'Black' if self.get_now_player() else 'White'}")
            x,y = self.__get_info()["step_log"][-2]
            y = " None" if y==None else chr(ord("A")+y)
            print(f"上一个落子位置:{x}{y}")

    def close(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        return super().close()
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        x,y = action//15,action%15
        done = self.__get_done()
        is_valid = self.play(x,y)
        obs = self.__get_obs()
        reward = self.__get_reward(not self.get_now_player(),x,y) if is_valid else -4320
        truncations = False
        info = self.__get_info()
        if is_valid and not self.__check_exist_model():
            self.random_think()
        if is_valid and self.__check_exist_model():
            self.white_ai_think()
        return obs,reward,done,truncations,info