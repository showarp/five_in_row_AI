from typing import Any, Tuple
from check_reward import check_reward
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from fir_game import FirInRowGame
import os
import random



class FiveInRowEnv(gym.Env,FirInRowGame):
    def __init__(self,render_mode=None):
        self.BOARD_CHANNEL = 0
        self.INDICATOR_CHANNEL = 1
        self.INVALID_CHANNEL = 2


        self.chess_board = np.zeros((3, 15, 15),dtype=np.uint8)
        self.now_player = True
        self.step_log = [[None,None],[None,None]]
        self.now_down = [None,None]
        self.render_mode=render_mode

        self.observation_space = spaces.Box(0,255,(3,37,37),dtype=np.uint8)
        self.action_space = spaces.Box(-1,1,(2,15))

    def check_chess_valid(self,x,y):
        y = self.__parser_y__(y)
        return self.chess_board[self.BOARD_CHANNEL,x,y]==0 and 15>x>=0 and 15>y>=0

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
        if len(self.step_log)>225:return True
        return self.check_win() in ["Black wins","White wins","Draw"]
    
    def __get_reward(self,nwp):
        ct = 1 if nwp else 2
        reward = check_reward(self.chess_board[self.BOARD_CHANNEL, :, :],ct)
        return reward

    def reset(self,seed=None,options=None) -> Tuple[Any, dict]:
        super().reset(seed=seed)
        self.__init__()
        obs = self.__get_obs()
        info = self.__get_info()
        return obs,info
    
    def render(self):
            os.system('cls' if os.name == 'nt' else 'clear')
            print(self)
            print(f"当前轮到:{'Black' if self.now_player else 'White'}")
            x,y = self.__get_info()["step_log"][-2]
            y = " None" if y==None else chr(ord("A")+y)
            print(f"上一个落子位置:{x}{y}")

    def close(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        return super().close()
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        x,y = action[0].argmax(),action[1].argmax()
        is_valide = True
        done = self.__get_done()
        if self.now_player:
            is_valide = self.down_black_chess(x,y)
        else:
            is_valide = self.down_white_chess(x,y)
        obs = self.__get_obs()
        reward = self.__get_reward(not self.now_player) if is_valide else -4320
        Terminated = False
        info = self.__get_info()
        if is_valide:self.random_think()
        return obs,reward,done,Terminated,info