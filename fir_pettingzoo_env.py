from collections import defaultdict
from typing import Any
from pettingzoo import AECEnv
from fir_game import FirInRowGame
from gymnasium import spaces
from check_reward import check_reward
import numpy as np
import os
from pettingzoo.test import api_test

class CustomEnvironment(AECEnv,FirInRowGame):

    metadata = {"render_modes": ["human"], "name": "fir","is_parallelizable":False}

    def __init__(self):
        FirInRowGame.__init__(self)
        AECEnv.__init__(self)
        super().__init__()

        self.agents = ["player_0","player_1"]
        self.possible_agents = ["player_0","player_1"]
        self.agent_selection = "player_0"
        
        self.terminations = {i:False for i in self.possible_agents}
        self.truncations = {i:False for i in self.possible_agents}
        self.rewards = {i:0 for i in self.possible_agents}
        self._cumulative_rewards = {i:0 for i in self.possible_agents}
        # self.infos = {i:{"action_mask": np.ones((15*15,),dtype=np.int8)} for i in self.possible_agents}
        self.infos = {i:{} for i in self.possible_agents}
        self.observation_spaces = {i:spaces.Box(0,2,(2,37,37),dtype=np.int8) for i in self.possible_agents}
        self.action_spaces = {i:spaces.Discrete(15*15) for i in self.possible_agents}

        self.step_log = [[None,None],[None,None]]
        self.now_down = [None,None]

    def __get_reward(self,nwp):
        ct = 1 if nwp else 2
        reward = check_reward(self.chess_board[self.BOARD_CHANNEL, :, :],ct)
        return reward
    
    def __get_done(self):
        if len(self.step_log)>225:return True
        return self.check_win() in ["Black wins","White wins","Draw"]

    def __get_info(self):
        # invalid = 1-self.chess_board[self.INVALID_CHANNEL].astype(np.int8).flatten()
        # self.infos = {i:{"action_mask": invalid} for i in self.possible_agents}
        pass
    
    def __get_next_agent(self,agent):
        return "player_0" if agent=="player_1" else "player_1"
    
    def reset(self, seed = None, options = None) -> None:
        self.__init__()

    def observe(self, agent: Any) -> Any:
        board_and_indicator = self.chess_board[:2].astype(np.int8)
        board_and_indicator = np.pad(board_and_indicator,[[0,0],[11,11],[11,11]])
        invalid = 1-self.chess_board[self.INVALID_CHANNEL].astype(np.int8).flatten()
        return board_and_indicator

    def render(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(FirInRowGame.__str__(self))
        print(f"当前轮到:{'player_0' if self.get_now_player() else 'player_1'}")
        x,y = self.__get_info()["step_log"][-2]
        y = " None" if y==None else chr(ord("A")+y)
        print(f"上一个落子位置:{x}{y}")
    
    def close(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        return super().close()
    
    def observation_space(self, agent: Any):
        return self.observation_spaces[agent]
    
    def action_space(self, agent: Any):
        return self.action_spaces[agent]
    
    def step(self, action: Any) -> None:
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return
        x,y = action//15,action%15
        agent = self.agent_selection
        self.terminations[agent] = self.__get_done()
        self.truncations[agent] = self.__get_done()
        is_valid = self.play(x,y)
        self._cumulative_rewards[agent] = 0
        self.rewards[agent] = self.__get_reward(not self.get_now_player()) if is_valid else -4320
        self.agent_selection = self.__get_next_agent(agent)
        self._accumulate_rewards()
        self.__get_info()
env = CustomEnvironment()
api_test(env)