from typing import Any, List, Optional, Tuple, Union
from check_reward import check_reward
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os



class FiveInRow(gym.Env):
    def __init__(self, player=True,type="p2p"):
        """
        player:bool,默认为True True代表黑棋(先手)否则白棋
        type:p2p|p2a,人人对战|人机对战
        """
        if type not in ["p2a",'p2p']:
            raise Exception("type 只能选p2a或者p2p")
        self.chess_board = np.zeros((15, 15))
        self.player = player
        self.now_player = True
        self.type = type
        self.step_log = [[None,None]]
        self.now_down = [None,None]

        self.observation_space = spaces.Box(0,2,(15,15),dtype=np.int32)
        self.action_space = spaces.Box(0,14,(2,),dtype=np.int32)

    def __str__(self):
        separator_line = "  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"  
        s = f"   A B C D E F G H I J K L M N O \n{separator_line}\n"
        now_down = self.step_log[-1]
        for i in range(15):
            s += f"{i:>2}|"
            for j in range(15):
                if self.chess_board[i, j] == 1.0:
                    if i==now_down[0] and j==now_down[1]:
                        s += "\033[31m○\033[0m"
                    else:s+="○"
                elif self.chess_board[i, j] == 2.0:
                    if i==now_down[0] and j==now_down[1]:
                        s += "\033[31m●\033[0m"  
                    else:s+="●"
                else:
                    s += " "  
                s += "|"
            s += "\n" + separator_line + "\n"

        return s
    def __parser_y__(self,y):
        if type(y) in (np.int32,int):
            return y
        else:
            return ord(y)-ord("A")

    def check_chess_valid(self,x,y):
        y = self.__parser_y__(y)
        return self.chess_board[x,y]==0 and 15>x>=0 and 15>y>=0
    
    def down_black_chess(self, x, y):
        y = self.__parser_y__(y)
        if not self.check_chess_valid(x,y):
            return False
        self.chess_board[x, y] = 1.0
        self.now_player = not self.now_player
        return True

    def down_white_chess(self, x, y):
        y = self.__parser_y__(y)
        if not self.check_chess_valid(x,y):
            return False
        self.chess_board[x, y] = 2.0
        self.now_player = not self.now_player
        return True

    def check_win(self):
        def consecutive_pieces(row):
            # Helper function to check for five consecutive pieces in a row
            for i in range(len(row) - 4):
                if row[i:i + 5] == [1.0, 1.0, 1.0, 1.0, 1.0]:
                    return True  # Consecutive black pieces
                elif row[i:i + 5] == [2.0, 2.0, 2.0, 2.0, 2.0]:
                    return True  # Consecutive white pieces
            return False

        # Check rows
        for i in range(15):
            if consecutive_pieces(list(self.chess_board[i, :])):
                return "Black wins" if 1.0 in list(self.chess_board[i, :]) else "White wins"

        # Check columns
        for j in range(15):
            if consecutive_pieces(list(self.chess_board[:, j])):
                return "Black wins" if 1.0 in list(self.chess_board[:, j]) else "White wins"

        # Check diagonals
        for i in range(11):
            for j in range(11):
                if consecutive_pieces([self.chess_board[i + k, j + k] for k in range(5)]):
                    return "Black wins" if 1.0 in [self.chess_board[i + k, j + k] for k in range(5)] else "White wins"

                if consecutive_pieces([self.chess_board[i + 4 - k, j + k] for k in range(5)]):
                    return "Black wins" if 1.0 in [self.chess_board[i + 4 - k, j + k] for k in range(5)] else "White wins"

    
    def play(self):
        while self.check_win()==None:
            print(self)
            operate = input("请输入您的落子位置(例如3L):")
            x,y = int(operate[:-1]),int(operate[-1]) if operate[-1].isdigit() else operate[-1]
            if self.now_player:
                if not self.down_black_chess(x,y):
                    print("请把棋子下在有效的位置")
            else:
                if not self.down_white_chess(x,y):
                    print("请把棋子下在有效的位置")
        print(self.check_win())
        return self.check_win()
    
    def __get_info(self):
        return {
            "now_player":self.now_player,
            "step_log":self.step_log
        }

    def __get_obs(self):
        return self.chess_board

    def __get_done(self):
        return self.check_win() in ["Black wins","White wins"]
    
    def __get_reward(self,x,y,nwp):
        ct = 1 if nwp else 2
        reward = check_reward(self.chess_board,x,y,ct)
        return reward

    def reset(self) -> Tuple[Any, dict]:
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
        x,y = action
        self.step_log.append(action)
        is_valide = True
        self.now_down = [x,y]
        if self.now_player:
            is_valide = self.down_black_chess(x,y)
        else:
            is_valide = self.down_white_chess(x,y)

        obs = self.__get_obs()
        reward = self.__get_reward(x,y,not self.now_player) if is_valide else 0
        done = self.__get_done()
        Terminated = False
        info = self.__get_info()

        return obs,reward,done,Terminated,info





env = FiveInRow()
while True:
    ac = env.action_space.sample()
    obs,reward,done,Terminated,info=env.step(ac)
    env.render()
    print(reward)
    if done:
        break