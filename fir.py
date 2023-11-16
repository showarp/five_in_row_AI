import numpy as np

class FiveInRow:
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
    def __str__(self):
        separator_line = "  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"  # Separator line for better board visualization
        s = f"   A B C D E F G H I J K L M N O \n{separator_line}\n"

        for i in range(15):
            s += f"{i:>2}|"
            for j in range(15):
                if self.chess_board[i, j] == 1.0:
                    s += "○"  
                elif self.chess_board[i, j] == 2.0:
                    s += "●"  
                else:
                    s += " "  
                s += "|"
            s += "\n" + separator_line + "\n"

        return s
    def __parser_y__(self,y):
        if type(y)==int:
            return y
        else:
            return ord(y)-ord("A")
    def check_chess_valid(self,x,y):
        y = self.__parser_y__(y)
        return self.chess_board[x,y]==0 and 15>x>=0 and 15>y>=0
    
    def down_black_chess(self, x, y):
        y = self.__parser_y__(y)
        if not self.check_chess_valid(x,y):
            raise Exception("请把棋子落在合法的位置")
        self.chess_board[x, y] = 1.0
        self.now_player = not self.now_player

    def down_white_chess(self, x, y):
        y = self.__parser_y__(y)
        if not self.check_chess_valid(x,y):
            raise Exception("请把棋子落在合法的位置")
        self.chess_board[x, y] = 2.0
        self.now_player = not self.now_player

    def check_win(self):
        # Check rows
        for i in range(15):
            for j in range(11):
                if (
                    self.chess_board[i, j] == self.chess_board[i, j + 1] == self.chess_board[i, j + 2]
                    == self.chess_board[i, j + 3] == self.chess_board[i, j + 4]
                ) and (self.chess_board[i, j] != 0):
                    return "black win" if self.chess_board[i, j] == 1.0 else "white win"

        # Check columns
        for i in range(11):
            for j in range(15):
                if (
                    self.chess_board[i, j] == self.chess_board[i + 1, j] == self.chess_board[i + 2, j]
                    == self.chess_board[i + 3, j] == self.chess_board[i + 4, j]
                ) and (self.chess_board[i, j] != 0):
                    return "black win" if self.chess_board[i, j] == 1.0 else "white win"

        # Check diagonals
        for i in range(11):
            for j in range(11):
                if (
                    self.chess_board[i, j] == self.chess_board[i + 1, j + 1] == self.chess_board[i + 2, j + 2]
                    == self.chess_board[i + 3, j + 3] == self.chess_board[i + 4, j + 4]
                ) and (self.chess_board[i, j] != 0):
                    return "black win" if self.chess_board[i, j] == 1.0 else "white win"

                if (
                    self.chess_board[i + 4, j] == self.chess_board[i + 3, j + 1] == self.chess_board[i + 2, j + 2]
                    == self.chess_board[i + 1, j + 3] == self.chess_board[i, j + 4]
                ) and (self.chess_board[i + 4, j] != 0):
                    return "black win" if self.chess_board[i + 4, j] == 1.0 else "white win"
        return None
    
    def play(self):
        while self.check_win()==None:
            print(self)
            operate = input("请输入您的落子位置(例如3L):")
            x,y = int(operate[:-1]),operate[-1]
            if self.now_player:
                self.down_black_chess(x,y)
            else:
                self.down_white_chess(x,y)
        return self.check_win()
chess = FiveInRow()
print(chess.play())